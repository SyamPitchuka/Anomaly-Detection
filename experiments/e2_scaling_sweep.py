"""E2 on the cluster: strong scaling of score_job, measured as the time to drain a fixed Kafka backlog.

For every (executors, cores) config in --configs (default 1x2 2x2 3x2 4x2) it runs, --repeats times (default 3,
repetitions interleaved: rep 1 of every config, then rep 2, ...):
    bash infra/cluster/drain_test.sh -e E -c C -n <records> -m <maxOffsetsPerTrigger> -r <replicate>
                                     -P <topic partitions> --run-id <id> -o data/scaling
and reads the run's data/scaling/<id>.json (see infra/cluster/drain_monitor.py for the definitions).

Per config (mean over the drained repetitions, error bars min..max):
    records_per_s     backlog / drain time
    speed-up          records_per_s / records_per_s of the base config (fewest total cores)
    ideal             total cores / base total cores (linear scaling)
    efficiency        speed-up / ideal
Strong scaling replays --replicate (default 4) renamed copies of every machine (replay.py --replicate): SMD has
28 machines, i.e. 28 keys for the stateful scoring, which would cap the parallelism at 28 tasks; 4 copies give
112. The topic is recreated with --topic-partitions (default 32) partitions, so the Kafka read stage is not capped
at the 8 partitions of contracts/topics.yaml either. drain_test.sh sets spark.sql.shuffle.partitions to
2 x total cores. All three are stored per run and shown in the report table.

--weak turns it into weak scaling: each config replays `replicate = total cores / base total cores` renamed
copies of every machine (replay.py --replicate) and the backlog grows by the same factor, so the load per core
stays fixed. Ideal is then a constant drain time; speed-up and efficiency keep the same formulas (with the
backlog growing K-fold, records/s ratio / K = base drain time / drain time).

Output: docs/results/e2_cluster.md, e2_cluster.png (throughput vs total cores with the ideal line, speed-up,
efficiency) and e2_cluster.json (every run, rewritten after each run; --from-json redraws without running).

Run ON THE SPARK MASTER VM from the repo root ($REPO_DIR), as the 'spark' user:
    python experiments/e2_scaling_sweep.py
    python experiments/e2_scaling_sweep.py --configs 1x2 2x2 4x2 --repeats 3 --records 4000000
    python experiments/e2_scaling_sweep.py --weak --records 1000000
    python experiments/e2_scaling_sweep.py --dry-run                        # print the commands only
    python experiments/e2_scaling_sweep.py --from-json docs/results/e2_cluster.json   # redraw (any machine)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAIN = ROOT / "infra" / "cluster" / "drain_test.sh"
OUT_DIR = ROOT / "docs" / "results"
RUN_DIR = ROOT / "data" / "scaling"
NAME = "e2_cluster"
NAN = float("nan")


# ---- configs, commands ----------------------------------------------------------------------------------

def parse_config(text: str) -> tuple[int, int]:
    """'4x2' -> (4 executors, 2 cores each)."""
    e, sep, c = text.lower().partition("x")
    if not sep or not e.isdigit() or not c.isdigit() or int(e) < 1 or int(c) < 1:
        raise ValueError(f"config must look like <executors>x<cores>, e.g. 4x2; got {text!r}")
    return int(e), int(c)


def replicate_factor(config: tuple[int, int], configs: list[tuple[int, int]]) -> int:
    """Weak scaling: total cores of `config` / the smallest total over `configs` (must divide evenly)."""
    base = min(e * c for e, c in configs)
    total = config[0] * config[1]
    if total % base:
        raise ValueError(f"--weak needs every total core count to be a multiple of {base}; {config} has {total}")
    return total // base


def run_id(sweep_id: str, config: tuple[int, int], rep: int) -> str:
    return f"{sweep_id}_e{config[0]}c{config[1]}_rep{rep}"


def drain_command(config: tuple[int, int], records: int, max_offsets: int, rid: str, out: Path,
                  replicate: int | None = None, extra: list[str] = (), topic_partitions: int | None = None) -> list[str]:
    cmd = ["bash", DRAIN.as_posix(), "-e", str(config[0]), "-c", str(config[1]), "-n", str(records),
           "-m", str(max_offsets), "--run-id", rid, "-o", out.as_posix()]
    if replicate is not None:
        cmd += ["-r", str(replicate)]
    if topic_partitions is not None:
        cmd += ["-P", str(topic_partitions)]
    return cmd + list(extra)


def run_replicate(config: tuple[int, int], configs: list[tuple[int, int]], weak: bool,
                  replicate: int | None) -> int | None:
    """Copies per machine for one run: weak -> total cores / base total cores; strong -> --replicate."""
    return replicate_factor(config, configs) if weak else replicate


def schedule(configs: list[tuple[int, int]], repeats: int) -> list[tuple[int, tuple[int, int]]]:
    """(rep, config) in run order: repetitions interleaved so drift spreads over every config."""
    return [(rep, cfg) for rep in range(1, repeats + 1) for cfg in configs]


# ---- aggregation ----------------------------------------------------------------------------------------

def _num(v) -> float:
    return NAN if v is None else float(v)


def _stats(values: list[float]) -> tuple[float, float, float]:
    vals = [v for v in values if not math.isnan(v)]
    return (sum(vals) / len(vals), min(vals), max(vals)) if vals else (NAN, NAN, NAN)


def _setting(rows: list[dict], key: str) -> str:
    """The value(s) of a per-run setting over a config's runs: '32', '16/32' when they differ, '–' when unknown."""
    vals = sorted({str(r[key]) for r in rows if r.get(key) not in (None, "")}, key=lambda v: (len(v), v))
    return "/".join(vals) if vals else "–"


def aggregate(runs: list[dict]) -> list[dict]:
    """Run results (drain_monitor JSON) -> one row per (executors, cores), sorted by total cores then cores per
    executor. Only status 'drained' runs count; speed-up / efficiency are relative to the base config (fewest
    total cores; ties -> fewer executors), with min..max error bars."""
    groups: dict[tuple[int, int], list[dict]] = {}
    for r in runs:
        groups.setdefault((int(r["executors"]), int(r["cores"])), []).append(r)
    table = []
    for (e, c), rs in groups.items():
        ok = [r for r in rs if r.get("status") == "drained"]
        row = {"executors": e, "cores": c, "total_cores": e * c, "runs": len(rs), "ok": len(ok),
               "records": max((int(r["records"]) for r in ok), default=0)}
        for key in ("replicate", "topic_partitions", "shuffle_partitions"):
            row[key] = _setting(ok or rs, key)
        for key in ("records_per_s", "steady_records_per_s", "drain_s", "batches", "batch_mean_s", "batch_p95_s",
                    "first_batch_s"):
            row[key], row[key + "_min"], row[key + "_max"] = _stats([_num(r.get(key)) for r in ok])
        seen = [r.get("executors_seen") for r in ok if r.get("executors_seen") is not None]
        row["executors_seen"] = (min(seen), max(seen)) if seen else None
        table.append(row)
    table.sort(key=lambda r: (r["total_cores"], r["executors"]))
    base = next((r for r in table if r["ok"]), None)
    for r in table:
        r["ideal"] = r["total_cores"] / base["total_cores"] if base else NAN
        b = base["records_per_s"] if base else NAN
        for suffix in ("", "_min", "_max"):
            r["speedup" + suffix] = r["records_per_s" + suffix] / b if b and not math.isnan(b) else NAN
            r["efficiency" + suffix] = r["speedup" + suffix] / r["ideal"] if r["ideal"] else NAN
    return table


# ---- report ---------------------------------------------------------------------------------------------

def _fmt(v, spec: str) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def _rng(r: dict, key: str, spec: str) -> str:
    """'mean (min–max)' or just the mean when there is one value."""
    mean, lo, hi = r[key], r[key + "_min"], r[key + "_max"]
    if math.isnan(mean):
        return "–"
    if lo == hi:
        return format(mean, spec)
    return f"{format(mean, spec)} ({format(lo, spec)}–{format(hi, spec)})"


def markdown_table(table: list[dict]) -> str:
    head = ["executors × cores", "total cores", "replicate", "topic partitions", "shuffle partitions",
            "runs drained", "drain time (s)", "records/s", "steady records/s",
            "speed-up", "ideal", "efficiency", "batches", "batch mean (s)", "batch p95 (s)", "executors seen"]
    lines = ["| " + " | ".join(head) + " |", "| --- |" + " ---: |" * (len(head) - 1)]
    for r in table:
        seen = r["executors_seen"]
        seen_s = "–" if seen is None else (str(seen[0]) if seen[0] == seen[1] else f"{seen[0]}–{seen[1]}")
        cells = [f"{r['executors']} × {r['cores']}", r["total_cores"], r["replicate"], r["topic_partitions"],
                 r["shuffle_partitions"], f"{r['ok']}/{r['runs']}",
                 _rng(r, "drain_s", ".1f"), _rng(r, "records_per_s", ",.0f"), _rng(r, "steady_records_per_s", ",.0f"),
                 _rng(r, "speedup", ".2f"), _fmt(r["ideal"], ".2f"), _rng(r, "efficiency", ".2f"),
                 _fmt(r["batches"], ".0f"), _fmt(r["batch_mean_s"], ".2f"), _fmt(r["batch_p95_s"], ".2f"), seen_s]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def runs_table(runs: list[dict]) -> str:
    head = ["run", "status", "replicate", "topic partitions", "shuffle partitions", "records", "drain (s)", "records/s", "batches", "batch mean (s)", "batch p95 (s)",
            "executors seen"]
    lines = ["| " + " | ".join(head) + " |", "| --- | --- |" + " ---: |" * (len(head) - 2)]
    for r in runs:
        cells = [r["run_id"], r.get("status", "?"),
                 *(r.get(k) if r.get(k) not in (None, "") else "–"
                   for k in ("replicate", "topic_partitions", "shuffle_partitions")),
                 f"{int(r.get('records') or 0):,}", _fmt(_num(r.get("drain_s")), ".1f"),
                 _fmt(_num(r.get("records_per_s")), ",.0f"), r.get("batches", "–"),
                 _fmt(_num(r.get("batch_mean_s")), ".2f"), _fmt(_num(r.get("batch_p95_s")), ".2f"),
                 r.get("executors_seen") if r.get("executors_seen") is not None else "–"]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (fixed order)
MARKERS = ["o", "s", "^", "D"]


def plot(table: list[dict], path: Path, weak: bool = False) -> None:
    """Three panels over total cores: records/s with the ideal line from the base config, speed-up vs ideal,
    efficiency. One series per cores-per-executor value; error bars = min..max of the repetitions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    done = [r for r in table if r["ok"]]
    if not done:
        raise ValueError("no drained runs to plot")
    base = done[0]
    xs_all = sorted({r["total_cores"] for r in done})
    per_exec = sorted({r["cores"] for r in done})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)
    panels = [("Throughput (records/s)", "records_per_s",
               [base["records_per_s"] * x / base["total_cores"] for x in xs_all], "{:,.0f}"),
              ("Speed-up vs " + f"{base['executors']}×{base['cores']}", "speedup",
               [x / base["total_cores"] for x in xs_all], "{:.2f}"),
              ("Parallel efficiency (speed-up / ideal)", "efficiency", [1.0] * len(xs_all), "{:.2f}")]
    for ax, (title, key, ideal, fmt) in zip(axes, panels):
        ax.plot(xs_all, ideal, linestyle="--", linewidth=1.5, color=muted, label="ideal (linear)")
        top = max(ideal)
        for i, c in enumerate(per_exec):
            rows = [r for r in done if r["cores"] == c]
            xs = [r["total_cores"] for r in rows]
            ys = [r[key] for r in rows]
            err = [[y - r[key + "_min"] for y, r in zip(ys, rows)], [r[key + "_max"] - y for y, r in zip(ys, rows)]]
            color, marker = SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]
            ax.errorbar(xs, ys, yerr=err, linewidth=2, color=color, marker=marker, markersize=7,
                        markeredgecolor="white", markeredgewidth=1.5, capsize=3, elinewidth=1.2,
                        label=f"{c} core{'s' if c > 1 else ''} per executor")
            for x, y, r in zip(xs, ys, rows):
                ax.annotate(fmt.format(y), (x, r[key + "_max"]), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7.5, color=ink)
            top = max(top, max(r[key + "_max"] for r in rows))
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xticks(xs_all, [str(x) for x in xs_all])
        ax.set_xlabel("total executor cores", color=muted)
        ax.set_ylim(bottom=0, top=top * 1.18)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.legend(frameon=False, fontsize=8, labelcolor=muted)
    kind = "weak scaling (load grows with the cores)" if weak else "strong scaling (fixed backlog)"
    fig.suptitle(f"E2 (cluster): score_job backlog drain, {kind}; bars = min–max of the repetitions",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def default_notes(table: list[dict]) -> str:
    done = [r for r in table if r["ok"]]
    if not done:
        return "No run drained the backlog."
    best = max(done, key=lambda r: r["records_per_s"])
    last = done[-1]
    return (f"Highest throughput: {best['executors']} × {best['cores']} ({best['total_cores']} cores), "
            f"{best['records_per_s']:,.0f} records/s, {best['speedup']:.2f}× the base config. "
            f"At {last['total_cores']} cores the parallel efficiency is {last['efficiency']:.2f}.")


def write_markdown(table: list[dict], runs: list[dict], meta: dict, notes: str, path: Path) -> None:
    weak = meta.get("weak", False)
    load = (f"weak scaling: config with T total cores replays `replicate = T / {table[0]['total_cores']}` copies of "
            f"every machine and a backlog of {meta['records']:,} × replicate records"
            if weak else f"strong scaling: the same backlog of {meta['records']:,} records for every config, "
                         f"`replicate = {meta.get('replicate')}` copies of every machine "
                         f"({28 * (meta.get('replicate') or 1)} keys)")
    settings = (f"Topic `telemetry.smd` recreated with **{meta.get('topic_partitions', '–')} partitions**; "
                f"`spark.sql.shuffle.partitions` = 2 × total cores (unless overridden). The per-config values of "
                f"replicate, topic partitions and shuffle partitions are in the table.")
    text = f"""# E2 (cluster): scaling of the stream scoring job (backlog drain)

Setup: for each config, `infra/cluster/drain_test.sh` recreates `telemetry.smd`, replays the backlog at max speed
with score_job **not** running, then starts score_job (`submit.sh -e E -c C`, `--reset`, trigger `0 seconds`,
`maxOffsetsPerTrigger = {meta['max_offsets_per_trigger']:,}`) and times how long it takes to consume it; {load}.
{settings}
{meta['repeats']} repetitions per config (interleaved), mean with min–max in brackets.
Cluster: {meta.get('cluster', 'see infra/cluster/cluster.env')}. Measured {meta['date']} with
`python experiments/e2_scaling_sweep.py {meta.get('argv', '')}`.

- **drain time**: start of the first micro-batch with input rows → end of the last one (job start-up excluded);
  **records/s** = backlog / drain time; **steady records/s** = the same without the first batch (it pays the
  Python workers' torch import and model load).
- **speed-up** = records/s / records/s of {table[0]['executors']} × {table[0]['cores']};
  **ideal** = total cores / {table[0]['total_cores']}; **efficiency** = speed-up / ideal.
- **batches**, **batch mean / p95**: micro-batches with input rows and their `triggerExecution` durations;
  **executors seen**: distinct executor ids in the driver's Spark UI during the run.

{markdown_table(table)}

![Throughput, speed-up and efficiency vs total cores]({path.with_suffix('.png').name})

{notes}

## Every run

{runs_table(runs)}
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def write_outputs(runs: list[dict], meta: dict, out_dir: Path, notes: str | None = None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{NAME}.json").write_text(json.dumps({"meta": meta, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    table = aggregate(runs)
    if any(r["ok"] for r in table):
        plot(table, out_dir / f"{NAME}.png", weak=meta.get("weak", False))
        write_markdown(table, runs, meta, notes or default_notes(table), out_dir / f"{NAME}.md")
    return table


# ---- main -----------------------------------------------------------------------------------------------

def _load_run(path: Path, rid: str, cfg: tuple[int, int], rc: int) -> dict:
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        r.pop("batches_detail", None)
        return r
    return {"run_id": rid, "status": f"failed (exit {rc}, no result file)", "executors": cfg[0], "cores": cfg[1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", default=["1x2", "2x2", "3x2", "4x2"], help="<executors>x<cores> ...")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--records", type=int, default=2_000_000, help="backlog per run (weak: of the base config)")
    ap.add_argument("--max-offsets-per-trigger", type=int, default=200_000)
    ap.add_argument("--weak", action="store_true", help="weak scaling: replicate machines by total cores / base")
    ap.add_argument("--replicate", type=int, default=4,
                    help="strong scaling: copies of every machine (default 4: 28 machines -> 112 keys); ignored with --weak")
    ap.add_argument("--topic-partitions", type=int, default=32, help="partitions of the recreated telemetry.smd")
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR, help="drain_test.sh result directory")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where e2_cluster.md/.png/.json go")
    ap.add_argument("--notes", type=Path, default=None, help="Markdown analysis put under the table")
    ap.add_argument("--cluster", default=None, help="one line describing the VMs, shown in the report")
    ap.add_argument("--from-json", type=Path, default=None, help="redraw from a saved e2_cluster.json")
    ap.add_argument("--dry-run", action="store_true", help="print the drain_test.sh commands and exit")
    ap.add_argument("drain_args", nargs="*", help="after --: extra drain_test.sh options (e.g. -p 32)")
    args = ap.parse_args(argv)
    notes = args.notes.read_text(encoding="utf-8") if args.notes else None

    if args.from_json:
        saved = json.loads(args.from_json.read_text(encoding="utf-8"))
        table = write_outputs(saved["runs"], saved["meta"], args.out_dir, notes)
        print(markdown_table(table))
        return 0

    if args.replicate < 1 or args.topic_partitions < 1:
        ap.error("--replicate and --topic-partitions must be >= 1")
    configs = [parse_config(c) for c in args.configs]
    sweep_id = "e2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = []
    for rep, cfg in schedule(configs, args.repeats):
        k = run_replicate(cfg, configs, args.weak, args.replicate)
        records = args.records * k if args.weak else args.records
        rid = run_id(sweep_id, cfg, rep)
        plan.append((rid, cfg, drain_command(cfg, records, args.max_offsets_per_trigger, rid, args.run_dir, k,
                                             args.drain_args, args.topic_partitions)))
    if args.dry_run:
        for _rid, _cfg, cmd in plan:
            print(" ".join(cmd))
        return 0

    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "sweep_id": sweep_id,
            "configs": args.configs, "repeats": args.repeats, "records": args.records,
            "max_offsets_per_trigger": args.max_offsets_per_trigger, "weak": args.weak,
            "replicate": None if args.weak else args.replicate, "topic_partitions": args.topic_partitions,
            "cluster": args.cluster, "argv": " ".join(argv if argv is not None else sys.argv[1:])}
    runs: list[dict] = []
    for i, (rid, cfg, cmd) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {rid}: {' '.join(cmd)}", flush=True)
        t = time.monotonic()
        rc = subprocess.call(cmd, cwd=ROOT)
        run = _load_run(args.run_dir / f"{rid}.json", rid, cfg, rc)
        runs.append(run)
        print(f"    {run.get('status')} in {time.monotonic() - t:.0f} s: "
              f"{_fmt(_num(run.get('records_per_s')), ',.0f')} records/s", flush=True)
        write_outputs(runs, meta, args.out_dir, notes)
    table = aggregate(runs)
    print(markdown_table(table))
    print(f"written {args.out_dir / (NAME + '.md')}")
    return 0 if all(r.get("status") == "drained" for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
