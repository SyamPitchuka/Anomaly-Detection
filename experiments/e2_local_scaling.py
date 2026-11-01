"""E2 fallback: strong scaling of the stream scoring job on one machine (local[n]).

The cluster version of E2 replaces this when the cloud VMs are available.

For each n in --cores (default 1 2 4 8; values above os.cpu_count() are skipped) it runs, --repeats
times (default 2, settings interleaved), on the same input (all SMD machines in data/raw/source=smd):
    python src/streaming/score_job.py --input files --once --reset --files-per-trigger 8
        --master local[n] --shuffle-partitions 2n
and parses the per-batch lines the job prints. Per run:
    wall_s         process wall time (JVM start + all batches + shutdown)
    batch_s        sum of the micro-batch durations (triggerExecution)
    rps            input records / batch_s
    peak_state     largest "state rows" over the batches
    free_before_mb available RAM just before the run
    peak_rss_mb    peak total RSS of the job's process tree (driver Python, JVM, Python workers), sampled
                   every second with psutil (Windows working sets; shared DLL pages count once per process)
    min_free_mb    lowest available RAM seen during the run
If available RAM drops below --min-free-gb (default 1), the run is killed and recorded as out of memory.
Each metric is the median over the completed repeats (with 2: their mean). speed-up = rps_n / rps_1,
efficiency = speed-up / n.

Why 8 files per trigger: with the default 1, every micro-batch holds one machine, i.e. one key of the
stateful operator, so only one task scores and extra cores sit idle. 8 machines per batch give the
applyInPandasWithState stage 8 keys to spread over the cores (tools/check_stream_parity.py passes with it).

Output: docs/results/e2_local.md, docs/results/e2_local.png (throughput and efficiency vs cores) and
docs/results/e2_local.json (every run, rewritten after each run; --from-json redraws without re-running).
Earlier results given with --extra-json (default: e2_local_fpt1.json and e2_local_before_fix.json in
docs/results, when present) appear as small tables before the main one, each with meta.title and
meta.explanation. Analysis text under the main table comes from --notes (Markdown), else one line.

Run from the repo root:
    python experiments/e2_local_scaling.py
    python experiments/e2_local_scaling.py --cores 1 2 --repeats 1 --machines 2   # quick try
    python experiments/e2_local_scaling.py --from-json docs/results/e2_local.json  # redraw only
"""
import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE_JOB = Path("src") / "streaming" / "score_job.py"
OUT_DIR = ROOT / "docs" / "results"
EXTRA_JSON = [OUT_DIR / "e2_local_fpt1.json", OUT_DIR / "e2_local_before_fix.json"]
OOM = "out of memory"
MB = 2 ** 20

BATCH_RE = re.compile(r"^batch (\d+): input rows (\d+), .*duration (\d+) ms, state rows (\d+)", re.M)
METRICS = ("wall_s", "batch_s", "records", "rps", "peak_state", "free_before_mb", "peak_rss_mb", "min_free_mb")


def parse_batches(log: str) -> list[dict]:
    """score_job stdout -> [{batch, rows, ms, state}] in batch order."""
    return [{"batch": int(b), "rows": int(r), "ms": int(ms), "state": int(s)}
            for b, r, ms, s in BATCH_RE.findall(log)]


def summarise_run(batches: list[dict], wall_s: float) -> dict:
    rows = sum(b["rows"] for b in batches)
    batch_s = sum(b["ms"] for b in batches) / 1000
    return {"wall_s": wall_s, "batch_s": batch_s, "records": rows, "batches": len(batches),
            "rps": rows / batch_s if batch_s else float("nan"),
            "peak_state": max((b["state"] for b in batches), default=0)}


def _median(values) -> float:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(values) if values else float("nan")


def aggregate(runs: dict[int, list[dict]]) -> list[dict]:
    """{cores: [run summaries]} -> one row per cores value (medians over completed runs) with speed-up and
    efficiency. A setting whose runs all ran out of memory gets status OOM and NaN metrics."""
    table = []
    for n in sorted(runs):
        ok = [r for r in runs[n] if r.get("status", "ok") == "ok"]
        row = {"cores": n, "shuffle_partitions": 2 * n, "runs": len(runs[n]), "ok": len(ok),
               "status": "ok" if ok else OOM}
        for k in METRICS:
            row[k] = _median(r.get(k) for r in ok)
        row["free_before_mb"] = _median(r.get("free_before_mb") for r in runs[n])  # measured before any run
        row["peak_rss_all_mb"] = max((r.get("peak_rss_mb") or float("nan")) for r in runs[n])  # incl. OOM runs
        table.append(row)
    base = next((r["rps"] for r in table if r["cores"] == 1 and r["status"] == "ok"), None)
    for r in table:
        r["speedup"] = r["rps"] / base if base else float("nan")
        r["efficiency"] = r["speedup"] / r["cores"]
    return table


def _fmt(v: float, spec: str) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def markdown_table(table: list[dict], memory: bool = True) -> str:
    head = ["cores", "shuffle partitions", "wall time (s)", "batch time (s)", "records/s", "peak state rows",
            "speed-up", "efficiency"]
    if memory:
        head += ["peak memory (GB)", "free RAM before (GB)", "runs completed"]
    lines = ["| " + " | ".join(head) + " |", "|" + " ---: |" * len(head)]
    for r in table:
        if r.get("status") == OOM:
            cells = [r["cores"], r["shuffle_partitions"], "–", "–", OOM, "–", "–", "–"]
        else:
            cells = [r["cores"], r["shuffle_partitions"], _fmt(r["wall_s"], ".1f"), _fmt(r["batch_s"], ".1f"),
                     _fmt(r["rps"], ",.0f"), _fmt(r["peak_state"], ".0f"), _fmt(r["speedup"], ".2f"),
                     _fmt(r["efficiency"], ".2f")]
        if memory:
            peak = r["peak_rss_mb"] if r.get("status") != OOM else r.get("peak_rss_all_mb", float("nan"))
            cells += [_fmt(peak / 1024, ".2f"), _fmt(r["free_before_mb"] / 1024, ".1f"),
                      f"{r['ok']}/{r['runs']}" + (f" ({r['runs'] - r['ok']} {OOM})" if r["ok"] < r["runs"] else "")]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def _tree(proc):
    import psutil
    try:
        root = psutil.Process(proc.pid)
        return [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _tree_rss(proc) -> int:
    import psutil
    total = 0
    for p in _tree(proc):
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _kill_tree(proc) -> None:
    import psutil
    for p in reversed(_tree(proc)):
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def run_once(n: int, machines: int | None, files_per_trigger: int, min_free_mb: float) -> tuple[dict, str]:
    import psutil
    cmd = [sys.executable, SCORE_JOB.as_posix(), "--input", "files", "--once", "--reset",
           "--files-per-trigger", str(files_per_trigger),
           "--master", f"local[{n}]", "--shuffle-partitions", str(2 * n)]
    if machines:
        cmd += ["--machines", str(machines)]
    free_before = psutil.virtual_memory().available / MB
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    mem = {"peak_rss": 0, "min_free": free_before, "oom": False}
    done = threading.Event()

    def sample():
        while not done.wait(1.0):
            mem["peak_rss"] = max(mem["peak_rss"], _tree_rss(proc))
            free = psutil.virtual_memory().available / MB
            mem["min_free"] = min(mem["min_free"], free)
            if free < min_free_mb and not mem["oom"]:
                mem["oom"] = True
                _kill_tree(proc)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    out, err = proc.communicate()
    done.set()
    sampler.join()
    wall = time.perf_counter() - t0
    batches = parse_batches(out)
    extra = {"free_before_mb": free_before, "peak_rss_mb": mem["peak_rss"] / MB, "min_free_mb": mem["min_free"]}
    if mem["oom"]:
        return {"status": OOM, "wall_s": wall, "batches_done": len(batches), **extra}, out + err
    if proc.returncode != 0 or not batches:
        raise RuntimeError(f"score_job failed for local[{n}] (exit {proc.returncode}):\n{(out + err)[-3000:]}")
    return {"status": "ok", **summarise_run(batches, wall), **extra, "batches_detail": batches}, out + err


def plot(table: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, series = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    done = [r for r in table if r["status"] == "ok"]
    cores = [r["cores"] for r in done]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    panels = [("Throughput (records/s)", [r["rps"] for r in done],
               [done[0]["rps"] * c / done[0]["cores"] for c in cores], "{:,.0f}"),
              ("Parallel efficiency (speed-up / cores)", [r["efficiency"] for r in done],
               [1.0] * len(cores), "{:.2f}")]
    for ax, (title, ys, ideal, fmt) in zip(axes, panels):
        ax.plot(cores, ideal, linestyle="--", linewidth=1.5, color=muted, label="ideal (linear)")
        ax.plot(cores, ys, linewidth=2, color=series, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=2, label="measured")
        for x, y in zip(cores, ys):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=ink)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_xscale("log", base=2)
        all_cores = [r["cores"] for r in table]
        ax.set_xticks(all_cores, [f"{c}\n({OOM})" if c not in cores else str(c) for c in all_cores])
        ax.minorticks_off()
        ax.set_xlabel("cores (Spark local[n])", color=muted)
        ax.set_ylim(bottom=0, top=max(max(ys), max(ideal)) * 1.15)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.legend(frameon=False, fontsize=8, labelcolor=muted)
    fig.suptitle("E2 (local fallback): stream scoring job, strong scaling on one machine",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def extra_section(saved: dict) -> str:
    """A saved earlier result as a small table: meta has title, explanation, and the setup fields."""
    meta, table = saved["meta"], aggregate({int(n): rs for n, rs in saved["runs"].items()})
    memory = any(not math.isnan(r["peak_rss_mb"]) for r in table)
    return f"""## {meta['title']}

{meta['machines']} machines ({meta['records']:,} records), `--files-per-trigger {meta['files_per_trigger']}`,
{meta['repeats']} run(s) per setting, measured {meta['date']}.

{markdown_table(table, memory)}

{meta['explanation'].strip()}

"""


def write_markdown(table: list[dict], meta: dict, notes: str, path: Path, extras: list[dict] = ()) -> None:
    main_title = "## Main result" if extras else ""
    text = f"""# E2 (local fallback): strong scaling of the stream scoring job

> **Fallback.** This is the single-machine version of E2, run in case the cloud VMs are late.
> The cluster version of E2 replaces it when the VMs are available.

{"".join(extra_section(e) for e in extras)}{main_title}

Setup: `python src/streaming/score_job.py --input files --once --reset --files-per-trigger {meta['files_per_trigger']}`
on all {meta['machines']} SMD machines ({meta['records']:,} records, {meta['batches']} micro-batches),
Spark master `local[n]`, `spark.sql.shuffle.partitions = 2n`, one math thread per Python worker,
{meta['repeats']} runs per setting, median of the completed runs. Machine: {meta['host']}
({meta['cpu_count']} logical cores, {meta['ram_gb']:.1f} GB RAM). Measured {meta['date']} with
`python experiments/e2_local_scaling.py`; a run is stopped and counted as {OOM} if free RAM drops below
{meta['min_free_gb']:g} GB.

- **wall time**: whole process (JVM start, all micro-batches, shutdown);
  **batch time**: sum of micro-batch durations;
  **records/s** = records / batch time; **peak state rows** = largest state store size (one row per machine).
- **speed-up** = records/s at n cores / records/s at 1 core; **efficiency** = speed-up / n.
- **peak memory**: peak total RSS of the job's processes (driver Python, JVM, Python workers), sampled every
  second; **free RAM before**: available RAM when the run started.

{markdown_table(table)}

![Throughput and efficiency vs cores](e2_local.png)

{notes}
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def default_notes(table: list[dict]) -> str:
    done = [r for r in table if r["status"] == "ok"]
    best = max(done, key=lambda r: r["rps"])
    return (f"Throughput peaks at {best['cores']} core(s) ({best['speedup']:.2f}x the 1-core rate, "
            f"efficiency {best['efficiency']:.2f}).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--files-per-trigger", type=int, default=8, help="machine files per micro-batch")
    parser.add_argument("--min-free-gb", type=float, default=1.0,
                        help="stop a run (recorded as out of memory) when available RAM drops below this")
    parser.add_argument("--extra-json", type=Path, nargs="*", default=EXTRA_JSON,
                        help="earlier results shown as small tables before the main one (missing files skipped)")
    parser.add_argument("--machines", type=int, default=None, help="only the first N machine files (quick try)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--from-json", type=Path, default=None, help="redraw from a saved e2_local.json")
    parser.add_argument("--notes", type=Path, default=None,
                        help="Markdown file with the analysis paragraph to put under the main table")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "e2_local.json"

    if args.from_json:
        saved = json.loads(args.from_json.read_text(encoding="utf-8"))
        runs = {int(n): rs for n, rs in saved["runs"].items()}
        meta = saved["meta"]
    else:
        import psutil
        cpu = os.cpu_count() or 1
        cores = [n for n in args.cores if n <= cpu]
        skipped = [n for n in args.cores if n > cpu]
        if skipped:
            print(f"skipping cores {skipped}: only {cpu} logical cores", flush=True)
        meta = {"repeats": args.repeats, "files_per_trigger": args.files_per_trigger,
                "min_free_gb": args.min_free_gb, "cpu_count": cpu,
                "ram_gb": psutil.virtual_memory().total / 2 ** 30,
                "host": f"{platform.system()}, {platform.processor()}", "date": time.strftime("%Y-%m-%d")}
        runs: dict[int, list[dict]] = {}
        for rep in range(args.repeats):          # interleave settings so drift hits all of them alike
            for n in cores:
                print(f"run {rep + 1}/{args.repeats}: local[{n}], shuffle partitions {2 * n} ...", flush=True)
                s, _ = run_once(n, args.machines, args.files_per_trigger, args.min_free_gb * 1024)
                runs.setdefault(n, []).append(s)
                mem = (f"peak RSS {s['peak_rss_mb'] / 1024:.2f} GB, free before {s['free_before_mb'] / 1024:.1f} GB, "
                       f"min free {s['min_free_mb'] / 1024:.1f} GB")
                if s["status"] == OOM:
                    print(f"  {OOM} after {s['wall_s']:.0f} s ({s['batches_done']} batches), {mem}", flush=True)
                else:
                    print(f"  wall {s['wall_s']:.1f} s, batch {s['batch_s']:.1f} s, {s['rps']:,.0f} records/s, "
                          f"peak state rows {s['peak_state']}, {mem}", flush=True)
                    meta.update(machines=s["peak_state"], records=s["records"], batches=s["batches"])
                json_path.write_text(json.dumps({"meta": meta, "runs": {str(k): v for k, v in runs.items()}},
                                                indent=1), encoding="utf-8", newline="\n")

    table = aggregate(runs)
    if not any(r["status"] == "ok" for r in table):
        print(f"FAIL every run ran {OOM}; nothing to tabulate (runs saved in {json_path.as_posix()})")
        return 1
    notes = args.notes.read_text(encoding="utf-8").strip() if args.notes else default_notes(table)
    extras = [json.loads(p.read_text(encoding="utf-8")) for p in (args.extra_json or []) if p.exists()]
    write_markdown(table, meta, notes, args.out_dir / "e2_local.md", extras)
    plot(table, args.out_dir / "e2_local.png")
    print(markdown_table(table))
    print(f"wrote {(args.out_dir / 'e2_local.md').as_posix()} and e2_local.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
