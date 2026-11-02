"""End-to-end latency of the Kafka -> score_job path, per replay run: table + plot.

Run from the repo root (after experiments/e2e_latency_runs.py has filled data/latency_runs/):
    python src/streaming/latency_report.py                       # every run under data/latency_runs
    python src/streaming/latency_report.py --runs speedup_100 speedup_500

Each run folder data/latency_runs/<run>/ holds
    meta.json   speedup, limit, trigger, records sent, status, notes
    latency/    score_job's latency side output (copied from data/latency_stream/source=smd)
    progress.jsonl  score_job's query progress per micro-batch (for the batch run times)
    lag.csv     t_s, high (sum of broker end offsets), committed (sum of offsets of the newest batch committed
                in the Spark checkpoint), lag = high - committed, free_mb; sampled every few seconds

Per run, over every emitted window (train and test):
    total       emit_ts_ms - max_ingest_ts_ms: newest record of the window sent -> window written
    processing  emit_ts_ms - batch_max_ingest_ts_ms: newest record of that machine in the micro-batch that
                closed the window sent -> window written (micro-batch run time plus the time the last records
                waited for the trigger)
    wait        batch_max_ingest_ts_ms - max_ingest_ts_ms: wall time between the window's newest record and
                the records that pushed the per-machine watermark past the window end
    wait_theory 11 min / speedup: a window [s, s+5) has its newest record at s+4 min and is complete once a
                record at s+15 min arrives (10-minute watermark); 0 for speedup 0 (as fast as possible)
    batch       triggerExecution of the micro-batches that had input rows (from progress.jsonl)
p50 / p95 / max for each, and the peak consumer lag.

Output: docs/results/e2e_latency.md and docs/results/e2e_latency.png.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS_DIR = ROOT / "data" / "latency_runs"
OUT_MD = ROOT / "docs" / "results" / "e2e_latency.md"
OUT_PNG = ROOT / "docs" / "results" / "e2e_latency.png"
GOAL_S = 10.0
WAIT_EVENT_MIN = 11          # newest record of a window (s+4) -> record that closes it (s+15)
# validated default categorical order (dataviz skill palette, light mode), assigned per run in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- inputs

def load_latency(folder: Path) -> pd.DataFrame:
    files = sorted(Path(folder).glob("batch-*.parquet"))
    if not files:
        return pd.DataFrame(columns=["entity_id", "window_start", "max_ingest_ts_ms", "batch_max_ingest_ts_ms",
                                     "emit_ts_ms", "batch_id"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def committed_offsets(checkpoint: Path, topic: str) -> dict[int, int]:
    """partition -> end offset of the newest micro-batch committed in a Spark Kafka checkpoint ({} if none)."""
    commits = Path(checkpoint) / "commits"
    ids = sorted((int(p.name) for p in commits.glob("*") if p.name.isdigit()), reverse=True) \
        if commits.exists() else []
    for i in ids:
        path = Path(checkpoint) / "offsets" / str(i)
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines[2:]:
                try:
                    offsets = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(offsets, dict) and topic in offsets:
                    return {int(k): int(v) for k, v in offsets[topic].items()}
    return {}


def lag(high: dict[int, int], committed: dict[int, int]) -> int:
    """Records in the topic not yet processed by a committed micro-batch."""
    return int(sum(max(h - committed.get(p, 0), 0) for p, h in high.items()))


# ---------------------------------------------------------------- summary

def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarise(df: pd.DataFrame, speedup: float) -> dict:
    d = df.dropna(subset=["max_ingest_ts_ms", "emit_ts_ms"])
    total = (d["emit_ts_ms"] - d["max_ingest_ts_ms"]).to_numpy(dtype=float) / 1000
    b = d.dropna(subset=["batch_max_ingest_ts_ms"])
    proc = (b["emit_ts_ms"] - b["batch_max_ingest_ts_ms"]).to_numpy(dtype=float) / 1000
    wait = (b["batch_max_ingest_ts_ms"] - b["max_ingest_ts_ms"]).to_numpy(dtype=float) / 1000
    out = {"windows": int(len(d)), "batches": int(d["batch_id"].nunique()) if len(d) else 0,
           "wait_theory_s": WAIT_EVENT_MIN * 60 / speedup if speedup > 0 else 0.0}
    for name, x in (("total", total), ("proc", proc), ("wait", wait)):
        out[f"{name}_p50_s"], out[f"{name}_p95_s"] = _pct(x, 50), _pct(x, 95)
        out[f"{name}_max_s"] = float(x.max()) if len(x) else float("nan")
    return out


def batch_seconds(progress: Path) -> np.ndarray:
    """Run time (s) of every micro-batch with input rows, from score_job's progress.jsonl."""
    if not Path(progress).exists():
        return np.empty(0)
    out = []
    for line in Path(progress).read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        if p.get("numInputRows", 0):
            out.append(p.get("durationMs", {}).get("triggerExecution", 0) / 1000)
    return np.asarray(out, dtype=float)


def load_run(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    df = load_latency(folder / "latency")
    lag_path = folder / "lag.csv"
    lag_df = pd.read_csv(lag_path) if lag_path.exists() else pd.DataFrame(columns=["t_s", "lag"])
    s = summarise(df, float(meta.get("speedup", 0)))
    s["peak_lag"] = int(lag_df["lag"].max()) if len(lag_df) else None
    bs = batch_seconds(folder / "progress.jsonl")
    s["batch_p50_s"], s["batch_max_s"] = _pct(bs, 50), (float(bs.max()) if len(bs) else float("nan"))
    return {"name": folder.name, "meta": meta, "df": df, "lag": lag_df, "summary": s}


# ---------------------------------------------------------------- output

def _f(v, spec=".1f") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def speed_label(meta: dict) -> str:
    sp = float(meta.get("speedup", 0))
    label = "max (0)" if sp == 0 else f"{sp:g}x"
    trigger = meta.get("trigger", "30 seconds")
    return label + (f", trigger {trigger}" if trigger != "30 seconds" else "")


def markdown_table(runs: list[dict]) -> str:
    head = ("| run | speed-up | master | status | records in topic | windows | total p50 / p95 / max (s) | processing p50 / p95 / max (s) "
            "| batch run time p50 / max (s) | watermark wait p50 / p95 (s) | wait 11 min / speed-up (s) "
            "| peak lag (records) | p95 total < 10 s | p95 processing < 10 s |\n"
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |")
    lines = [head]
    for r in runs:
        s, m = r["summary"], r["meta"]
        ok = "–" if not s["windows"] else ("yes" if s["total_p95_s"] < GOAL_S else "no")
        ok_proc = "–" if not s["windows"] else ("yes" if s["proc_p95_s"] < GOAL_S else "no")
        # records the producer got into the topic: replay's "done" count, else the last broker end offset
        records = m.get("records_sent") or (int(r["lag"]["high"].max()) if "high" in r["lag"] and len(r["lag"])
                                            else 0)
        status = m.get("status", "ok").split(":")[0]
        lines.append(
            f"| {r['name']} | {speed_label(m)} | {m.get('master', '–')} | {status} | {records:,} | {s['windows']:,} "
            f"| {_f(s['total_p50_s'])} / {_f(s['total_p95_s'])} / {_f(s['total_max_s'])} "
            f"| {_f(s['proc_p50_s'])} / {_f(s['proc_p95_s'])} / {_f(s['proc_max_s'])} "
            f"| {_f(s.get('batch_p50_s'))} / {_f(s.get('batch_max_s'))} "
            f"| {_f(s['wait_p50_s'])} / {_f(s['wait_p95_s'])} | {_f(s['wait_theory_s'], '.2f')} "
            f"| {_f(s['peak_lag'], ',')} | {ok} | {ok_proc} |")
    return "\n".join(lines)


def plot(runs: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, r in enumerate(runs):
        c = SERIES[i % len(SERIES)]
        d = r["df"].dropna(subset=["max_ingest_ts_ms"])
        total = np.sort((d["emit_ts_ms"] - d["max_ingest_ts_ms"]).to_numpy(dtype=float) / 1000)
        if len(total):
            ax1.plot(total, np.arange(1, len(total) + 1) / len(total), color=c, lw=2,
                     label=f"{r['name']} ({speed_label(r['meta'])})")
        lg = r["lag"]
        if len(lg):
            ax2.plot(lg["t_s"] / 60, lg["lag"], color=c, lw=2, label=r["name"])
    ax1.axvline(GOAL_S, color=INK2, lw=1, ls="--")
    ax1.text(GOAL_S, 0.03, " goal p95 < 10 s", color=INK2, fontsize=8)
    ax1.axhline(0.95, color=GRID, lw=1)
    ax1.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax1.xaxis.set_major_formatter(plain)
    ax1.xaxis.set_minor_formatter(plain)
    ax1.set_xlabel("emit − newest ingest of the window (s, log scale)")
    ax1.set_ylabel("share of windows")
    ax1.set_title("End-to-end latency per window (CDF)", loc="left", fontsize=10)
    ax2.set_xlabel("wall time since run start (min)")
    ax2.set_ylabel("consumer lag (records not yet in a committed batch)")
    ax2.set_title("Kafka consumer lag over time", loc="left", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def write_markdown(runs: list[dict], path: Path, png: Path, notes: str = "") -> None:
    hw = runs[0]["meta"].get("hardware", "") if runs else ""
    date = "on " + runs[0]["meta"]["started"][:10] if runs and runs[0]["meta"].get("started") else ""
    body = f"""# End-to-end latency: Kafka -> score_job

Measured {date} with `python experiments/e2e_latency_runs.py`, report by `python src/streaming/latency_report.py`.
Each run: `src/producer/replay.py --source smd --speedup S --limit N` and
`src/streaming/score_job.py --input kafka --reset` started together on an emptied `telemetry.smd` topic
(`OMP_NUM_THREADS=1`). {hw}

{markdown_table(runs)}

![latency CDF and consumer lag]({png.name})

## What the columns mean

* **total** = `emit_ts_ms − max_ingest_ts_ms`: from the moment the producer sent the *newest record of a
  window* to the moment the scored window was written (scores file and, for anomalies, the `anomalies` topic).
  This is what a user would call end-to-end latency, and it contains the watermark wait below.
* **processing** = `emit_ts_ms − batch_max_ingest_ts_ms`: from the newest record of that machine in the
  micro-batch that closed the window to the write: the time that record waited in Kafka for the next trigger
  (short while the producer is still sending, since Spark fixes a batch's offsets when it starts; up to one
  trigger interval when the producer stopped before the trigger) plus the micro-batch run time (Kafka read,
  parse, state update, AE + KMeans + PCA scoring, Parquet write, Kafka write) and any queueing behind a
  previous long batch.
* **batch run time** = `triggerExecution` of the micro-batches that had input rows.
* **watermark wait** = `batch_max_ingest_ts_ms − max_ingest_ts_ms` (measured): the 11 minutes of event time
  explained below converted to wall time, *plus* the time the closing records sat in Kafka until the next
  trigger fired (up to one trigger interval, 30 s by default). **11 min / speed-up** is the event-time part
  alone.
* **peak lag** = the largest number of records in `telemetry.smd` that no committed micro-batch had
  processed yet (broker end offsets minus the offsets of the newest committed batch in the checkpoint),
  sampled every 5 s.

## Why the 10-minute watermark is reported separately

A 5-minute window `[s, s+5)` cannot be scored when its last record (minute `s+4`) arrives: with a
10-minute watermark it is complete only once that machine has sent a record with event time `>= s+15`.
That is **11 minutes of event time** of pure waiting, by design (it lets out-of-order records join their
window). How long that is in wall-clock seconds depends only on how fast event time runs:

| replay speed-up | event-time wait in wall seconds (11 min / speed-up) |
| --- | ---: |
| 1x (real time) | 660 |
| 100x | 6.6 |
| 500x | 1.32 |
| max (0) | ≈ 0 (bounded by the producer rate) |

In production (speed-up 1) every window is therefore scored at the earliest 11 minutes after its newest
record, whatever the cluster does; this is a correctness choice, not a performance property. The
proposal's goal **p95 < 10 s** is about the stream, so it should be read against the **processing**
column; the **total** column shows the sum at each replay speed.

{notes}
"""
    path.write_text(body, encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--runs", nargs="*", default=None, help="run folder names (default: all, by name)")
    ap.add_argument("--notes", type=Path, default=None, help="Markdown appended under the explanation")
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-png", type=Path, default=OUT_PNG)
    args = ap.parse_args(argv)

    folders = ([args.runs_dir / n for n in args.runs] if args.runs else
               sorted((p for p in args.runs_dir.glob("*") if (p / "meta.json").exists()),
                      key=lambda p: json.loads((p / "meta.json").read_text())["order"]
                      if "order" in json.loads((p / "meta.json").read_text()) else 99))
    if not folders:
        print(f"FAIL no runs under {args.runs_dir}")
        return 1
    runs = [load_run(f) for f in folders]
    for r in runs:
        s = r["summary"]
        print(f"{r['name']}: {s['windows']} windows, total p50/p95/max {_f(s['total_p50_s'])}/"
              f"{_f(s['total_p95_s'])}/{_f(s['total_max_s'])} s, processing p95 {_f(s['proc_p95_s'])} s, "
              f"peak lag {_f(s['peak_lag'], ',')}")
    plot(runs, args.out_png)
    notes = args.notes.read_text(encoding="utf-8") if args.notes and args.notes.exists() else ""
    write_markdown(runs, args.out_md, args.out_png, notes)
    print(f"wrote {args.out_md.relative_to(ROOT).as_posix() if args.out_md.is_relative_to(ROOT) else args.out_md}"
          f" and {args.out_png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
