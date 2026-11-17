"""E2 bottleneck: per-batch time model T(n) = a + b / n from the score_job.log files of a scaling sweep.

score_job.py prints, for every micro-batch,
    batch 7: input rows 200000, windows scored ... , duration 5123 ms, state rows ...
      durationMs {"addBatch": 4980, "getBatch": 0, "latestOffset": 12, ..., "triggerExecution": 5123, ...}
      foreachBatch steps ms {"count(stateful scoring)": 4700, "toPandas(test)": 80, ...}
(--read-only: "rows counted N" and the single step "count(read only)"). This script reads those lines from every
run of a sweep, keeps the full batches (input rows >= --full-fraction x the run's largest batch, so the partial
last batch is dropped) minus the first batch of each run (it pays the Python workers' torch import and model
load), and averages every durationMs key and every foreachBatch step per executor count n (pooled over the
repetitions). For triggerExecution, addBatch and the first foreachBatch step (the count that runs the stateful
scoring, or the read in --read-only) it fits, by least squares on the per-n means,
    T(n) = a + b / n          a = serial part (s/batch), b = parallel work (core-independent, s/batch x n)
and reports serial fraction a / (a + b) (= Amdahl's f at n = 1) and R^2.

Run from the repo root (any machine with the logs):
    python experiments/e2_batch_model.py --sweep-json docs/results/e2_cluster_A.json      # logs in data/scaling/logs
    python experiments/e2_batch_model.py --sweep-json docs/results/e2_cluster_A.json --run-dir /path/to/scaling
    python experiments/e2_batch_model.py data/scaling/logs/e2_A_*/score_job.log --out docs/results/e2_batch_model_A.md
Without --sweep-json the executor count is read from the run id in the log path (..._e<E>c<C>_rep<k>/score_job.log).
Output: Markdown table (default docs/results/e2_batch_model_<tag>.md, from the sweep JSON name), also printed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data" / "scaling"
OUT_DIR = ROOT / "docs" / "results"
NAN = float("nan")

BATCH_RE = re.compile(r"^batch (?P<id>\d+): input rows (?P<rows>\d+)")
DURATION_RE = re.compile(r"^\s+durationMs (?P<json>\{.*\})\s*$")
STEPS_RE = re.compile(r"^\s+foreachBatch steps ms (?P<json>\{.*\})\s*$")
RUN_RE = re.compile(r"_e(?P<e>\d+)c(?P<c>\d+)_rep\d+")
FIT_DURATIONS = ("triggerExecution", "addBatch")


# ---- parsing --------------------------------------------------------------------------------------------

def parse_log(text: str) -> list[dict]:
    """score_job.log text -> [{batch, rows, duration: {key: ms}, steps: {name: ms}}] in log order."""
    batches: list[dict] = []
    for line in text.splitlines():
        m = BATCH_RE.match(line)
        if m:
            batches.append({"batch": int(m["id"]), "rows": int(m["rows"]), "duration": {}, "steps": {}})
            continue
        if not batches:
            continue
        for pattern, key in ((DURATION_RE, "duration"), (STEPS_RE, "steps")):
            m = pattern.match(line)
            if m:
                try:
                    batches[-1][key] = {k: float(v) for k, v in json.loads(m["json"]).items()}
                except (ValueError, TypeError):
                    pass
    return batches


def steady_batches(batches: list[dict], full_fraction: float = 0.9) -> list[dict]:
    """Batches with input rows, without the first one, and only full ones (rows >= full_fraction x largest)."""
    with_rows = [b for b in batches if b["rows"] > 0]
    if not with_rows:
        return []
    full = full_fraction * max(b["rows"] for b in with_rows)
    return [b for b in with_rows[1:] if b["rows"] >= full]


def executors_from_path(path: Path) -> int | None:
    """Executor count from a run id in the path (..._e4c2_rep1/score_job.log -> 4)."""
    m = RUN_RE.search(path.as_posix())
    return int(m["e"]) if m else None


# ---- model ----------------------------------------------------------------------------------------------

def fit_amdahl(ns: list[float], ts: list[float]) -> dict:
    """Least squares T = a + b / n. Returns a, b, serial fraction a / (a + b), R^2 (NaN with < 2 distinct n)."""
    pts = [(1.0 / n, t) for n, t in zip(ns, ts) if n and not math.isnan(t)]
    out = {"a": NAN, "b": NAN, "serial_fraction": NAN, "r2": NAN, "points": len(pts)}
    if len({x for x, _ in pts}) < 2:
        return out
    k = len(pts)
    mx, mt = sum(x for x, _ in pts) / k, sum(t for _, t in pts) / k
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (t - mt) for x, t in pts) / sxx
    a = mt - b * mx
    ss_tot = sum((t - mt) ** 2 for _, t in pts)
    ss_res = sum((t - (a + b * x)) ** 2 for x, t in pts)
    out.update(a=a, b=b, serial_fraction=a / (a + b) if a + b else NAN,
               r2=1 - ss_res / ss_tot if ss_tot else 1.0)
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else NAN


def summarise(runs: list[tuple[int, list[dict]]]) -> dict:
    """[(executors, steady batches of one run)] -> per-n means (seconds) of every durationMs key and step, and
    the fits. Keys keep their first-seen order."""
    pooled: dict[int, list[dict]] = {}
    runs_per_n: dict[int, int] = {}
    for n, batches in runs:
        runs_per_n[n] = runs_per_n.get(n, 0) + 1
        pooled.setdefault(n, []).extend(batches)
    dkeys, skeys = [], []
    for bs in pooled.values():
        for b in bs:
            dkeys += [k for k in b["duration"] if k not in dkeys]
            skeys += [k for k in b["steps"] if k not in skeys]
    rows = []
    for n in sorted(pooled):
        bs = pooled[n]
        rows.append({"executors": n, "runs": runs_per_n[n], "batches": len(bs),
                     "rows_mean": _mean([b["rows"] for b in bs]),
                     "duration": {k: _mean([b["duration"][k] / 1000 for b in bs if k in b["duration"]]) for k in dkeys},
                     "steps": {k: _mean([b["steps"][k] / 1000 for b in bs if k in b["steps"]]) for k in skeys}})
    ns = [r["executors"] for r in rows]
    fits = {}
    for k in FIT_DURATIONS:
        if k in dkeys:
            fits[f"durationMs.{k}"] = fit_amdahl(ns, [r["duration"][k] for r in rows])
    if skeys:
        fits[f"step.{skeys[0]}"] = fit_amdahl(ns, [r["steps"][skeys[0]] for r in rows])
    return {"rows": rows, "duration_keys": dkeys, "step_keys": skeys, "fits": fits}


# ---- report ---------------------------------------------------------------------------------------------

def _f(v: float, spec: str = ".2f") -> str:
    return "–" if v is None or math.isnan(v) else format(v, spec)


def markdown(summary: dict, title: str, source: str) -> str:
    rows, dkeys, skeys = summary["rows"], summary["duration_keys"], summary["step_keys"]
    head = ["executors", "runs", "batches", "rows/batch", *[f"`{k}`" for k in dkeys], *[f"step `{k}`" for k in skeys]]
    lines = [f"# {title}", "",
             f"Source: {source}. Steady full batches only (first batch of each run and partial batches dropped); "
             "mean seconds per batch, pooled over the repetitions.", "",
             "| " + " | ".join(head) + " |", "|" + " ---: |" * len(head)]
    for r in rows:
        cells = [r["executors"], r["runs"], r["batches"], _f(r["rows_mean"], ",.0f"),
                 *[_f(r["duration"][k]) for k in dkeys], *[_f(r["steps"][k]) for k in skeys]]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines += ["", "Fit T(n) = a + b / n on the per-n means (n = executors); serial fraction = a / (a + b).", "",
              "| quantity | a (s, serial) | b (s × n, parallel) | serial fraction | R² | points |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, f in summary["fits"].items():
        lines.append(f"| `{name}` | {_f(f['a'])} | {_f(f['b'])} | {_f(f['serial_fraction'])} | {_f(f['r2'], '.3f')} "
                     f"| {f['points']} |")
    return "\n".join(lines) + "\n"


def print_fits(summary: dict) -> None:
    for name, f in summary["fits"].items():
        print(f"{name}: T(n) = {_f(f['a'])} s + {_f(f['b'])} s / n, serial fraction {_f(f['serial_fraction'])}, "
              f"R^2 {_f(f['r2'], '.3f')} ({f['points']} executor counts)")


# ---- inputs ---------------------------------------------------------------------------------------------

def runs_from_sweep(sweep_json: Path, run_dir: Path) -> list[tuple[int, Path]]:
    """(executors, score_job.log) for every run of a saved sweep whose log exists (missing logs are reported)."""
    saved = json.loads(sweep_json.read_text(encoding="utf-8"))
    out = []
    for r in saved["runs"]:
        log = run_dir / "logs" / r["run_id"] / "score_job.log"
        if log.exists():
            out.append((int(r["executors"]), log))
        else:
            print(f"note: no log for {r['run_id']} at {log.as_posix()}", file=sys.stderr)
    return out


def load_runs(pairs: list[tuple[int, Path]], full_fraction: float) -> list[tuple[int, list[dict]]]:
    return [(n, steady_batches(parse_log(p.read_text(encoding="utf-8", errors="replace")), full_fraction))
            for n, p in pairs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*", type=Path, help="score_job.log files (executors from the run id in the path)")
    ap.add_argument("--sweep-json", type=Path, default=None, help="e2_cluster_<tag>.json of the sweep")
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR, help="drain_test.sh result dir holding logs/<run_id>/")
    ap.add_argument("--full-fraction", type=float, default=0.9,
                    help="keep batches with input rows >= this x the run's largest batch (default 0.9)")
    ap.add_argument("--out", type=Path, default=None, help="Markdown output (default from the sweep JSON name)")
    args = ap.parse_args(argv)

    pairs: list[tuple[int, Path]] = []
    if args.sweep_json:
        pairs += runs_from_sweep(args.sweep_json, args.run_dir)
    for p in args.logs:
        n = executors_from_path(p)
        if n is None:
            ap.error(f"cannot tell the executor count from {p} (expected ..._e<E>c<C>_rep<k>/...)")
        pairs.append((n, p))
    if not pairs:
        ap.error("no logs: give --sweep-json or score_job.log paths")
    summary = summarise([r for r in load_runs(pairs, args.full_fraction) if r[1]])
    if not summary["rows"]:
        print("FAIL no steady batches in the logs")
        return 1
    if args.out is None:
        stem = args.sweep_json.stem if args.sweep_json else "e2_cluster"
        args.out = OUT_DIR / (stem.replace("e2_cluster", "e2_batch_model", 1) + ".md")
    source = f"`{args.sweep_json.as_posix()}`" if args.sweep_json else f"{len(pairs)} score_job.log files"
    text = markdown(summary, "E2 bottleneck: per-batch time vs executors", source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print_fits(summary)
    print(f"written {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
