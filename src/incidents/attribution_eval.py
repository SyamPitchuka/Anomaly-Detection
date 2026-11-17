"""Attribution hit-rate@3 of incidents against the SMD interpretation labels.

Run from the repo root:
    python src/incidents/attribution_eval.py [--root DIR] [--out docs/results/attribution.md]

Segments: every row of data/reference/smd_interpretation (one labelled test anomaly segment: entity_id, start_idx,
end_idx, metrics). start_idx / end_idx index the machine's test rows, both inclusive. Test row j of a machine is
global row n_train + j, so it lies at TIME_BASE_UTC + (n_train + j) minutes (contract rule); n_train is the number
of raw rows with a null label (= train rows) of that machine. A segment covers [time(start_idx), time(end_idx) + 1 min).

Overlap: an incident of the same entity_id overlaps a segment if incident.start_time < segment end and
incident.end_time > segment start (incident end_time is the last alert's window_end, so it is exclusive as well).

Hit@3: a segment is a hit if at least one of its labelled metrics is among the first 3 entries of top_metrics of at
least one overlapping incident. A segment without an overlapping incident is a miss, so hit-rate@3 = hits / all
segments. The report also gives the share of segments with an incident and the hit-rate among those segments.
Random baseline (chance level): every overlapping incident is given TOP_K metrics drawn at random (without
replacement) from the SMD metrics m00..m37 instead of its real top_metrics; a segment is a hit if one of its labelled
metrics is among the drawn ones. Segments without an overlapping incident stay misses, so coverage is the same as for
the real run and only the choice of metrics differs. Repeated --trials times with a fixed --seed; the report gives the
mean +- standard deviation and the exact expectation (1 - prod over a segment's incidents of C(38-k, 3) / C(38, 3),
k = labelled metrics of the segment) as a check.
Only pyarrow is used (no Spark), so it runs on any machine that has the incidents layer on disk.
"""
import argparse
import math
import random
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402

TOP_K = 3
SOURCE = "smd"
MINUTE_US = 60 * 1_000_000
DEFAULT_OUT = Path("docs/results/attribution.md")
_UNIT_TO_US = {"s": 1_000_000, "ms": 1_000, "us": 1}


def _base_us() -> int:
    """TIME_BASE_UTC as integer microseconds since the epoch."""
    base = datetime.fromisoformat(S.TIME_BASE_UTC.replace("Z", "+00:00"))
    return int(base.timestamp()) * 1_000_000


def _read_parquet_dir(path: Path, columns: list[str]) -> pa.Table:
    """Concatenate all *.parquet files directly under path (ignores Spark's _SUCCESS and .crc files)."""
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {path}")
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


def _micros(column: pa.ChunkedArray) -> list[int]:
    """A timestamp column as integer microseconds since the epoch (no tz database needed)."""
    factor = _UNIT_TO_US[column.type.unit]
    return [v * factor for v in column.cast(pa.int64()).to_pylist()]


def load_segments(ref_dir: Path) -> list[dict]:
    """The labelled segments: dicts with entity_id, start_idx, end_idx, metrics (set of metric keys)."""
    table = pq.read_table(ref_dir / "interpretation.parquet") if ref_dir.is_dir() else pq.read_table(ref_dir)
    rows = table.to_pylist()
    return [{"entity_id": r["entity_id"], "start_idx": r["start_idx"], "end_idx": r["end_idx"],
             "metrics": set(r["metrics"] or [])} for r in rows]


def load_incidents(incidents_dir: Path) -> dict[str, list[tuple[int, int, list[str]]]]:
    """Incidents per entity_id as (start_us, end_us, top_metrics)."""
    table = _read_parquet_dir(incidents_dir, ["entity_id", "start_time", "end_time", "top_metrics"])
    entities = table["entity_id"].to_pylist()
    starts, ends = _micros(table["start_time"]), _micros(table["end_time"])
    tops = table["top_metrics"].to_pylist()
    result: dict[str, list[tuple[int, int, list[str]]]] = {}
    for entity, start, end, top in zip(entities, starts, ends, tops):
        result.setdefault(entity, []).append((start, end, list(top or [])))
    return result


def train_row_counts(raw_dir: Path, machines: list[str]) -> dict[str, int]:
    """n_train per machine = number of raw rows whose label is null (train rows carry no label)."""
    counts = {}
    for machine in machines:
        path = raw_dir / f"{machine}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"raw layer file missing for {machine}: {path}")
        counts[machine] = pq.read_table(path, columns=["label"])["label"].null_count
    return counts


def segment_bounds_us(n_train: int, start_idx: int, end_idx: int) -> tuple[int, int]:
    """[start, end) of a segment in epoch microseconds; start_idx / end_idx are inclusive test-row indexes."""
    base = _base_us()
    return base + (n_train + start_idx) * MINUTE_US, base + (n_train + end_idx + 1) * MINUTE_US


def evaluate(segments: list[dict], incidents: dict[str, list[tuple[int, int, list[str]]]],
             n_train: dict[str, int]) -> list[dict]:
    """One result per segment: entity_id, has_incident (bool), hit (bool)."""
    results = []
    for seg in segments:
        seg_start, seg_end = segment_bounds_us(n_train[seg["entity_id"]], seg["start_idx"], seg["end_idx"])
        overlapping = [top for (start, end, top) in incidents.get(seg["entity_id"], [])
                       if start < seg_end and end > seg_start]
        predicted = {m for top in overlapping for m in top[:TOP_K]}
        results.append({"entity_id": seg["entity_id"], "has_incident": bool(overlapping),
                        "hit": bool(seg["metrics"] & predicted)})
    return results


def random_baseline(segments: list[dict], incidents: dict[str, list[tuple[int, int, list[str]]]],
                    n_train: dict[str, int], trials: int = 1000, seed: int = 0) -> dict:
    """Hit-rate@3 when every overlapping incident gets TOP_K random metrics instead of its real top_metrics.

    Returns hits per trial (mean, std), the exact expected hits, the number of segments / covered segments."""
    pool = list(S.SMD_METRICS)
    pairs = []   # (labelled metrics of the segment, number of overlapping incidents)
    for seg in segments:
        seg_start, seg_end = segment_bounds_us(n_train[seg["entity_id"]], seg["start_idx"], seg["end_idx"])
        n_overlap = sum(1 for (start, end, _) in incidents.get(seg["entity_id"], [])
                        if start < seg_end and end > seg_start)
        if n_overlap:
            pairs.append((seg["metrics"], n_overlap))
    rng = random.Random(seed)
    hits = []
    for _ in range(trials):
        n_hit = 0
        for metrics, n_overlap in pairs:
            drawn = set()
            for _i in range(n_overlap):
                drawn.update(rng.sample(pool, TOP_K))
            n_hit += bool(metrics & drawn)
        hits.append(n_hit)
    miss_one = lambda k: math.comb(len(pool) - k, TOP_K) / math.comb(len(pool), TOP_K)   # noqa: E731
    expected = sum(1 - miss_one(len(m & set(pool))) ** n for m, n in pairs)
    return {"segments": len(segments), "with_incident": len(pairs), "trials": trials, "seed": seed,
            "mean_hits": statistics.fmean(hits), "std_hits": statistics.pstdev(hits), "expected_hits": expected}


def _machine_key(name: str) -> tuple:
    """Natural sort: machine-1-2 before machine-1-10."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def _count_row(label: str, rows: list[dict]) -> dict:
    return {"label": label, "segments": len(rows), "with_incident": sum(r["has_incident"] for r in rows),
            "hits": sum(r["hit"] for r in rows)}


def summarize(results: list[dict]) -> list[dict]:
    """Overall row first, then one row per machine (natural order)."""
    summary = [_count_row("**All machines**", results)]
    for machine in sorted({r["entity_id"] for r in results}, key=_machine_key):
        summary.append(_count_row(machine, [r for r in results if r["entity_id"] == machine]))
    return summary


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.1f} %" if den else "n/a"


def render_markdown(summary: list[dict], baseline: dict | None = None) -> str:
    lines = [
        "# Incident attribution: hit-rate@3 on SMD",
        "",
        "Each labelled SMD test anomaly segment (`data/reference/smd_interpretation`) is matched with the incidents "
        "(`data/incidents/source=smd`) of the same machine that overlap it in time. A segment is a **hit** if at "
        f"least one of its labelled metrics is in the `top_metrics` (first {TOP_K}) of an overlapping incident. "
        "**Hit-rate@3 = hits / all segments**, so a segment with no incident counts as a miss. "
        "*Hit-rate if covered* = hits / segments that have an overlapping incident.",
        "",
        "| Machine | Segments | With incident | Hits@3 | Hit-rate@3 | Hit-rate if covered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['label']} | {row['segments']} | {row['with_incident']} | {row['hits']} | "
                     f"{_pct(row['hits'], row['segments'])} | {_pct(row['hits'], row['with_incident'])} |")
    if baseline is not None:
        n, cov = baseline["segments"], baseline["with_incident"]
        real = summary[0]
        lines += [
            "",
            "## Chance level (random baseline)",
            "",
            f"Same calculation, but every overlapping incident gets {TOP_K} metrics drawn at random from the "
            f"{len(S.SMD_METRICS)} SMD metrics instead of its `top_metrics`. Segments without an incident stay misses, "
            f"so coverage is identical. {baseline['trials']} trials, seed {baseline['seed']}; mean +- std over trials.",
            "",
            "| | Hits@3 | Hit-rate@3 | Hit-rate if covered |",
            "|---|---:|---:|---:|",
            f"| Real `top_metrics` | {real['hits']} | {_pct(real['hits'], n)} | {_pct(real['hits'], cov)} |",
            f"| Random metrics | {baseline['mean_hits']:.1f} +- {baseline['std_hits']:.1f} | "
            f"{_pct(baseline['mean_hits'], n)} | {_pct(baseline['mean_hits'], cov)} |",
            "",
            f"Exact expectation of the random baseline: {baseline['expected_hits']:.1f} hits "
            f"({_pct(baseline['expected_hits'], n)} of all segments, {_pct(baseline['expected_hits'], cov)} if covered).",
        ]
    return "\n".join(lines) + "\n"


def evaluate_root(root: Path, trials: int = 1000, seed: int = 0) -> tuple[list[dict], dict | None]:
    """Summary rows and random baseline (None when trials = 0) for the layers under root."""
    segments = load_segments(root / S.PATHS["smd_interpretation"])
    incidents = load_incidents(root / S.PATHS["incidents"] / f"source={SOURCE}")
    machines = sorted({s["entity_id"] for s in segments}, key=_machine_key)
    n_train = train_row_counts(root / S.PATHS["raw"] / f"source={SOURCE}", machines)
    summary = summarize(evaluate(segments, incidents, n_train))
    baseline = random_baseline(segments, incidents, n_train, trials, seed) if trials > 0 else None
    return summary, baseline


def run(root: Path, out: Path, trials: int = 1000, seed: int = 0) -> list[dict]:
    """Evaluate the layers under root and write the Markdown report to out; returns the summary rows."""
    summary, baseline = evaluate_root(root, trials, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(summary, baseline), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/")
    parser.add_argument("--out", type=Path, default=None, help="report path (default docs/results/attribution.md)")
    parser.add_argument("--trials", type=int, default=1000, help="random-baseline trials (0 = skip the baseline)")
    parser.add_argument("--seed", type=int, default=0, help="random-baseline seed")
    args = parser.parse_args()
    out = args.out if args.out is not None else args.root / DEFAULT_OUT
    summary, baseline = evaluate_root(args.root, args.trials, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(summary, baseline), encoding="utf-8")
    overall = summary[0]
    print(f"hit-rate@{TOP_K}: {overall['hits']}/{overall['segments']} segments "
          f"({_pct(overall['hits'], overall['segments'])}); wrote {out.as_posix()}")
    if baseline is not None:
        print(f"random baseline: {baseline['mean_hits']:.1f} hits "
              f"({_pct(baseline['mean_hits'], overall['segments'])}), exact {baseline['expected_hits']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
