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
Only pyarrow is used (no Spark), so it runs on any machine that has the incidents layer on disk.
"""
import argparse
import re
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


def render_markdown(summary: list[dict]) -> str:
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
    return "\n".join(lines) + "\n"


def run(root: Path, out: Path) -> list[dict]:
    """Evaluate the layers under root and write the Markdown report to out; returns the summary rows."""
    segments = load_segments(root / S.PATHS["smd_interpretation"])
    incidents = load_incidents(root / S.PATHS["incidents"] / f"source={SOURCE}")
    machines = sorted({s["entity_id"] for s in segments}, key=_machine_key)
    n_train = train_row_counts(root / S.PATHS["raw"] / f"source={SOURCE}", machines)
    summary = summarize(evaluate(segments, incidents, n_train))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/")
    parser.add_argument("--out", type=Path, default=None, help="report path (default docs/results/attribution.md)")
    args = parser.parse_args()
    out = args.out if args.out is not None else args.root / DEFAULT_OUT
    summary = run(args.root, out)
    overall = summary[0]
    print(f"hit-rate@{TOP_K}: {overall['hits']}/{overall['segments']} segments "
          f"({_pct(overall['hits'], overall['segments'])}); wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
