"""Convert the OmniAnomaly ServerMachineDataset (SMD) to the raw layer and the interpretation reference.

Run from the repo root:
    python src/detection/smd_to_parquet.py [--input DIR] [--root DIR]

Raw layer: one file per machine at data/raw/source=smd/<machine>.parquet (train rows first, then test rows).
Row i of a machine gets event_time = 2026-01-01T00:00:00Z + i minutes. NaN values are left out of `metrics`.

Interpretation reference: data/reference/smd_interpretation/interpretation.parquet.
Source lines look like 'start-end:d1,d2,...'. The dimension numbers d are 1-based (1..38) and are
mapped to metric keys with m{d-1:02d}, i.e. 1 -> m00 and 38 -> m37. start_idx / end_idx are copied
unchanged from the file; they index the machine's test rows.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402

_SMD_SUBPATH = Path("datasets/OmniAnomaly/ServerMachineDataset")


def _default_input() -> Path:
    """~/datasets/... if it exists, else the datasets/ folder next to the repo, else the ~ path."""
    candidates = [Path("~").expanduser() / _SMD_SUBPATH, ROOT.parent / _SMD_SUBPATH]
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    return candidates[0]


DEFAULT_INPUT = _default_input()
N_METRICS = len(S.SMD_METRICS)
EXPECTED_MACHINES = 28
INTERPRETATION_FILE = "interpretation.parquet"


def _base_us() -> int:
    """TIME_BASE_UTC as integer microseconds since the epoch."""
    base = datetime.fromisoformat(S.TIME_BASE_UTC.replace("Z", "+00:00"))
    return int(base.timestamp()) * 1_000_000


def _load_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if data.shape[1] != N_METRICS:
        raise ValueError(f"{path}: expected {N_METRICS} columns, found {data.shape[1]}")
    return data


def _load_labels(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def build_raw_table(machine: str, train: np.ndarray, test: np.ndarray, test_label: np.ndarray) -> pa.Table:
    """Build the raw-layer table of one machine; train rows first, then test rows."""
    if len(test_label) != len(test):
        raise ValueError(f"{machine}: test has {len(test)} rows but test_label has {len(test_label)}")
    values = np.vstack([train, test])
    n_train, n = len(train), len(values)

    # metrics: map<string,double>, NaN values are omitted (never written as 0)
    valid = ~np.isnan(values)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(valid.sum(axis=1), dtype=np.int32)
    key_names = np.array(S.SMD_METRICS, dtype=object)
    col_idx = np.broadcast_to(np.arange(N_METRICS), values.shape)[valid]
    metrics = pa.MapArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(key_names[col_idx].tolist(), type=pa.string()),
        pa.array(values[valid], type=pa.float64()),
    )

    # attrs: {"split": "train" | "test"}
    split = np.array(["train"] * n_train + ["test"] * (n - n_train), dtype=object)
    attrs = pa.MapArray.from_arrays(
        pa.array(np.arange(n + 1, dtype=np.int32), type=pa.int32()),
        pa.array(["split"] * n, type=pa.string()),
        pa.array(split.tolist(), type=pa.string()),
    )

    event_us = _base_us() + np.arange(n, dtype=np.int64) * 60_000_000
    event_time = pa.array(event_us, type=pa.int64()).cast(pa.timestamp("us", tz="UTC"))

    label = pa.concat_arrays([
        pa.nulls(n_train, type=pa.int32()),
        pa.array(test_label.astype(np.int32), type=pa.int32()),
    ])

    return pa.Table.from_arrays(
        [pa.array([machine] * n, type=pa.string()), event_time, metrics, attrs, label],
        schema=S.arrow_schema("raw"),
    )


def parse_interpretation_line(line: str) -> tuple[int, int, list[int]]:
    """Parse 'start-end:d1,d2,...' into (start, end, [d1, d2, ...]) with 1-based dimensions."""
    span, dims = line.strip().split(":")
    start, end = span.split("-")
    return int(start), int(end), [int(d) for d in dims.split(",") if d.strip()]


def build_interpretation_table(input_dir: Path, machines: list[str]) -> tuple[pa.Table, int | None, int | None]:
    """Parse all interpretation_label files; returns (table, min_dimension, max_dimension)."""
    ent, starts, ends, mets = [], [], [], []
    all_dims: list[int] = []
    for machine in machines:
        path = input_dir / "interpretation_label" / f"{machine}.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            start, end, dims = parse_interpretation_line(line)
            bad = [d for d in dims if not 1 <= d <= N_METRICS]
            if bad:
                raise ValueError(f"{path}: dimension(s) {bad} outside 1..{N_METRICS}")
            ent.append(machine)
            starts.append(start)
            ends.append(end)
            mets.append([f"m{d - 1:02d}" for d in dims])
            all_dims.extend(dims)
    table = pa.Table.from_arrays(
        [
            pa.array(ent, type=pa.string()),
            pa.array(starts, type=pa.int64()),
            pa.array(ends, type=pa.int64()),
            pa.array(mets, type=pa.list_(pa.string())),
        ],
        schema=S.arrow_schema("smd_interpretation"),
    )
    return table, (min(all_dims) if all_dims else None), (max(all_dims) if all_dims else None)


def convert(input_dir: Path, root: Path) -> list[str]:
    """Convert every machine found in <input_dir>/train; returns the machine names."""
    machines = sorted(p.stem for p in (input_dir / "train").glob("*.txt"))
    if not machines:
        raise FileNotFoundError(f"no train/*.txt files under {input_dir}")

    raw_dir = root / S.PATHS["raw"] / "source=smd"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for machine in machines:
        train = _load_matrix(input_dir / "train" / f"{machine}.txt")
        test = _load_matrix(input_dir / "test" / f"{machine}.txt")
        test_label = _load_labels(input_dir / "test_label" / f"{machine}.txt")
        table = build_raw_table(machine, train, test, test_label)
        pq.write_table(table, raw_dir / f"{machine}.parquet")
        print(f"{machine}: {len(train)} train + {len(test)} test rows, {int(test_label.sum())} anomalous")

    ref_dir = root / S.PATHS["smd_interpretation"]
    ref_dir.mkdir(parents=True, exist_ok=True)
    interp, dmin, dmax = build_interpretation_table(input_dir, machines)
    pq.write_table(interp, ref_dir / INTERPRETATION_FILE)
    print(f"interpretation: {interp.num_rows} segments")
    print(f"interpretation dimension numbers: min={dmin}, max={dmax} "
          f"(1-based; d maps to m{{d-1:02d}}, so 1 -> m00 and {dmax} -> m{(dmax or 1) - 1:02d})")

    if len(machines) != EXPECTED_MACHINES:
        print(f"WARNING: expected {EXPECTED_MACHINES} machines, converted {len(machines)}")
    return machines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ServerMachineDataset folder")
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/")
    args = parser.parse_args()
    convert(args.input.expanduser(), args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
