"""Profile the original OmniAnomaly ServerMachineDataset (SMD), one row per machine.

Run from the repo root:
    python src/detection/smd_profile.py [--input DIR] [--output FILE]

For every machine found in <input>/train the script reports: train rows, test rows, anomalous test rows,
anomaly ratio, number of anomaly segments (runs of consecutive 1s in test_label), longest segment and the
number of interpretation-label lines. The result is written as a Markdown table plus a totals row and a
three-sentence summary to docs/smd_profile.md. Only pandas is used; no contract file is touched.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_SMD_SUBPATH = Path("datasets/OmniAnomaly/ServerMachineDataset")
DEFAULT_OUTPUT = ROOT / "docs" / "smd_profile.md"
N_COLUMNS = 38
EXPECTED_MACHINES = 28

COLUMNS = [
    "Machine", "Train rows", "Test rows", "Anomalous test rows", "Anomaly ratio",
    "Anomaly segments", "Longest segment", "Interpretation lines",
]


def _default_input() -> Path:
    """~/datasets/... if it exists, else the datasets/ folder next to the repo, else the ~ path."""
    candidates = [Path("~").expanduser() / _SMD_SUBPATH, ROOT.parent / _SMD_SUBPATH]
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    return candidates[0]


def _count_rows(path: Path) -> int:
    """Number of data rows of a comma-separated SMD file; checks the 38-column layout."""
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != N_COLUMNS:
        raise ValueError(f"{path}: expected {N_COLUMNS} columns, found {frame.shape[1]}")
    return int(len(frame))


def _read_labels(path: Path) -> pd.Series:
    return pd.read_csv(path, header=None).iloc[:, 0].astype("int64")


def segment_lengths(labels: pd.Series) -> list[int]:
    """Lengths of the runs of consecutive 1s in a 0/1 label series."""
    is_one = labels.eq(1)
    run_id = (is_one != is_one.shift(fill_value=False)).cumsum()
    lengths = is_one.groupby(run_id).agg(["first", "size"])
    return [int(n) for n in lengths.loc[lengths["first"], "size"]]


def _count_lines(path: Path) -> int:
    """Number of non-blank lines; 0 if the file does not exist."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def profile_machine(input_dir: Path, machine: str) -> dict:
    """All profile figures of one machine."""
    name = f"{machine}.txt"
    train_rows = _count_rows(input_dir / "train" / name)
    test_rows = _count_rows(input_dir / "test" / name)
    labels = _read_labels(input_dir / "test_label" / name)
    if len(labels) != test_rows:
        raise ValueError(f"{machine}: test has {test_rows} rows but test_label has {len(labels)}")
    lengths = segment_lengths(labels)
    anomalous = int(labels.eq(1).sum())
    return {
        "Machine": machine,
        "Train rows": train_rows,
        "Test rows": test_rows,
        "Anomalous test rows": anomalous,
        "Anomaly ratio": anomalous / test_rows if test_rows else 0.0,
        "Anomaly segments": len(lengths),
        "Longest segment": max(lengths) if lengths else 0,
        "Interpretation lines": _count_lines(input_dir / "interpretation_label" / name),
    }


def build_profile(input_dir: Path) -> pd.DataFrame:
    """One row per machine (sorted by name) plus a final 'Total' row."""
    machines = sorted(p.stem for p in (input_dir / "train").glob("*.txt"))
    if not machines:
        raise FileNotFoundError(f"no train/*.txt files under {input_dir}")
    table = pd.DataFrame([profile_machine(input_dir, m) for m in machines], columns=COLUMNS)
    total_test = int(table["Test rows"].sum())
    total_anomalous = int(table["Anomalous test rows"].sum())
    total = {
        "Machine": "Total",
        "Train rows": int(table["Train rows"].sum()),
        "Test rows": total_test,
        "Anomalous test rows": total_anomalous,
        "Anomaly ratio": total_anomalous / total_test if total_test else 0.0,
        "Anomaly segments": int(table["Anomaly segments"].sum()),
        "Longest segment": int(table["Longest segment"].max()),
        "Interpretation lines": int(table["Interpretation lines"].sum()),
    }
    return pd.concat([table, pd.DataFrame([total], columns=COLUMNS)], ignore_index=True)


def _fmt(column: str, value) -> str:
    if column == "Machine":
        return str(value)
    if column == "Anomaly ratio":
        return f"{value * 100:.2f}%"
    return f"{int(value):,}"


def to_markdown(profile: pd.DataFrame) -> str:
    """Markdown table of the profile (the last row is the totals row, printed in bold)."""
    last = len(profile) - 1
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(COLUMNS) - 1)) + " |"]
    for idx, row in profile.iterrows():
        cells = [_fmt(c, row[c]) for c in COLUMNS]
        if idx == last:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(profile: pd.DataFrame) -> str:
    """Exactly three sentences describing the profile."""
    machines = profile.iloc[:-1]
    total = profile.iloc[-1]
    ratio = machines["Anomaly ratio"]
    hi, lo = machines.loc[ratio.idxmax()], machines.loc[ratio.idxmin()]
    longest = machines.loc[machines["Longest segment"].idxmax()]
    return (
        f"SMD has {len(machines)} machines with {int(total['Train rows']):,} train rows and "
        f"{int(total['Test rows']):,} test rows, of which {int(total['Anomalous test rows']):,} "
        f"({total['Anomaly ratio'] * 100:.2f}%) are labelled anomalous. "
        f"The anomaly ratio per machine ranges from {lo['Anomaly ratio'] * 100:.2f}% ({lo['Machine']}) "
        f"to {hi['Anomaly ratio'] * 100:.2f}% ({hi['Machine']}), so a single global threshold will not suit every machine. "
        f"The anomalies form {int(total['Anomaly segments']):,} segments, the longest being "
        f"{int(longest['Longest segment']):,} rows ({longest['Machine']}), and "
        f"{int(total['Interpretation lines']):,} interpretation-label lines are available."
    )


def render_document(profile: pd.DataFrame) -> str:
    return (
        "# SMD profile\n\n"
        "Source: original OmniAnomaly ServerMachineDataset (train, test, test_label, interpretation_label). "
        "A segment is a run of consecutive 1s in `test_label`.\n\n"
        f"{to_markdown(profile)}\n\n"
        "## Summary\n\n"
        f"{summarize(profile)}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=_default_input(), help="ServerMachineDataset folder")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown file to write")
    args = parser.parse_args()
    input_dir = args.input.expanduser()
    if not (input_dir / "train").is_dir():
        print(f"ERROR: no train/ folder under {input_dir}", file=sys.stderr)
        return 1
    profile = build_profile(input_dir)
    if len(profile) - 1 != EXPECTED_MACHINES:
        print(f"WARNING: expected {EXPECTED_MACHINES} machines, found {len(profile) - 1}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_document(profile), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
