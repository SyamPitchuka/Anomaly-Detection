import pandas as pd
import pytest

from src.detection import smd_profile as M

N_TRAIN = {"machine-1-1": 5, "machine-1-2": 4}
N_TEST = {"machine-1-1": 8, "machine-1-2": 3}
TEST_LABELS = {"machine-1-1": [0, 1, 1, 1, 0, 1, 0, 0], "machine-1-2": [0, 0, 0]}


def _rows(n: int) -> str:
    return "\n".join(",".join(f"{i + c / 100:.2f}" for c in range(38)) for i in range(n)) + "\n"


@pytest.fixture()
def smd_dir(tmp_path):
    src = tmp_path / "ServerMachineDataset"
    for sub in ("train", "test", "test_label", "interpretation_label"):
        (src / sub).mkdir(parents=True)
    for machine in N_TRAIN:
        (src / "train" / f"{machine}.txt").write_text(_rows(N_TRAIN[machine]))
        (src / "test" / f"{machine}.txt").write_text(_rows(N_TEST[machine]))
        (src / "test_label" / f"{machine}.txt").write_text(
            "\n".join(str(v) for v in TEST_LABELS[machine]) + "\n")
    (src / "interpretation_label" / "machine-1-1.txt").write_text("2-4:1,9\n6-6:2\n")
    return src


def test_segment_lengths():
    assert M.segment_lengths(pd.Series([0, 1, 1, 0, 1, 1, 1, 1])) == [2, 4]
    assert M.segment_lengths(pd.Series([0, 0])) == []
    assert M.segment_lengths(pd.Series([1])) == [1]


def test_build_profile(smd_dir):
    profile = M.build_profile(smd_dir).set_index("Machine")
    m1 = profile.loc["machine-1-1"]
    assert (m1["Train rows"], m1["Test rows"], m1["Anomalous test rows"]) == (5, 8, 4)
    assert m1["Anomaly ratio"] == pytest.approx(0.5)
    assert (m1["Anomaly segments"], m1["Longest segment"], m1["Interpretation lines"]) == (2, 3, 2)
    m2 = profile.loc["machine-1-2"]
    assert (m2["Anomalous test rows"], m2["Anomaly segments"], m2["Longest segment"], m2["Interpretation lines"]) == (0, 0, 0, 0)
    total = profile.loc["Total"]
    assert (total["Train rows"], total["Test rows"], total["Anomalous test rows"]) == (9, 11, 4)
    assert total["Anomaly ratio"] == pytest.approx(4 / 11)
    assert (total["Anomaly segments"], total["Longest segment"], total["Interpretation lines"]) == (2, 3, 2)


def test_document(smd_dir, tmp_path):
    out = tmp_path / "docs" / "smd_profile.md"
    assert M.main.__module__ == M.__name__
    text = M.render_document(M.build_profile(smd_dir))
    out.parent.mkdir()
    out.write_text(text, encoding="utf-8")
    table_lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert len(table_lines) == 2 + 2 + 1  # header, separator, two machines, totals
    assert table_lines[-1].startswith("| **Total**")
    summary = text.split("## Summary")[1].strip()
    assert summary.count(". ") + summary.endswith(".") == 3
