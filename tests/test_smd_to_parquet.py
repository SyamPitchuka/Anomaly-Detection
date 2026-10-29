from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.detection import smd_to_parquet as M

N_TRAIN = {"machine-1-1": 5, "machine-1-2": 4}
N_TEST = {"machine-1-1": 6, "machine-1-2": 3}
TEST_LABELS = {"machine-1-1": [0, 0, 1, 1, 0, 0], "machine-1-2": [0, 1, 0]}


def _row(machine_no: int, i: int) -> str:
    # value = machine_no * 1000 + i + column / 100
    cells = [f"{machine_no * 1000 + i + c / 100:.2f}" for c in range(38)]
    return ",".join(cells)


@pytest.fixture()
def smd_dir(tmp_path):
    src = tmp_path / "ServerMachineDataset"
    for sub in ("train", "test", "test_label", "interpretation_label"):
        (src / sub).mkdir(parents=True)
    for no, machine in enumerate(N_TRAIN, start=1):
        (src / "train" / f"{machine}.txt").write_text(
            "\n".join(_row(no, i) for i in range(N_TRAIN[machine])) + "\n")
        test_rows = [_row(no, 100 + i) for i in range(N_TEST[machine])]
        if machine == "machine-1-1":
            cells = test_rows[0].split(",")
            cells[4] = "nan"
            test_rows[0] = ",".join(cells)
        (src / "test" / f"{machine}.txt").write_text("\n".join(test_rows) + "\n")
        (src / "test_label" / f"{machine}.txt").write_text(
            "\n".join(str(v) for v in TEST_LABELS[machine]) + "\n")
    (src / "interpretation_label" / "machine-1-1.txt").write_text("2-3:1,9,38\n4-4:2\n")
    (src / "interpretation_label" / "machine-1-2.txt").write_text("1-2:10,11\n")
    return src


def test_convert_raw_layer(smd_dir, tmp_path, capsys):
    root = tmp_path / "repo"
    assert M.convert(smd_dir, root) == ["machine-1-1", "machine-1-2"]

    # event_time is compared as integer microseconds so the test does not need the tzdata package
    base_us = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
    for machine in N_TRAIN:
        path = root / "data/raw/source=smd" / f"{machine}.parquet"
        table = pq.read_table(path)
        assert table.schema.equals(S.arrow_schema("raw"))
        times_us = table["event_time"].cast(pa.int64()).to_pylist()
        rows = table.select(["entity_id", "metrics", "attrs", "label"]).to_pylist()
        n_tr, n_te = N_TRAIN[machine], N_TEST[machine]
        assert len(rows) == n_tr + n_te
        for i, r in enumerate(rows):
            assert r["entity_id"] == machine
            assert times_us[i] == base_us + i * 60_000_000
            split = "train" if i < n_tr else "test"
            assert dict(r["attrs"]) == {"split": split}
            if split == "train":
                assert r["label"] is None
            else:
                assert r["label"] == TEST_LABELS[machine][i - n_tr]
            expected_keys = set(S.SMD_METRICS)
            if machine == "machine-1-1" and i == n_tr:
                expected_keys -= {"m04"}  # NaN is left out, never 0
            assert {k for k, _ in r["metrics"]} == expected_keys

    first_table = pq.read_table(root / "data/raw/source=smd/machine-1-1.parquet", columns=["metrics"])
    first = dict(first_table.to_pylist()[0]["metrics"])
    assert first["m00"] == pytest.approx(1000.00)  # machine 1, row 0, column 0
    assert first["m37"] == pytest.approx(1000.37)  # column 37 -> m37
    assert "dimension numbers: min=1, max=38" in capsys.readouterr().out


def test_interpretation(smd_dir, tmp_path):
    root = tmp_path / "repo"
    M.convert(smd_dir, root)
    table = pq.read_table(root / "data/reference/smd_interpretation" / M.INTERPRETATION_FILE)
    assert table.schema.equals(S.arrow_schema("smd_interpretation"))
    rows = table.to_pylist()
    assert len(rows) == 3
    assert rows[0] == {"entity_id": "machine-1-1", "start_idx": 2, "end_idx": 3,
                       "metrics": ["m00", "m08", "m37"]}  # 1-based 1, 9, 38 -> m00, m08, m37
    assert rows[2]["entity_id"] == "machine-1-2"
    assert rows[2]["metrics"] == ["m09", "m10"]


def test_label_length_mismatch_raises(smd_dir, tmp_path):
    (smd_dir / "test_label" / "machine-1-2.txt").write_text("0\n1\n")
    with pytest.raises(ValueError):
        M.convert(smd_dir, tmp_path / "repo")


def test_dimension_out_of_range_raises(smd_dir, tmp_path):
    (smd_dir / "interpretation_label" / "machine-1-2.txt").write_text("1-2:0,5\n")
    with pytest.raises(ValueError):
        M.convert(smd_dir, tmp_path / "repo")
