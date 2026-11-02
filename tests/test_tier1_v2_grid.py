"""experiments/tier1_v2_grid.py: windows, MAD floor, recentring, agreement, persistence and the selection rule."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from contracts import schemas as S
from experiments import tier1_v2_grid as M

BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
METRICS = ["m00", "m01", "m02"]


def _write_raw(path, n_train=60, n_test=40):
    """One machine, one row per minute: m00 noisy, m01 constant (MAD 0), m02 noisy with a gap."""
    rng = np.random.default_rng(0)
    n = n_train + n_test
    m00, m02 = 10 + rng.normal(size=n), 5 + rng.normal(size=n)
    metrics = [[("m00", float(m00[i])), ("m01", 3.0)] + ([] if 20 <= i < 25 else [("m02", float(m02[i]))])
               for i in range(n)]
    label = [None] * n_train + [0] * n_test
    table = pa.Table.from_arrays(
        [pa.array(["machine-x"] * n), pa.array(BASE_US + np.arange(n, dtype=np.int64) * 60_000_000).cast(pa.timestamp("us", tz="UTC")),
         pa.array(metrics, type=pa.map_(pa.string(), pa.float64())),
         pa.array([[("split", "train" if i < n_train else "test")] for i in range(n)],
                  type=pa.map_(pa.string(), pa.string())),
         pa.array(label, type=pa.int32())], schema=S.arrow_schema("raw"))
    pq.write_table(table, path)
    return m00


def test_windows_and_z(tmp_path):
    m00 = _write_raw(tmp_path / "machine-x.parquet")
    w = M.windows(tmp_path / "machine-x.parquet", METRICS)
    assert len(w) == 20 and w["is_test"].sum() == 8 and (w["label"] == 0).all()
    assert w["m00"].iloc[0] == m00[:5].mean()
    assert np.isnan(w["m02"].iloc[4])                     # minutes 20-24 have no m02
    Z = M.z_matrix(w, METRICS, recenter=4)
    # m01 is constant: MAD 0 is floored to 0.01*3 + 1e-3, so z is 0, not inf / nan
    assert np.all(Z[:, 1] == 0.0)
    assert np.isnan(Z[4, 2]) and np.isfinite(Z[5:, 0]).all()


def test_v2_flags_agreement_and_persistence():
    Z = np.array([[5, 5, 0], [5, 5, 0], [5, 0, 0], [7, 7, 7], [7, 7, 7], [np.nan, 9, 9]], dtype=float)
    assert M.v2_flags(Z, 2, 4.0, persist=1).tolist() == [True, True, False, True, True, True]
    assert M.v2_flags(Z, 2, 4.0).tolist() == [False, True, False, False, True, True]
    assert M.v2_flags(Z, 3, 6.0).tolist() == [False, False, False, False, True, False]


def test_select_rule():
    grid = pd.DataFrame({"k": [2, 2, 3, 3], "t": [4.0, 6.0, 4.0, 6.0],
                         "t1_composite_f1": [0.6, 0.5, 0.5, 0.4],
                         "t1_false_alarms_per_machine_day": [3.0, 1.5, 1.0, 0.5]})
    best = M.select(grid)
    assert (best["k"], best["t"]) == (3, 4.0)             # 0.6 has too many FA; 0.5 tie -> fewer FA
    assert M.select(grid.assign(t1_false_alarms_per_machine_day=5.0)) is None
