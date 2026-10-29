"""Dashboard data layer (WP-4): read the scores and incidents Parquet layers into pandas.

Checked by tools/check_dashboard.py.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from contracts import schemas as S

ROOT = Path(__file__).resolve().parents[2]


def _parquet_files(folder: Path):
    """Every data file under folder, skipping Spark side files such as _SUCCESS and .crc."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*.parquet")
                  if not p.name.startswith(("_", ".")) and p.stat().st_size > 0)


def _load_layer(layer: str, source: str, root: Path) -> pd.DataFrame:
    if source not in S.SOURCES:
        raise ValueError(f"unknown source {source!r}, expected one of {S.SOURCES}")
    schema = S.arrow_schema(layer)
    tables = []
    for path in _parquet_files(Path(root) / S.PATHS[layer] / f"source={source}"):
        t = pq.read_table(path)
        missing = [c for c in schema.names if c not in t.column_names]
        if missing:
            raise ValueError(f"{path} is missing {layer} columns {missing}")
        # Cast smooths over Spark/pyarrow differences (timestamp unit or tz, int width).
        tables.append(t.select(schema.names).cast(schema))
    table = pa.concat_tables(tables) if tables else schema.empty_table()
    return table.to_pandas()


def load_scores(source: str = "smd", root: Path = ROOT) -> pd.DataFrame:
    """All rows of data/scores/source=<source>, with the contract scores columns in order."""
    return _load_layer("scores", source, root)


def load_incidents(source: str = "smd", root: Path = ROOT) -> pd.DataFrame:
    """All rows of data/incidents/source=<source>, with the contract incidents columns in order."""
    return _load_layer("incidents", source, root)
