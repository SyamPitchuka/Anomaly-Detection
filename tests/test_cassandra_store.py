"""src/serving/cassandra_store.py with a fake session (no Cassandra, no cassandra-driver needed)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import argparse
import numpy as np
import pandas as pd
import pytest

from src.serving import cassandra_store as CS

WS = int(datetime(2026, 1, 3, 10, 5, tzinfo=timezone.utc).timestamp() * 1_000_000)   # microseconds
MIN_US = 60 * 1_000_000


class FakeSession:
    """Keeps rows by table and primary key, as Cassandra does: an INSERT with an existing key replaces the row."""

    def __init__(self, fail_keys=()):
        self.tables: dict[str, dict] = {}
        self.ddl: list[str] = []
        self.prepared: list = []
        self.fail_keys = set(fail_keys)

    def prepare(self, cql):
        stmt = SimpleNamespace(query=cql, consistency_level=None)
        self.prepared.append(stmt)
        return stmt

    def execute(self, stmt, params=None):
        if isinstance(stmt, str):
            self.ddl.append(stmt)
            return SimpleNamespace(one=lambda: (sum(len(t) for t in self.tables.values()),))
        assert stmt.query.startswith("INSERT INTO "), "only plain INSERTs (whole-row upserts) are sent"
        table = stmt.query.split()[2].split(".")[1]
        key = tuple(params[:2])
        if key in self.fail_keys:
            raise RuntimeError("Unavailable: not enough replicas")
        self.tables.setdefault(table, {})[key] = tuple(params)
        return None


def scored_batch():
    """What score_job.write_batch has collected: test windows with ws_us, ae_score, top_metrics, is_anomaly."""
    return pd.DataFrame({
        "entity_id": ["machine-1-1", "machine-1-1", "machine-1-2", "machine-1-2"],
        "ws_us": np.array([WS, WS + 5 * MIN_US, WS, WS + 5 * MIN_US], dtype=np.int64),
        "ae_score": [3.5, 0.1, np.nan, 7.25],
        "top_metrics": [["m03", "m01", "m10"], ["m00", "m01", "m02"], None, np.array(["m07", "m08", "m09"])],
        "is_anomaly": [True, False, True, True],
    })


def test_parse_consistency_and_codes():
    assert [CS.parse_consistency(x) for x in ("one", " Quorum ", "ALL")] == ["ONE", "QUORUM", "ALL"]
    for bad in ("TWO", "LOCAL_QUORUM", "", "any"):
        with pytest.raises(ValueError):
            CS.parse_consistency(bad)
    with pytest.raises(argparse.ArgumentTypeError):
        CS.consistency_arg("SERIAL")
    assert CS.CONSISTENCY_CODES == {"ONE": 1, "QUORUM": 4, "ALL": 5}


def test_codes_match_the_driver_when_installed():
    driver = pytest.importorskip("cassandra")
    for name, code in CS.CONSISTENCY_CODES.items():
        assert getattr(driver.ConsistencyLevel, name) == code


def test_parse_hosts():
    assert CS.parse_hosts("spark-w2,spark-w5") == ["spark-w2", "spark-w5"]
    assert CS.parse_hosts("spark-w2 spark-w5,spark-w2") == ["spark-w2", "spark-w5"]
    assert CS.parse_hosts(["spark-w2", "spark-w5"]) == ["spark-w2", "spark-w5"]
    with pytest.raises(ValueError):
        CS.parse_hosts(" , ")


def test_schema_cql_text():
    ks, anomalies, incidents = CS.schema_cql()
    assert ks == ("CREATE KEYSPACE IF NOT EXISTS anomaly WITH replication = "
                  "{'class': 'SimpleStrategy', 'replication_factor': 2}")
    assert anomalies.startswith("CREATE TABLE IF NOT EXISTS anomaly.anomalies_by_machine (")
    for col in ("entity_id text", "window_start timestamp", "score double", "top_metrics list<text>"):
        assert col in anomalies
    assert "PRIMARY KEY ((entity_id), window_start)" in anomalies
    assert incidents.startswith("CREATE TABLE IF NOT EXISTS anomaly.incidents_by_machine (")
    assert "PRIMARY KEY ((entity_id), start_time)" in incidents and "incident_id text" in incidents
    assert CS.insert_cql(CS.ANOMALIES_TABLE) == ("INSERT INTO anomaly.anomalies_by_machine "
                                                 "(entity_id, window_start, score, top_metrics) VALUES (?, ?, ?, ?)")
    assert CS.insert_cql(CS.INCIDENTS_TABLE).count("?") == len(CS.INCIDENT_COLUMNS)
    assert "'replication_factor': 3" in CS.schema_cql(rf=3)[0]


def test_anomaly_rows_keep_only_anomalies_with_exact_types():
    rows = CS.anomaly_rows(scored_batch())
    assert [r[0] for r in rows] == ["machine-1-1", "machine-1-2", "machine-1-2"]
    assert rows[0] == ("machine-1-1", datetime(2026, 1, 3, 10, 5, tzinfo=timezone.utc), 3.5, ["m03", "m01", "m10"])
    assert rows[1][2] is None and rows[1][3] is None           # NaN score and missing top_metrics -> null
    assert rows[2][1] == datetime(2026, 1, 3, 10, 10, tzinfo=timezone.utc) and rows[2][3] == ["m07", "m08", "m09"]
    assert CS.anomaly_rows(scored_batch().iloc[0:0]) == []
    # window_start as a pandas timestamp (scores layer) gives the same row
    pdf = scored_batch().drop(columns="ws_us")
    pdf["window_start"] = pd.to_datetime(scored_batch()["ws_us"], unit="us", utc=True)
    assert CS.anomaly_rows(pdf) == rows


def test_upsert_is_idempotent_on_entity_and_window_start():
    session = FakeSession()
    store = CS.CassandraStore("spark-w2,spark-w5", "quorum", session=session)
    rows = CS.anomaly_rows(scored_batch())
    first = store.upsert_anomalies(rows)
    assert (first.rows, first.ok, first.failed) == (3, 3, 0)
    snapshot = dict(session.tables[CS.ANOMALIES_TABLE])
    # a replayed micro-batch (E3 restart) and a duplicate inside one call: still the same 3 rows, same values
    again = store.upsert_anomalies(rows + rows[:1])
    assert again.rows == 3
    assert session.tables[CS.ANOMALIES_TABLE] == snapshot
    # a later write of the same key replaces the row (last wins), never adds one
    changed = [(rows[0][0], rows[0][1], 9.0, ["m01"])]
    store.upsert_anomalies(changed)
    assert len(session.tables[CS.ANOMALIES_TABLE]) == 3
    assert session.tables[CS.ANOMALIES_TABLE][rows[0][:2]] == changed[0]
    # one prepared statement per table, at the store's consistency level (QUORUM = protocol code 4)
    assert len(session.prepared) == 1 and session.prepared[0].consistency_level == 4


def test_dedupe_keeps_last_and_first_seen_order():
    rows = [("a", 1, 1.0), ("b", 1, 2.0), ("a", 1, 3.0), ("a", 2, 4.0)]
    assert CS.dedupe(rows) == [("a", 1, 3.0), ("b", 1, 2.0), ("a", 2, 4.0)]


def test_failed_upserts_are_counted_not_hidden():
    rows = CS.anomaly_rows(scored_batch())
    session = FakeSession(fail_keys=[rows[1][:2]])
    res = CS.CassandraStore(["h1"], "ALL", session=session).upsert_anomalies(rows)
    assert (res.rows, res.ok, res.failed) == (3, 2, 1)
    assert res.errors and "Unavailable" in res.errors[0]
    assert res.ms >= 0
    assert CS.CassandraStore(["h1"], session=FakeSession()).upsert_anomalies([]).rows == 0


def test_ensure_schema_runs_every_statement_and_incidents_upsert():
    session = FakeSession()
    store = CS.CassandraStore(["h1"], "ONE", session=session)
    store.ensure_schema()
    assert session.ddl == CS.schema_cql()
    inc = pd.DataFrame({
        "incident_id": ["machine-1-1_202601031005", "machine-1-1_202601031005"],
        "entity_id": ["machine-1-1"] * 2,
        "start_time": pd.to_datetime(["2026-01-03T10:05:00Z"] * 2),
        "end_time": pd.to_datetime(["2026-01-03T10:20:00Z", "2026-01-03T10:25:00Z"]),
        "n_windows": [3, 4], "max_score": [5.0, 6.0], "top_metrics": [["m01"], ["m02"]],
        "created_at": pd.to_datetime(["2026-11-18T00:00:00Z"] * 2),
    })
    rows = CS.incident_rows(inc)
    assert len(rows) == 1 and rows[0][4] == 4      # same (entity_id, start_time) = same incident, last wins
    store.upsert_incidents(rows)
    stored = session.tables[CS.INCIDENTS_TABLE]
    assert list(stored) == [("machine-1-1", datetime(2026, 1, 3, 10, 5, tzinfo=timezone.utc))]
    assert stored[list(stored)[0]][2] == "machine-1-1_202601031005"
    assert session.prepared[-1].consistency_level == 1
