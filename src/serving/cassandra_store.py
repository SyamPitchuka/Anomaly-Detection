"""Cassandra serving store for anomalies and incidents (E6). Optional: nothing in the pipeline needs it.

Run from the repo root:
    python src/serving/cassandra_store.py --hosts spark-w2,spark-w5 schema           # keyspace + tables
    python src/serving/cassandra_store.py --hosts spark-w2,spark-w5 load-incidents   # data/incidents -> store
    python src/serving/cassandra_store.py --hosts spark-w2,spark-w5 count            # rows per table
    python src/streaming/score_job.py ... --cassandra spark-w2,spark-w5 [--cassandra-cl QUORUM]

Data model (query-first: "the latest anomalies / incidents of machine X"):
    keyspace anomaly            SimpleStrategy, replication_factor 2 (one replica on each of the 2 nodes)
    anomalies_by_machine        PRIMARY KEY ((entity_id), window_start), newest first
                                entity_id text, window_start timestamp, score double (= ae_score), top_metrics list<text>
    incidents_by_machine        PRIMARY KEY ((entity_id), start_time), newest first
                                + incident_id, end_time, n_windows, max_score, top_metrics, created_at
                                (incident_id = <entity_id>_<start_time yyyyMMddHHmm>, so the key is the incident)

Exactly-once at the store: every write is a plain CQL INSERT, which in Cassandra is an upsert of the whole row
(a list column is overwritten, never appended). The primary key is the contract's dedup key, (entity_id,
window_start) for anomalies, so writing the same window again -- a replayed micro-batch after an E3 restart,
or the same batch sent twice -- gives the same single row with the same values (the scores are deterministic).
The rows of one call are also deduplicated on the key before they are sent (the last one wins).

Consistency level: ONE, QUORUM or ALL per store (parse_consistency()); with RF 2, QUORUM = 2 replicas = ALL.
The cassandra-driver package is imported only when a store connects, so this module (and the tests) work
without it; the levels are the CQL protocol codes, which the driver accepts as ConsistencyLevel values.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEYSPACE = "anomaly"
REPLICATION_FACTOR = 2
ANOMALIES_TABLE = "anomalies_by_machine"
INCIDENTS_TABLE = "incidents_by_machine"
DEFAULT_PORT = 9042
DEFAULT_CL = "QUORUM"
LOCAL_DC = "datacenter1"   # SimpleSnitch puts every node in datacenter1 / rack1

# CQL native protocol consistency codes (= cassandra.ConsistencyLevel.ONE / QUORUM / ALL in the driver)
CONSISTENCY_CODES = {"ONE": 1, "QUORUM": 4, "ALL": 5}
CONSISTENCY_LEVELS = tuple(CONSISTENCY_CODES)

ANOMALY_COLUMNS = ("entity_id", "window_start", "score", "top_metrics")
INCIDENT_COLUMNS = ("entity_id", "start_time", "incident_id", "end_time", "n_windows", "max_score",
                    "top_metrics", "created_at")
TABLE_COLUMNS = {ANOMALIES_TABLE: ANOMALY_COLUMNS, INCIDENTS_TABLE: INCIDENT_COLUMNS}
KEY_COLUMNS = {ANOMALIES_TABLE: ("entity_id", "window_start"), INCIDENTS_TABLE: ("entity_id", "start_time")}


# ---------------------------------------------------------------- parsing

def parse_consistency(text: str) -> str:
    """'one' / 'Quorum' / 'ALL' -> 'ONE' / 'QUORUM' / 'ALL'; anything else -> ValueError (argparse shows it)."""
    name = str(text).strip().upper()
    if name not in CONSISTENCY_CODES:
        raise ValueError(f"consistency level must be one of {', '.join(CONSISTENCY_LEVELS)}, got {text!r}")
    return name


def consistency_code(name: str) -> int:
    return CONSISTENCY_CODES[parse_consistency(name)]


def parse_hosts(text: str | Sequence[str]) -> list[str]:
    """'spark-w2,spark-w5' or 'spark-w2 spark-w5' (or a list of those) -> ['spark-w2', 'spark-w5']."""
    parts = [text] if isinstance(text, str) else list(text)
    hosts = [h for p in parts for h in str(p).replace(",", " ").split() if h]
    if not hosts:
        raise ValueError("no Cassandra hosts given")
    return list(dict.fromkeys(hosts))


def _argparse_type(fn):
    def wrapped(text):
        try:
            return fn(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    wrapped.__name__ = fn.__name__
    return wrapped


consistency_arg = _argparse_type(parse_consistency)
hosts_arg = _argparse_type(parse_hosts)


# ---------------------------------------------------------------- CQL text

def schema_cql(keyspace: str = KEYSPACE, rf: int = REPLICATION_FACTOR) -> list[str]:
    """CREATE ... IF NOT EXISTS statements for the keyspace and both tables (safe to run again)."""
    return [
        f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = "
        f"{{'class': 'SimpleStrategy', 'replication_factor': {int(rf)}}}",
        f"CREATE TABLE IF NOT EXISTS {keyspace}.{ANOMALIES_TABLE} ("
        "entity_id text, window_start timestamp, score double, top_metrics list<text>, "
        "PRIMARY KEY ((entity_id), window_start)) WITH CLUSTERING ORDER BY (window_start DESC)",
        f"CREATE TABLE IF NOT EXISTS {keyspace}.{INCIDENTS_TABLE} ("
        "entity_id text, start_time timestamp, incident_id text, end_time timestamp, n_windows bigint, "
        "max_score double, top_metrics list<text>, created_at timestamp, "
        "PRIMARY KEY ((entity_id), start_time)) WITH CLUSTERING ORDER BY (start_time DESC)",
    ]


def insert_cql(table: str, keyspace: str = KEYSPACE) -> str:
    """Prepared INSERT (= upsert of the whole row) for one of the tables."""
    cols = TABLE_COLUMNS[table]
    return f"INSERT INTO {keyspace}.{table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"


def count_cql(table: str, keyspace: str = KEYSPACE) -> str:
    return f"SELECT COUNT(*) FROM {keyspace}.{table}"


# ---------------------------------------------------------------- rows

def _utc(value) -> datetime | None:
    """int microseconds since the epoch, pandas Timestamp or datetime -> timezone-aware UTC datetime."""
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
    except ImportError:  # pragma: no cover - pandas is a pipeline requirement
        pass
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    us = int(value)
    return datetime.fromtimestamp(us // 1_000_000, tz=timezone.utc).replace(microsecond=us % 1_000_000)


def _float(value) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if v != v else v   # NaN -> null


def _metrics(value) -> list[str] | None:
    if value is None:
        return None
    try:
        return [str(m) for m in value]
    except TypeError:
        return None


def dedupe(rows: Iterable[tuple], key_len: int = 2) -> list[tuple]:
    """Keep one row per primary key (the first key_len fields), the last one wins; key order = first seen."""
    out: dict[tuple, tuple] = {}
    for row in rows:
        out[tuple(row[:key_len])] = tuple(row)
    return list(out.values())


def anomaly_rows(pdf) -> list[tuple]:
    """Scored windows of one micro-batch (score_job: entity_id, ws_us or window_start, ae_score, top_metrics,
    is_anomaly) -> (entity_id, window_start, score, top_metrics) tuples for the windows with is_anomaly,
    deduplicated on (entity_id, window_start)."""
    if pdf is None or len(pdf) == 0:
        return []
    if "is_anomaly" in pdf:
        pdf = pdf[pdf["is_anomaly"].fillna(False).astype(bool)]
    starts = pdf["ws_us"] if "ws_us" in pdf else pdf["window_start"]
    rows = ((str(e), _utc(s), _float(sc), _metrics(t))
            for e, s, sc, t in zip(pdf["entity_id"], starts, pdf["ae_score"], pdf["top_metrics"]))
    return dedupe(rows)


def incident_rows(pdf) -> list[tuple]:
    """Incidents layer rows (contracts LAYERS['incidents']) -> INCIDENT_COLUMNS tuples, deduplicated on
    (entity_id, start_time)."""
    if pdf is None or len(pdf) == 0:
        return []
    rows = ((str(r.entity_id), _utc(r.start_time), str(r.incident_id), _utc(r.end_time),
             None if r.n_windows is None else int(r.n_windows), _float(r.max_score), _metrics(r.top_metrics),
             _utc(r.created_at))
            for r in pdf.itertuples(index=False))
    return dedupe(rows)


# ---------------------------------------------------------------- store

@dataclass
class UpsertResult:
    rows: int = 0             # distinct rows sent
    failed: int = 0           # rows whose write raised (e.g. Unavailable at CL ALL with a node down)
    ms: float = 0.0           # wall time of the whole call
    errors: list[str] = field(default_factory=list)   # first few error messages

    @property
    def ok(self) -> int:
        return self.rows - self.failed


def _driver_execute_many(session, prepared, params: list[tuple], concurrency: int) -> list[tuple[bool, object]]:
    from cassandra.concurrent import execute_concurrent_with_args
    return list(execute_concurrent_with_args(session, prepared, params, concurrency=concurrency,
                                             raise_on_first_error=False))


def _serial_execute_many(session, prepared, params: list[tuple], concurrency: int) -> list[tuple[bool, object]]:
    """Same contract as cassandra.concurrent.execute_concurrent_with_args, one statement at a time (tests)."""
    out = []
    for p in params:
        try:
            out.append((True, session.execute(prepared, p)))
        except Exception as exc:  # noqa: BLE001
            out.append((False, exc))
    return out


def connect_session(hosts: Sequence[str], port: int = DEFAULT_PORT, consistency: str = DEFAULT_CL,
                    request_timeout: float = 10.0, connect_timeout: float = 10.0):
    """(cluster, session) with the real driver: token-aware round robin over datacenter1."""
    from cassandra.cluster import EXEC_PROFILE_DEFAULT, Cluster, ExecutionProfile
    from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy

    profile = ExecutionProfile(load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy(local_dc=LOCAL_DC)),
                               consistency_level=consistency_code(consistency), request_timeout=request_timeout)
    cluster = Cluster(list(hosts), port=port, execution_profiles={EXEC_PROFILE_DEFAULT: profile},
                      connect_timeout=connect_timeout)
    return cluster, cluster.connect()


class CassandraStore:
    """Upserts anomalies / incidents at one consistency level.

    Pass `session` (and optionally `execute_many`) to use a fake in tests; otherwise connect() opens a driver
    session to `hosts`."""

    def __init__(self, hosts: str | Sequence[str], consistency: str = DEFAULT_CL, *, keyspace: str = KEYSPACE,
                 rf: int = REPLICATION_FACTOR, port: int = DEFAULT_PORT, concurrency: int = 64,
                 session=None, execute_many: Callable | None = None):
        self.hosts = parse_hosts(hosts)
        self.consistency = parse_consistency(consistency)
        self.keyspace, self.rf, self.port, self.concurrency = keyspace, rf, port, concurrency
        self.session = session
        self.cluster = None
        self._execute_many = execute_many or (_serial_execute_many if session is not None else _driver_execute_many)
        self._prepared: dict[str, object] = {}

    def connect(self) -> "CassandraStore":
        if self.session is None:
            self.cluster, self.session = connect_session(self.hosts, self.port, self.consistency)
        return self

    def ensure_schema(self) -> None:
        self.connect()
        for stmt in schema_cql(self.keyspace, self.rf):
            self.session.execute(stmt)

    def _statement(self, table: str):
        if table not in self._prepared:
            prepared = self.session.prepare(insert_cql(table, self.keyspace))
            prepared.consistency_level = consistency_code(self.consistency)
            self._prepared[table] = prepared
        return self._prepared[table]

    def upsert(self, table: str, rows: Iterable[tuple]) -> UpsertResult:
        t0 = time.perf_counter()
        params = dedupe(rows, len(KEY_COLUMNS[table]))
        res = UpsertResult(rows=len(params))
        if params:
            self.connect()
            for ok, value in self._execute_many(self.session, self._statement(table), params, self.concurrency):
                if not ok:
                    res.failed += 1
                    if len(res.errors) < 3:
                        res.errors.append(f"{type(value).__name__}: {value}")
        res.ms = (time.perf_counter() - t0) * 1000
        return res

    def upsert_anomalies(self, rows: Iterable[tuple]) -> UpsertResult:
        return self.upsert(ANOMALIES_TABLE, rows)

    def upsert_incidents(self, rows: Iterable[tuple]) -> UpsertResult:
        return self.upsert(INCIDENTS_TABLE, rows)

    def count(self, table: str) -> int:
        self.connect()
        row = self.session.execute(count_cql(table, self.keyspace)).one()
        return int(row[0])

    def close(self) -> None:
        if self.cluster is not None:
            self.cluster.shutdown()
        self.cluster = None


# ---------------------------------------------------------------- CLI

def read_incidents(path: Path):
    import pandas as pd
    import pyarrow.dataset as ds
    if not path.exists():
        raise FileNotFoundError(f"no incidents layer at {path}")
    return ds.dataset(path.as_posix(), format="parquet", partitioning="hive").to_table().to_pandas() \
        if path.is_dir() else pd.read_parquet(path)


def main(argv=None) -> int:
    from contracts import schemas as S
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hosts", type=hosts_arg, required=True, help="contact points, e.g. spark-w2,spark-w5")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cl", type=consistency_arg, default=DEFAULT_CL, help="ONE | QUORUM | ALL")
    parser.add_argument("command", choices=("schema", "load-incidents", "count"))
    parser.add_argument("--incidents", type=Path, default=ROOT / S.PATHS["incidents"],
                        help="load-incidents: incidents layer directory or file (default data/incidents)")
    args = parser.parse_args(argv)

    store = CassandraStore(args.hosts, args.cl, port=args.port)
    try:
        store.ensure_schema()
        print(f"schema ok: keyspace {KEYSPACE} (SimpleStrategy, RF {REPLICATION_FACTOR}), "
              f"tables {ANOMALIES_TABLE}, {INCIDENTS_TABLE} on {','.join(store.hosts)}", flush=True)
        if args.command == "load-incidents":
            res = store.upsert_incidents(incident_rows(read_incidents(args.incidents)))
            print(f"incidents upserted {res.ok}/{res.rows} at CL {store.consistency} in {res.ms:.0f} ms"
                  + (f"; errors: {res.errors}" if res.failed else ""), flush=True)
            if res.failed:
                return 1
        elif args.command == "count":
            for table in (ANOMALIES_TABLE, INCIDENTS_TABLE):
                print(f"{KEYSPACE}.{table}: {store.count(table)} rows", flush=True)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
