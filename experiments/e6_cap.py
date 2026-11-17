"""E6: CAP behaviour of the Cassandra serving store (2 nodes, keyspace anomaly, RF 2) at CL ONE, QUORUM and ALL.

Run ON spark-master from the repo root, as the ssh user of deploy.sh (key-based ssh + passwordless sudo to the
Cassandra VMs), with the cluster venv, after 'deploy.sh cassandra':
    /opt/anomaly-venv/bin/python experiments/e6_cap.py                         # hosts from cluster.runtime.env
    /opt/anomaly-venv/bin/python experiments/e6_cap.py --rows 2000 --victim spark-w5
    python experiments/e6_cap.py --dry-run                                      # simulated 2-node cluster, no network

Steps (every op is one synchronous CQL statement, timed on the client; a row is (entity_id, window_start)):
  (a) both nodes up: for CL in ONE, QUORUM, ALL write N rows of a fresh partition, then read them back at the
      same CL -> success rate, p50 / p95 latency (successful ops), error classes of the failed ones;
  (b) 'sudo systemctl stop cassandra' on --victim (default: the last host) via ssh, wait until the survivor's
      'nodetool status' shows it DN (and the driver marks it down), repeat (a);
  (c) 'systemctl start cassandra' on the victim, wait until both are UN; record the hints the survivor held
      ('nodetool listpendinghints'), wait until they are delivered, count the victim's replica on its own;
      run 'nodetool repair -full anomaly e6_cap' and count each replica on its own again.
      Counting one replica "on its own" = stop the OTHER node, SELECT COUNT(*) at CL ONE (the only live
      replica must answer), start the other node again: the only exact per-replica count Cassandra offers.
      --no-isolated-counts skips that (then only the CL ALL count after repair is reported).
The rows go to their own table anomaly.e6_cap (same columns as anomalies_by_machine), truncated at the start,
so the serving tables are never touched.

Writes docs/results/e6_cap.md + e6_cap.json (--dry-run: data/e6_dryrun/, --out DIR to choose), with the CAP
reading: ONE = AP (stays available with a node down, replicas diverge until hints / repair converge them);
ALL = CP (refuses rather than write fewer than every replica); QUORUM with RF 2 needs floor(2/2)+1 = 2
replicas = both nodes, so it behaves like ALL here (RF 3 would be needed for a quorum that survives one node).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "infra" / "cluster"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.serving import cassandra_store as CS  # noqa: E402

TABLE = "e6_cap"
LEVELS = CS.CONSISTENCY_LEVELS            # ONE, QUORUM, ALL
OUT_DIR = ROOT / "docs" / "results"
DRY_OUT_DIR = ROOT / "data" / "e6_dryrun"
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]


def required_replicas(cl: str, rf: int) -> int:
    """Replicas that must acknowledge at this level: ONE 1, QUORUM floor(rf/2)+1, ALL rf."""
    return {"ONE": 1, "QUORUM": rf // 2 + 1, "ALL": rf}[CS.parse_consistency(cl)]


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100) of a list; None if empty."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarise(lat_ok: list[float], errors: dict[str, int], attempted: int, found: int | None = None) -> dict:
    ok = len(lat_ok)
    out = {"attempted": attempted, "ok": ok, "failed": attempted - ok,
           "success_rate": round(ok / attempted, 4) if attempted else None,
           "p50_ms": _r(percentile(lat_ok, 50)), "p95_ms": _r(percentile(lat_ok, 95)), "errors": dict(errors)}
    if found is not None:
        out["found"] = found
    return out


def _r(x):
    return None if x is None else round(x, 3)


def parse_nodetool_status(text: str) -> dict[str, str]:
    """'nodetool status' -> {address: state}, state = the 2-letter code (UN, DN, UJ, ...)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([UD][NLJM])\s+(\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1)
    return out


def parse_pending_hints(text: str) -> int:
    """'nodetool listpendinghints' -> number of target hosts with pending hints (0 = none)."""
    if "not have any pending hints" in text:
        return 0
    uuid = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
    return sum(1 for line in text.splitlines() if uuid.search(line))


class OpError(Exception):
    """A failed CQL op; .kind is the driver's exception class (Unavailable, NoHostAvailable, ...)."""
    def __init__(self, kind: str, message: str = ""):
        super().__init__(message or kind)
        self.kind = kind


# ---------------------------------------------------------------- backends

class SimBackend:
    """--dry-run: an in-memory RF-2 cluster with Cassandra's availability rules, hints and repair.
    Latencies are synthetic (seeded), so the report shape can be tested without a cluster."""

    def __init__(self, hosts: list[str], rf: int = 2, seed: int = 6):
        self.hosts, self.rf = list(hosts), rf
        self.up = set(hosts)
        self.data = {h: set() for h in hosts}
        self.hints = {h: {} for h in hosts}     # holder -> {target: set(keys)}
        self.rng = random.Random(seed)
        self.log: list[str] = []

    def _lat(self, cl: str) -> float:
        return self.rng.lognormvariate(0.0, 0.25) * {"ONE": 0.9, "QUORUM": 1.2, "ALL": 1.3}[cl]

    def reset_table(self):
        if self.up != set(self.hosts):
            raise OpError("Unavailable", "TRUNCATE needs every node")
        for h in self.hosts:
            self.data[h].clear()

    def _check(self, cl):
        if len(self.up) < required_replicas(cl, self.rf):
            raise OpError("Unavailable", f"CL {cl} needs {required_replicas(cl, self.rf)} live replicas, "
                                         f"{len(self.up)} alive")

    def write(self, cl: str, key: tuple) -> float:
        self._check(cl)
        coordinator = next(h for h in self.hosts if h in self.up)
        for h in self.hosts:
            if h in self.up:
                self.data[h].add(key)
            else:
                self.hints[coordinator].setdefault(h, set()).add(key)
        return self._lat(cl)

    def read(self, cl: str, key: tuple) -> tuple[float, bool]:
        self._check(cl)
        live = [h for h in self.hosts if h in self.up][:required_replicas(cl, self.rf)]
        return self._lat(cl), any(key in self.data[h] for h in live)

    def stop(self, node):
        self.up.discard(node)
        self.log.append(f"stop {node}")

    def start(self, node):
        self.up.add(node)
        self.log.append(f"start {node}")

    def pending_hints(self, holder) -> int:
        return sum(1 for keys in self.hints[holder].values() if keys)

    def deliver_hints(self, holder, timeout_s: float) -> bool:
        for target, keys in self.hints[holder].items():
            if target in self.up:
                self.data[target] |= keys
                keys.clear()
        return self.pending_hints(holder) == 0

    def repair(self, node) -> tuple[bool, str]:
        everything = set().union(*(self.data[h] for h in self.hosts if h in self.up))
        for h in self.up:
            self.data[h] |= everything
        return True, "simulated full repair"

    def count(self, cl: str) -> int:
        self._check(cl)
        live = [h for h in self.hosts if h in self.up][:required_replicas(cl, self.rf)]
        return len(set().union(*(self.data[h] for h in live)))

    def replica_count(self, node) -> int:
        return len(self.data[node])

    def close(self):
        pass


class RealBackend:
    """The 2-node cluster: CQL through cassandra-driver, node control and nodetool through ssh."""

    def __init__(self, hosts: list[str], ssh_user: str, cassandra_home: str, port: int = CS.DEFAULT_PORT,
                 rf: int = CS.REPLICATION_FACTOR, wait_s: float = 300.0):
        self.hosts, self.ssh_user, self.home, self.rf, self.wait_s = list(hosts), ssh_user, cassandra_home, rf, wait_s
        self.ip = {h: socket.gethostbyname(h) for h in hosts}
        self.cluster, self.session = CS.connect_session(hosts, port, "ONE", request_timeout=10.0)
        for stmt in CS.schema_cql():
            self.session.execute(stmt)
        self.session.execute(
            f"CREATE TABLE IF NOT EXISTS {CS.KEYSPACE}.{TABLE} (entity_id text, window_start timestamp, "
            "score double, top_metrics list<text>, PRIMARY KEY ((entity_id), window_start))")
        self.ins = self.session.prepare(f"INSERT INTO {CS.KEYSPACE}.{TABLE} (entity_id, window_start, score, "
                                        "top_metrics) VALUES (?, ?, ?, ?)")
        self.ins.is_idempotent = True
        self.sel = self.session.prepare(f"SELECT score FROM {CS.KEYSPACE}.{TABLE} "
                                        "WHERE entity_id = ? AND window_start = ?")
        self.sel.is_idempotent = True

    # -- ssh / nodetool
    def ssh(self, node: str, cmd: str, timeout: float = 600) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", "-n", *SSH_OPTS, f"{self.ssh_user}@{node}", cmd],
                              capture_output=True, text=True, timeout=timeout)

    def nodetool(self, node: str, args: str, timeout: float = 600) -> subprocess.CompletedProcess:
        return self.ssh(node, f"{self.home}/bin/nodetool {args}", timeout)

    def status(self, viewer: str) -> dict[str, str]:
        try:
            r = self.nodetool(viewer, "status", 60)
        except subprocess.TimeoutExpired:
            return {}
        return parse_nodetool_status(r.stdout) if r.returncode == 0 else {}

    def _driver_up(self, node: str) -> bool | None:
        for h in self.cluster.metadata.all_hosts():
            if h.address == self.ip[node]:
                return bool(h.is_up)
        return None

    def wait_state(self, node: str, state: str, viewer: str) -> float:
        """Until `viewer` reports `node` in `state` (UN / DN) and the driver agrees. Returns seconds waited."""
        t0 = time.monotonic()
        want_up = state == "UN"
        while time.monotonic() - t0 < self.wait_s:
            if self.status(viewer).get(self.ip[node]) == state and self._driver_up(node) in (want_up, None):
                return time.monotonic() - t0
            time.sleep(2)
        raise TimeoutError(f"{node} not {state} as seen from {viewer} after {self.wait_s:.0f} s")

    def stop(self, node):
        r = self.ssh(node, "sudo -n systemctl stop cassandra", 300)
        if r.returncode:
            raise RuntimeError(f"systemctl stop cassandra on {node}: {r.stderr.strip()}")
        viewer = next(h for h in self.hosts if h != node)
        self.wait_state(node, "DN", viewer)

    def start(self, node):
        r = self.ssh(node, "sudo -n systemctl start cassandra", 120)
        if r.returncode:
            raise RuntimeError(f"systemctl start cassandra on {node}: {r.stderr.strip()}")
        viewer = next(h for h in self.hosts if h != node)
        self.wait_state(node, "UN", viewer)
        self.wait_state(node, "UN", node)

    def pending_hints(self, holder) -> int:
        r = self.nodetool(holder, "listpendinghints", 60)
        return parse_pending_hints(r.stdout) if r.returncode == 0 else -1

    def deliver_hints(self, holder, timeout_s: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.pending_hints(holder) == 0:
                return True
            time.sleep(3)
        return False

    def repair(self, node) -> tuple[bool, str]:
        r = self.nodetool(node, f"repair -full {CS.KEYSPACE} {TABLE}", 1800)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-2000:]

    # -- CQL
    def reset_table(self):
        self.session.execute(f"TRUNCATE {CS.KEYSPACE}.{TABLE}", timeout=120)

    def _run(self, prepared, params, cl):
        bound = prepared.bind(params)
        bound.consistency_level = CS.consistency_code(cl)
        t0 = time.perf_counter()
        try:
            rows = self.session.execute(bound)
        except Exception as exc:  # noqa: BLE001  (Unavailable, NoHostAvailable, WriteTimeout, ...)
            raise OpError(type(exc).__name__, str(exc)[:300]) from exc
        return (time.perf_counter() - t0) * 1000, rows

    def write(self, cl: str, key: tuple) -> float:
        ms, _ = self._run(self.ins, (key[0], key[1], float(key[2]), ["m00"]), cl)
        return ms

    def read(self, cl: str, key: tuple) -> tuple[float, bool]:
        ms, rows = self._run(self.sel, (key[0], key[1]), cl)
        return ms, rows.one() is not None

    def count(self, cl: str) -> int:
        from cassandra.query import SimpleStatement
        stmt = SimpleStatement(f"SELECT COUNT(*) FROM {CS.KEYSPACE}.{TABLE}",
                               consistency_level=CS.consistency_code(cl))
        return int(self.session.execute(stmt, timeout=120).one()[0])

    def replica_count(self, node) -> int:
        """Exact row count of `node`'s replica: stop every other node, count at CL ONE, start them again."""
        others = [h for h in self.hosts if h != node]
        for h in others:
            self.stop(h)
        try:
            return self.count("ONE")
        finally:
            for h in others:
                self.start(h)

    def close(self):
        self.cluster.shutdown()


# ---------------------------------------------------------------- experiment

def key_for(phase: str, cl: str, i: int) -> tuple:
    return (f"e6-{phase}-{cl.lower()}", BASE_TIME + timedelta(minutes=i), i * 0.001)


def run_level(backend, phase: str, cl: str, n: int) -> dict:
    lat_w, err_w, lat_r, err_r, found = [], {}, [], {}, 0
    written = []
    for i in range(n):
        key = key_for(phase, cl, i)
        try:
            lat_w.append(backend.write(cl, key))
            written.append(key)
        except OpError as exc:
            err_w[exc.kind] = err_w.get(exc.kind, 0) + 1
    for i in range(n):
        key = key_for(phase, cl, i)
        try:
            ms, hit = backend.read(cl, key)
            lat_r.append(ms)
            found += int(hit)
        except OpError as exc:
            err_r[exc.kind] = err_r.get(exc.kind, 0) + 1
    return {"cl": cl, "required_replicas": required_replicas(cl, backend.rf),
            "write": summarise(lat_w, err_w, n), "read": summarise(lat_r, err_r, n, found)}


def run_phase(backend, phase: str, up: list[str], n: int, log) -> dict:
    results = []
    for cl in LEVELS:
        res = run_level(backend, phase, cl, n)
        log(f"  {phase} CL {cl:6s} write ok {res['write']['ok']}/{n} p50 {res['write']['p50_ms']} ms "
            f"p95 {res['write']['p95_ms']} ms | read ok {res['read']['ok']}/{n} found {res['read']['found']}")
        results.append(res)
    return {"phase": phase, "nodes_up": list(up), "results": results}


def run_experiment(backend, hosts: list[str], victim: str, n: int, isolated: bool = True,
                   hint_timeout_s: float = 300.0, log=print) -> dict:
    survivor = next(h for h in hosts if h != victim)
    t_start = datetime.now(timezone.utc)
    log(f"E6: {len(hosts)} nodes {hosts}, RF {backend.rf}, {n} rows per CL, victim {victim}")
    backend.reset_table()
    phases = [run_phase(backend, "both_up", hosts, n, log)]

    log(f"(b) stopping cassandra on {victim}")
    t0 = time.monotonic()
    backend.stop(victim)
    stop_s = time.monotonic() - t0
    phases.append(run_phase(backend, "one_down", [survivor], n, log))
    hints_before = backend.pending_hints(survivor)

    log(f"(c) starting cassandra on {victim}")
    t0 = time.monotonic()
    backend.start(victim)
    start_s = time.monotonic() - t0
    t0 = time.monotonic()
    delivered = backend.deliver_hints(survivor, hint_timeout_s)
    hints_s = time.monotonic() - t0
    expected = sum(r["write"]["ok"] for ph in phases for r in ph["results"])
    rec = {"survivor": survivor, "victim": victim, "stop_seconds": round(stop_s, 1),
           "restart_seconds": round(start_s, 1), "pending_hints_before_restart": hints_before,
           "hints_delivered": delivered, "hints_wait_seconds": round(hints_s, 1),
           "expected_rows": expected, "isolated_counts": isolated}
    if isolated:
        rec["victim_count_after_hints"] = backend.replica_count(victim)
        log(f"  victim {victim} alone after hints: {rec['victim_count_after_hints']} / {expected} rows")
    t0 = time.monotonic()
    ok, out = backend.repair(victim)
    rec["repair"] = {"ok": ok, "seconds": round(time.monotonic() - t0, 1), "output_tail": out[-600:]}
    log(f"  repair -full on {victim}: {'ok' if ok else 'FAILED'} in {rec['repair']['seconds']} s")
    if isolated:
        rec["replica_counts_after_repair"] = {h: backend.replica_count(h) for h in hosts}
        log(f"  replicas alone after repair: {rec['replica_counts_after_repair']}")
    rec["count_all_after_repair"] = backend.count("ALL")
    counts = list(rec.get("replica_counts_after_repair", {}).values())
    rec["converged"] = (len(set(counts)) == 1 and counts[0] == expected) if counts \
        else rec["count_all_after_repair"] == expected
    return {"experiment": "E6 CAP", "started_utc": t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hosts": hosts, "victim": victim, "rf": backend.rf, "rows_per_level": n,
            "keyspace": CS.KEYSPACE, "table": TABLE, "phases": phases, "recovery": rec}


# ---------------------------------------------------------------- report

def _level(result: dict, phase: str, cl: str) -> dict:
    ph = next(p for p in result["phases"] if p["phase"] == phase)
    return next(r for r in ph["results"] if r["cl"] == cl)


def interpret(result: dict) -> list[str]:
    """CAP reading of the observed numbers; any deviation from the expected behaviour is called out."""
    rf, lines = result["rf"], []
    for cl in LEVELS:
        need = required_replicas(cl, rf)
        down = _level(result, "one_down", cl)
        up = _level(result, "both_up", cl)
        avail = (down["write"]["success_rate"] or 0) >= 0.99 and (down["read"]["success_rate"] or 0) >= 0.99
        refused = down["write"]["ok"] == 0 and down["read"]["ok"] == 0
        expect_avail = need <= rf - 1
        tag = "as expected" if (avail if expect_avail else refused) else "UNEXPECTED, see the table"
        if up["write"]["success_rate"] != 1.0 or up["read"]["success_rate"] != 1.0:
            tag += "; NOTE: not every op succeeded with both nodes up"
        if expect_avail:
            lines.append(f"**{cl}** (needs {need} of {rf} replicas) stayed available with one node down "
                         f"({down['write']['ok']}/{down['write']['attempted']} writes, "
                         f"{down['read']['ok']}/{down['read']['attempted']} reads): it chooses **A over C** "
                         f"(AP). The down replica misses those writes until hints or repair copy them, so a "
                         f"read at ONE from it in that time can be stale ({tag}).")
        else:
            lines.append(f"**{cl}** (needs {need} of {rf} replicas) refused every op with one node down "
                         f"({down['write']['ok']} writes, {down['read']['ok']} reads succeeded; errors "
                         f"{down['write']['errors'] or '-'}): it chooses **C over A** (CP), never acknowledging "
                         f"a write that not every required replica holds ({tag}).")
    q = required_replicas("QUORUM", rf)
    lines.append(f"With RF {rf}, QUORUM = floor({rf}/2)+1 = {q} replicas"
                 + (" = ALL: on two nodes a quorum cannot lose a node, so QUORUM behaves exactly like ALL. "
                    "Reads and writes at QUORUM overlap (R + W = 4 > RF 2), so they are strongly consistent; "
                    "ONE + ONE (2, not > 2) is not. RF 3 on 3 nodes would give a QUORUM (2 of 3) that is both "
                    "consistent and survives one node." if q == rf else "."))
    rec = result["recovery"]
    conv = "converged" if rec["converged"] else "did NOT converge"
    lines.append(f"Recovery: the survivor held hints for {rec['pending_hints_before_restart']} host(s) while "
                 f"{rec['victim']} was down; after the restart they were "
                 f"{'delivered' if rec['hints_delivered'] else 'NOT all delivered'} in "
                 f"{rec['hints_wait_seconds']} s"
                 + (f" (victim alone then held {rec['victim_count_after_hints']} of {rec['expected_rows']} rows)"
                    if "victim_count_after_hints" in rec else "")
                 + f"; after nodetool repair the replicas {conv} ({rec['expected_rows']} rows expected). "
                   "That is the eventual consistency side of AP: availability during the partition, "
                   "convergence after it.")
    lines.append("A stopped node is what a network partition looks like to the coordinator (no gossip, no "
                 "replies), so these runs stand for a partition of one node; a real partition would also "
                 "leave clients on the other side, which a 2-node test cannot show.")
    return lines


def _fmt(x, unit=""):
    return "-" if x is None else f"{x}{unit}"


def render_markdown(result: dict) -> str:
    rec = result["recovery"]
    out = [f"# E6: CAP behaviour of the Cassandra serving store{' (DRY RUN, simulated)' if result.get('dry_run') else ''}",
           "",
           f"Nodes {', '.join(result['hosts'])}; keyspace `{result['keyspace']}` (SimpleStrategy, RF {result['rf']}); "
           f"table `{result['table']}`; {result['rows_per_level']} rows written and read back per consistency "
           f"level and phase; one synchronous statement per op, latency measured on the client "
           f"(spark-master). Victim: {result['victim']}. Started {result['started_utc']}. "
           f"Produced by `experiments/e6_cap.py`.",
           "",
           "| phase | CL | replicas needed | write ok | write p50 ms | write p95 ms | read ok | read p50 ms | read p95 ms | errors |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ph in result["phases"]:
        for r in ph["results"]:
            w, rd = r["write"], r["read"]
            errs = {**{f"write {k}": v for k, v in w["errors"].items()},
                    **{f"read {k}": v for k, v in rd["errors"].items()}}
            out.append(f"| {ph['phase']} ({'+'.join(ph['nodes_up'])} up) | {r['cl']} | {r['required_replicas']} "
                       f"| {w['ok']}/{w['attempted']} ({w['success_rate']:.1%}) | {_fmt(w['p50_ms'])} | {_fmt(w['p95_ms'])} "
                       f"| {rd['ok']}/{rd['attempted']} ({rd['success_rate']:.1%}) | {_fmt(rd['p50_ms'])} | {_fmt(rd['p95_ms'])} "
                       f"| {', '.join(f'{k} {v}' for k, v in errs.items()) or '-'} |")
    out += ["", "## Recovery", "",
            f"- stop of {rec['victim']} seen as DN after {rec['stop_seconds']} s; restart to UN in {rec['restart_seconds']} s",
            f"- pending hints on {rec['survivor']} before the restart: {rec['pending_hints_before_restart']} host(s); "
            f"delivered: {'yes' if rec['hints_delivered'] else 'NO'} (waited {rec['hints_wait_seconds']} s)",
            f"- rows written successfully in total (expected in every replica): {rec['expected_rows']}"]
    if "victim_count_after_hints" in rec:
        out.append(f"- {rec['victim']} alone after hint delivery: {rec['victim_count_after_hints']} rows")
    out.append(f"- `nodetool repair -full {result['keyspace']} {result['table']}` on {rec['victim']}: "
               f"{'ok' if rec['repair']['ok'] else 'FAILED'} in {rec['repair']['seconds']} s")
    if "replica_counts_after_repair" in rec:
        out.append("- each replica alone after repair: "
                   + ", ".join(f"{h} {c}" for h, c in rec["replica_counts_after_repair"].items()))
    out.append(f"- COUNT(*) at CL ALL after repair: {rec['count_all_after_repair']}")
    out.append(f"- converged: **{'yes' if rec['converged'] else 'NO'}**")
    out += ["", "## CAP interpretation", ""] + [f"- {line}" for line in interpret(result)] + [""]
    return "\n".join(out)


def write_report(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / "e6_cap.json", out_dir / "e6_cap.md"
    js.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    md.write_text(render_markdown(result), encoding="utf-8")
    return md, js


def default_hosts() -> list[str]:
    try:
        from cluster_common import load_runtime_env
        value = load_runtime_env().get("CASSANDRA_HOSTS")
    except Exception:  # noqa: BLE001
        value = None
    return CS.parse_hosts(value or "spark-w2,spark-w5")


def default_ssh_user() -> str:
    try:
        from cluster_common import load_runtime_env
        return load_runtime_env().get("SSH_USER") or "spark"
    except Exception:  # noqa: BLE001
        return "spark"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hosts", type=CS.hosts_arg, default=None,
                        help="Cassandra nodes (default CASSANDRA_HOSTS of cluster.runtime.env)")
    parser.add_argument("--victim", default=None, help="node to stop in (b) (default: the last host)")
    parser.add_argument("--rows", type=int, default=1000, help="rows written and read per CL and phase (default 1000)")
    parser.add_argument("--ssh-user", default=None, help="default SSH_USER of cluster.runtime.env, else spark")
    parser.add_argument("--cassandra-home", default="/opt/cassandra")
    parser.add_argument("--hint-timeout", type=float, default=300.0, help="seconds to wait for hint delivery")
    parser.add_argument("--no-isolated-counts", action="store_true",
                        help="skip the per-replica counts (each needs a stop + start of the other node)")
    parser.add_argument("--out", type=Path, default=None,
                        help="report directory (default docs/results; --dry-run: data/e6_dryrun)")
    parser.add_argument("--dry-run", action="store_true", help="simulated 2-node cluster: no driver, no ssh")
    args = parser.parse_args(argv)

    hosts = args.hosts or default_hosts()
    if len(hosts) < 2:
        parser.error("E6 needs at least 2 Cassandra hosts")
    victim = args.victim or hosts[-1]
    if victim not in hosts:
        parser.error(f"--victim {victim} is not one of {hosts}")
    if args.rows < 1:
        parser.error("--rows must be >= 1")

    if args.dry_run:
        backend = SimBackend(hosts)
    else:
        backend = RealBackend(hosts, args.ssh_user or default_ssh_user(), args.cassandra_home)
    try:
        result = run_experiment(backend, hosts, victim, args.rows, not args.no_isolated_counts, args.hint_timeout)
    finally:
        backend.close()
    result["dry_run"] = bool(args.dry_run)
    md, js = write_report(result, args.out or (DRY_OUT_DIR if args.dry_run else OUT_DIR))
    print(f"wrote {md.as_posix()} and {js.name}; converged: {result['recovery']['converged']}", flush=True)
    return 0 if result["recovery"]["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())
