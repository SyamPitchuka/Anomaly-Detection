"""infra/cluster/job_group.py: SIGINT -> SIGTERM -> SIGKILL escalation, the master's active-app poll and the
process-group liveness check (fake clock / fake /proc; one real-process test on POSIX only)."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "infra" / "cluster"
if str(CLUSTER) not in sys.path:
    sys.path.insert(0, str(CLUSTER))

import job_group as J  # noqa: E402

INT, TERM, KILL = (n for _, n in J.STEPS)


class FakeGroup:
    """A process group on a fake clock that dies `delay` s after receiving one of the signals in `dies_on`."""

    def __init__(self, dies_on=(), delay=0.0, alive=True):
        self.t = 0.0
        self.sent: list[tuple[float, int]] = []
        self.dies_on, self.delay, self.dead_at = set(dies_on), delay, (None if alive else -1.0)

    def now(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.t += max(s, 1e-3)

    def send(self, pgid, sig):
        assert pgid == 4242
        self.sent.append((self.t, sig))
        if sig in self.dies_on and self.dead_at is None:
            self.dead_at = self.t + self.delay

    def alive(self, pgid):
        return self.dead_at is None or self.t < self.dead_at

    def stop(self, **kw):
        return J.stop_group(4242, send=self.send, alive=self.alive, sleep=self.sleep, now=self.now, **kw)


def test_sigint_is_enough():
    g = FakeGroup(dies_on={INT}, delay=12)
    how, took = g.stop()
    assert how == "SIGINT" and 12 <= took <= 13 and [s for _, s in g.sent] == [INT]


def test_escalates_to_sigterm_after_the_int_wait():
    g = FakeGroup(dies_on={TERM}, delay=3)
    how, took = g.stop(int_wait=60, term_wait=20)
    assert how == "SIGTERM" and [s for _, s in g.sent] == [INT, TERM]
    assert g.sent[1][0] == pytest.approx(60, abs=1e-6) and 63 <= took <= 64


def test_escalates_to_sigkill_after_int_and_term_waits():
    g = FakeGroup(dies_on={KILL}, delay=0.5)
    how, took = g.stop(int_wait=60, term_wait=20)
    assert how == "SIGKILL" and [s for _, s in g.sent] == [INT, TERM, KILL]
    assert g.sent[2][0] == pytest.approx(80, abs=1e-6) and took < 82


def test_never_blocks_forever_on_an_unkillable_group():
    g = FakeGroup(dies_on=())
    how, took = g.stop(int_wait=60, term_wait=20, kill_wait=10)
    assert how == "survived" and took == pytest.approx(90, abs=1e-6) and len(g.sent) == 3


def test_already_gone_and_vanishing_during_send():
    g = FakeGroup(alive=False)
    assert g.stop() == ("gone", 0.0) and g.sent == []

    def gone(pgid, sig):
        raise ProcessLookupError
    how, _ = J.stop_group(1, send=gone, alive=lambda p: True, sleep=lambda s: None, now=time.monotonic)
    assert how == "SIGINT"


def test_wait_no_active_apps():
    clock = {"t": 0.0}
    now = lambda: clock["t"]                                         # noqa: E731
    sleep = lambda s: clock.__setitem__("t", clock["t"] + s)         # noqa: E731
    pages = iter([{"activeapps": [{"id": "app-1"}]}, {"activeapps": [{"id": "app-1"}]}, {"activeapps": []}])
    assert J.wait_no_active_apps("u", timeout=60, fetch=lambda u: next(pages), sleep=sleep, now=now) == []
    assert clock["t"] == pytest.approx(4)

    clock["t"] = 0.0
    left = J.wait_no_active_apps("u", timeout=60, fetch=lambda u: {"activeapps": [{"id": "app-9"}]},
                                 sleep=sleep, now=now)
    assert left == ["app-9"] and clock["t"] == pytest.approx(60)

    def down(u):
        raise OSError("connection refused")
    clock["t"] = 0.0
    assert J.wait_no_active_apps("u", timeout=5, fetch=down, sleep=sleep, now=now) == ["<master unreachable>"]
    assert J.active_apps({"activeapps": [], "completedapps": [{"id": "old"}]}) == []


def test_group_alive_ignores_zombies_and_other_groups(tmp_path):
    def proc(pid, comm, state, pgrp):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} ({comm}) {state} 1 {pgrp} {pgrp} 0 -1 4194304\n")
    (tmp_path / "self").mkdir()
    proc(500, "bash", "Z", 500)                    # the exited, unreaped leader
    proc(501, "java) S 7 (x", "S", 600)            # other group; comm with ') ' in it
    assert not J.group_alive(500, proc=tmp_path)
    proc(502, "python3", "S", 500)
    assert J.group_alive(500, proc=tmp_path)


def test_stop_cli_message(capsys, monkeypatch):
    monkeypatch.setattr(J, "stop_group", lambda pgid, **kw: ("SIGTERM", 61.2))
    assert J.main(["stop", "--pgid", "77"]) == 0
    assert "process group 77 ended after SIGTERM (61 s)" in capsys.readouterr().out
    monkeypatch.setattr(J, "stop_group", lambda pgid, **kw: ("survived", 90.0))
    assert J.main(["stop", "--pgid", "77"]) == 1


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="process groups are POSIX only")
def test_real_job_in_own_group_ignoring_sigint_is_stopped_by_sigterm(tmp_path):
    # Mimics drain_test.sh: background start from a non-interactive bash, output only to the log; the job ignores
    # SIGINT (like a hung JVM/driver pair), so the escalation must end it with SIGTERM, well within the waits.
    log = tmp_path / "job.log"
    script = (f"{sys.executable} {CLUSTER / 'job_group.py'} start --log {log} -- "
              f"bash -c \"trap '' INT; echo started; sleep 30 & wait\" < /dev/null > /dev/null 2>&1 & echo $!")
    pgid = int(subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10).stdout.strip())
    for _ in range(50):
        if log.exists() and "started" in log.read_text():
            break
        time.sleep(0.1)
    assert os.getpgid(pgid) == pgid                  # its own process group, led by the pid bash saw as $!
    how, took = J.stop_group(pgid, int_wait=2, term_wait=5, kill_wait=5, poll_s=0.1)
    assert how == "SIGTERM" and took < 7
    assert not J.group_alive(pgid)
