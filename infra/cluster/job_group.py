"""Run score_job in its own process group and stop it without ever blocking forever (used by drain_test.sh).

    python job_group.py start --log LOG -- bash submit.sh ...    # run in the background: '... &'; PGID = $!
    python job_group.py stop --pgid PGID [--int-wait 60] [--term-wait 20] [--kill-wait 10]
    python job_group.py wait-apps --master http://<master>:8080/json/ [--timeout 60]

start   puts the job in a new session (setsid), so the job's process group id is this process's pid ($! in bash),
        restores the default action of SIGINT / SIGQUIT / SIGTERM / SIGPIPE (a background job of a non-interactive
        bash starts with SIGINT ignored, and the JVM then ignores it too), sends stdout and stderr only to LOG and
        reads stdin from /dev/null (nothing is shared with the caller's terminal or pipe), then execs the command.
stop    SIGINT to the group, wait up to --int-wait s; then SIGTERM, wait --term-wait s; then SIGKILL, wait
        --kill-wait s. Prints which step ended the job. Exit 0 if the group is gone, 1 if it survived SIGKILL.
wait-apps  poll the standalone master's JSON page until it lists no active application (so the next run gets its
        executors). Exit 0 when none is left, 1 (with a warning) if one remains after --timeout s.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

POLL_S = 1.0
# (name, number) of the escalation steps; numbers spelled out so the logic also imports and tests on Windows
STEPS = (("SIGINT", int(getattr(signal, "SIGINT", 2))), ("SIGTERM", int(getattr(signal, "SIGTERM", 15))),
         ("SIGKILL", int(getattr(signal, "SIGKILL", 9))))


# ---- process group ----------------------------------------------------------------------------------------

def group_alive(pgid: int, proc: Path = Path("/proc")) -> bool:
    """True if any non-zombie process is in process group `pgid`. A zombie member (the job's leader, exited but not
    yet reaped by its parent shell) does not count: kill(-pgid, 0) would still succeed on it."""
    if proc.is_dir():
        for d in proc.iterdir():
            if not d.name.isdigit():
                continue
            try:
                stat = (d / "stat").read_text()
            except OSError:
                continue
            # pid (comm) state ppid pgrp ...; comm may hold spaces or ')', so split after the last ')'
            fields = stat[stat.rfind(")") + 2:].split()
            if len(fields) >= 3 and int(fields[2]) == pgid and fields[0] != "Z":
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_group(pgid: int, *, int_wait: float = 60, term_wait: float = 20, kill_wait: float = 10,
               send: Callable[[int, int], None] | None = None, alive: Callable[[int], bool] | None = None,
               sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.monotonic,
               poll_s: float = POLL_S) -> tuple[str, float]:
    """Escalate SIGINT -> SIGTERM -> SIGKILL on process group `pgid`, each step bounded in time.
    Returns (what ended it, seconds taken): 'gone' (nothing to stop), 'SIGINT', 'SIGTERM', 'SIGKILL', or 'survived'
    (still there after SIGKILL + kill_wait). Never blocks longer than int_wait + term_wait + kill_wait (+ one poll)."""
    send = send or os.killpg
    alive = alive or group_alive
    t0 = now()
    if not alive(pgid):
        return "gone", 0.0
    for (name, sig), wait_s in zip(STEPS, (int_wait, term_wait, kill_wait)):
        try:
            send(pgid, sig)
        except ProcessLookupError:
            return name, now() - t0
        deadline = now() + wait_s
        while True:
            if not alive(pgid):
                return name, now() - t0
            if now() >= deadline:
                break
            sleep(min(poll_s, max(0.0, deadline - now())))
    return "survived", now() - t0


# ---- standalone master ------------------------------------------------------------------------------------

def fetch_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (Spark master UI)
        return json.loads(r.read().decode("utf-8"))


def active_apps(page: dict) -> list[str]:
    """Ids of the running applications on the master's /json/ page ('activeapps'; WAITING ones count too)."""
    return [str(a.get("id", "?")) for a in page.get("activeapps", []) or []]


def wait_no_active_apps(url: str, *, timeout: float = 60, fetch: Callable[[str], dict] = fetch_json,
                        sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.monotonic,
                        poll_s: float = 2.0) -> list[str]:
    """Poll until the master lists no active app. Returns the apps still active at the deadline ([] = all clear).
    An unreachable master counts as 'unknown' and is retried until the deadline."""
    deadline = now() + timeout
    left: list[str] = ["<master unreachable>"]
    while True:
        try:
            left = active_apps(fetch(url))
        except (OSError, ValueError):
            left = ["<master unreachable>"]
        if not left or now() >= deadline:
            return left
        sleep(min(poll_s, max(0.0, deadline - now())))


# ---- commands ---------------------------------------------------------------------------------------------

def cmd_start(args) -> int:
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        print("job_group.py start: no command given", file=sys.stderr)
        return 2
    for sig in (signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGPIPE, signal.SIGXFSZ):
        signal.signal(sig, signal.SIG_DFL)
    if os.getpgrp() == os.getpid():
        # already a group leader (would make setsid fail): cannot happen for a '... &' job of a non-interactive
        # bash, but keep the contract PGID == our pid honest
        print("job_group.py start: already a process group leader", file=sys.stderr)
        return 2
    os.setsid()
    log = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(null)
    os.close(log)
    os.execvp(cmd[0], cmd)
    return 127  # not reached


def cmd_stop(args) -> int:
    how, took = stop_group(args.pgid, int_wait=args.int_wait, term_wait=args.term_wait, kill_wait=args.kill_wait)
    msg = {"gone": f"process group {args.pgid} had already exited",
           "survived": f"process group {args.pgid} is STILL RUNNING after SIGKILL ({took:.0f} s)"}.get(
        how, f"process group {args.pgid} ended after {how} ({took:.0f} s)")
    print(f"[job_group] {msg}", flush=True)
    return 1 if how == "survived" else 0


def cmd_wait_apps(args) -> int:
    left = wait_no_active_apps(args.master, timeout=args.timeout)
    if left:
        print(f"[job_group] WARNING: the Spark master still lists active application(s) after {args.timeout:.0f} s: "
              f"{', '.join(left)}; the next run may not get its executors", file=sys.stderr, flush=True)
        return 1
    print("[job_group] no active application on the Spark master", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="setsid + exec the command, output only to --log")
    s.add_argument("--log", required=True)
    s.add_argument("command", nargs=argparse.REMAINDER)
    t = sub.add_parser("stop", help="SIGINT -> SIGTERM -> SIGKILL on a process group")
    t.add_argument("--pgid", type=int, required=True)
    t.add_argument("--int-wait", type=float, default=60)
    t.add_argument("--term-wait", type=float, default=20)
    t.add_argument("--kill-wait", type=float, default=10)
    w = sub.add_parser("wait-apps", help="wait until the standalone master has no active application")
    w.add_argument("--master", required=True, help="http://<master>:8080/json/")
    w.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args(argv)
    return {"start": cmd_start, "stop": cmd_stop, "wait-apps": cmd_wait_apps}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
