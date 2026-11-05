"""Shared helpers for the cluster tools in infra/cluster.

Settings come from cluster.runtime.env (written by `deploy.sh env`, and shipped to the shared
repo copy by `deploy.sh repo`). A same-named environment variable overrides the file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME_ENV = HERE / "cluster.runtime.env"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_runtime_env() -> dict[str, str]:
    """Read key=value lines; environment variables override the file."""
    cfg: dict[str, str] = {}
    if RUNTIME_ENV.exists():
        for line in RUNTIME_ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    for key in list(cfg) + ["KAFKA_BOOTSTRAP", "SPARK_MASTER_URL"]:
        if key in os.environ:
            cfg[key] = os.environ[key]
    return cfg


def split_list(value: str | None) -> list[str]:
    return [v for v in (value or "").split(",") if v]


def broker_count(cfg: dict[str, str], default: int) -> int:
    """Number of Kafka brokers of this cluster (KAFKA_BROKER_COUNT); `default` if unknown or not a positive number."""
    try:
        n = int(cfg.get("KAFKA_BROKER_COUNT", ""))
    except ValueError:
        return default
    return n if n >= 1 else default


def effective_replication(wanted: int, brokers: int) -> int:
    """Replication factor a cluster with `brokers` brokers can hold: contracts/topics.yaml asks for
    `replication_cluster` (3), a one-broker cluster can only do 1. contracts/ itself is never edited."""
    return max(1, min(wanted, brokers))


def resolve_bootstrap(cli_value: str | None) -> str:
    """--bootstrap beats $KAFKA_BOOTSTRAP beats cluster.runtime.env; exits with a hint if none is set."""
    bootstrap = cli_value or load_runtime_env().get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        print("FAIL no Kafka bootstrap list. Use --bootstrap spark-master:9092 (or the full broker list), "
              "set KAFKA_BOOTSTRAP, or run: bash infra/cluster/deploy.sh env", file=sys.stderr)
        sys.exit(2)
    return bootstrap
