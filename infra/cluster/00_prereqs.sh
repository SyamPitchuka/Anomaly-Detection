#!/usr/bin/env bash
# Run on EVERY VM (sudo). Installs Java 17 + base tools and creates the 'spark' user.
# 'spark' gets a fixed uid/gid so NFS file ownership is identical on all nodes.
# Kafka needs no host user or JDK: it runs in Docker (20_kafka_broker.sh installs Docker on KAFKA_HOST).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_root
validate_layout

export DEBIAN_FRONTEND=noninteractive
log "installing packages"
apt-get update -y
apt-get install -y --no-install-recommends openjdk-17-jre-headless curl ca-certificates tar gzip procps netcat-openbsd chrony

java_major="$(java -version 2>&1 | awk -F '"' '/version/ {split($2, v, "."); print v[1]}')"
[[ "${java_major}" == "17" ]] || die "default java is ${java_major}, contract says Java 17 (use update-alternatives)"

if ! getent group spark > /dev/null; then
    groupadd --gid "${SPARK_GID}" spark
fi
if ! id spark > /dev/null 2>&1; then
    useradd --system --uid "${SPARK_UID}" --gid spark --home-dir /var/lib/spark --create-home --shell /usr/sbin/nologin spark
fi
resolve_check
log "prereqs done (java ${java_major}, user spark)"
