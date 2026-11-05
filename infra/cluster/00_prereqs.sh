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
# A pre-existing 'spark' account (for example the Azure admin user, uid 1000) is kept as it is, so its ids must match
# the contract. Otherwise NFS file ownership differs between VMs and checkpoints cannot be written.
[[ "$(id -u spark)" == "${SPARK_UID}" ]] \
    || die "user spark has uid $(id -u spark) on $(hostname -s) but SPARK_UID=${SPARK_UID}; fix SPARK_UID in cluster.env or recreate the user"
[[ "$(id -g spark)" == "${SPARK_GID}" ]] \
    || die "user spark has gid $(id -g spark) on $(hostname -s) but SPARK_GID=${SPARK_GID}; fix SPARK_GID in cluster.env or recreate the user"
resolve_check
log "prereqs done (java ${java_major}, user spark)"
