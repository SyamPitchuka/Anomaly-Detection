#!/usr/bin/env bash
# Generate the KRaft cluster id once and store it in cluster.env (same id on all 3 brokers).
# Runs on the admin machine (needs only bash, base64, sed). Safe to re-run: it never overwrites an id.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${CLUSTER_ENV:-${HERE}/cluster.env}"

[[ -f "${ENV_FILE}" ]] || { echo "ERROR: ${ENV_FILE} missing; copy cluster.env.example first" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -n "${KAFKA_CLUSTER_ID:-}" ]]; then
    echo "KAFKA_CLUSTER_ID already set: ${KAFKA_CLUSTER_ID}"
    exit 0
fi

# Kafka ids are 16 random bytes, URL-safe base64 without padding (22 chars); they must not start with '-'.
id="-"
while [[ "${id}" == -* ]]; do
    id="$(head -c 16 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=\n')"
done

tmp="$(mktemp)"
sed "s|^KAFKA_CLUSTER_ID=.*|KAFKA_CLUSTER_ID=${id}|" "${ENV_FILE}" > "${tmp}"
cat "${tmp}" > "${ENV_FILE}"
rm -f "${tmp}"
echo "KAFKA_CLUSTER_ID=${id} written to ${ENV_FILE}"
