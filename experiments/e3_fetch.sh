#!/usr/bin/env bash
# Copy an E3 run folder from the cluster to this machine: <local repo>/data/e3/<run-id>/ (git-ignored, see .gitignore).
# The folder holds the clean/ and fault/ snapshots (score Parquet + anomalies topic dump), report.md/json, timings and logs.
#
# Run on the admin machine (laptop), from anywhere:
#     E3_VM=<user>@<spark-master host or IP> bash experiments/e3_fetch.sh              # newest run on the VM
#     E3_VM=<user>@<host> bash experiments/e3_fetch.sh e3_20261115T170457Z             # a named run
#   E3_VM          ssh target of spark-master (required; not stored in the repo)
#   E3_REMOTE_DIR  where e3_faults.sh writes runs (default /srv/spark-shared/Anomaly-Detection/data/e3)
# e3_faults.sh prints this command with the right run id when it finishes.
set -euo pipefail
: "${E3_VM:?set E3_VM=<user>@<host> (ssh target of spark-master)}"
REMOTE_DIR="${E3_REMOTE_DIR:-/srv/spark-shared/Anomaly-Detection/data/e3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-$(ssh -o BatchMode=yes "${E3_VM}" "ls -t '${REMOTE_DIR}' | head -n 1")}"
[[ -n "${RUN_ID}" ]] || { echo "ERROR: no run found in ${REMOTE_DIR}" >&2; exit 1; }
DEST="${REPO}/data/e3/${RUN_ID}"
[[ ! -e "${DEST}" ]] || { echo "ERROR: ${DEST} already exists; move or delete it first" >&2; exit 1; }
mkdir -p "${DEST}"
trap 'rc=$?; [[ ${rc} -eq 0 ]] || rm -rf "${DEST}"' EXIT      # a failed fetch leaves nothing behind
trap 'exit 143' INT TERM                                                         # so the EXIT trap also runs on Ctrl+C / kill
echo "fetching ${E3_VM}:${REMOTE_DIR}/${RUN_ID} -> ${DEST}"
ssh -o BatchMode=yes "${E3_VM}" "tar -C '${REMOTE_DIR}/${RUN_ID}' -cf - ." | tar -C "${DEST}" -xf -
echo "done: $(find "${DEST}" -type f | wc -l) files, $(du -sh "${DEST}" | cut -f1); report: ${DEST}/report.md"
