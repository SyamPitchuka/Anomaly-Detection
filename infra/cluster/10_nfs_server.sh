#!/usr/bin/env bash
# Run on NFS_SERVER_HOST (sudo), after 00_prereqs.sh. Exports SHARED_DIR to NFS_NETWORK.
# Layout on the share (all owned by 'spark'):
#   $REPO_DIR                the repo checkout (same path on every node)
#   $REPO_DIR/checkpoints    Spark Structured Streaming checkpoints  (contract: checkpoints/<job_name>/)
#   $REPO_DIR/data           Parquet layers
#   $REPO_DIR/models         trained models
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_root
is_nfs_server || die "this VM is $(hostname -s) but NFS_SERVER_HOST=${NFS_SERVER_HOST}"

export DEBIAN_FRONTEND=noninteractive
apt-get install -y --no-install-recommends nfs-kernel-server

install -d -o spark -g spark -m 0775 "${SHARED_DIR}" "${REPO_DIR}" \
    "${REPO_DIR}/checkpoints" "${REPO_DIR}/data" "${REPO_DIR}/models"

# One export line for SHARED_DIR; replace it if it already exists. root_squash stays on (default):
# every node writes as the 'spark' user, which has the same uid everywhere.
sed -i "\|^${SHARED_DIR}[[:space:]]|d" /etc/exports
echo "${SHARED_DIR} ${NFS_NETWORK}(rw,sync,no_subtree_check)" >> /etc/exports

exportfs -ra
systemctl enable --now nfs-server
# (no 'grep -q' here: it can close the pipe early and trip 'pipefail')
if ! exportfs -v | grep "^${SHARED_DIR}" > /dev/null; then
    die "export of ${SHARED_DIR} not active"
fi
log "exporting ${SHARED_DIR} to ${NFS_NETWORK}"
