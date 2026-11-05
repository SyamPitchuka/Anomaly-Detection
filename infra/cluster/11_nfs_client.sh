#!/usr/bin/env bash
# Run on every Spark VM that is NOT the NFS server (sudo), after 00_prereqs.sh and 10_nfs_server.sh.
# Mounts NFS_SERVER_HOST:SHARED_DIR at the same path, persistently via /etc/fstab.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_root

if is_nfs_server; then
    log "this VM is the NFS server; nothing to mount"
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get install -y --no-install-recommends nfs-common

# hard = never return I/O errors to Spark on a server hiccup; actimeo=1 keeps directory listings fresh
# enough for Spark's file-based checkpoint log (it lists and renames files).
opts="vers=4.1,hard,timeo=600,retrans=2,noatime,actimeo=1,_netdev"
sed -i "\|[[:space:]]${SHARED_DIR}[[:space:]]|d" /etc/fstab
echo "${NFS_SERVER_HOST}:${SHARED_DIR} ${SHARED_DIR} nfs4 ${opts} 0 0" >> /etc/fstab

mkdir -p "${SHARED_DIR}"
systemctl daemon-reload
mountpoint -q "${SHARED_DIR}" || mount "${SHARED_DIR}"

fstype="$(findmnt -n -o FSTYPE --target "${SHARED_DIR}")"
[[ "${fstype}" == "nfs4" ]] || die "${SHARED_DIR} is '${fstype}', expected nfs4"
log "mounted ${NFS_SERVER_HOST}:${SHARED_DIR} at ${SHARED_DIR}"
