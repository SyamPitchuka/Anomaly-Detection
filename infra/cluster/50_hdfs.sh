#!/usr/bin/env bash
# Run on the HDFS_NAMENODE_HOST and every HDFS_DATANODE_HOSTS VM (sudo), after 00_prereqs.sh (Java 17).
# deploy.sh phase 'hdfs' does it (NameNode host first). E7: HDFS as the Spark checkpoint store, so streaming state
# no longer depends on the single NFS server. Optional; nothing uses it unless a run passes -k hdfs://...
#
# Installs Apache Hadoop ${HADOOP_VERSION} (binary tarball from the Apache archive, sha512-checked, via the outbound
# proxy on VMs without internet) under /opt, with:
#   fs.defaultFS ${HDFS_URI}; dfs.replication ${HDFS_REPLICATION};
#   NameNode on HDFS_NAMENODE_HOST (RPC :${HDFS_NAMENODE_PORT}, web :9870, heap HDFS_NAMENODE_HEAP), metadata in
#   HDFS_DATA_DIR/name, formatted ONCE (cluster id HDFS_CLUSTER_ID; an existing name dir is never re-formatted);
#   a DataNode on each HDFS_DATANODE_HOSTS VM (:9866 data, :9864 web, :9867 ipc, heap HDFS_DATANODE_HEAP), blocks in
#   HDFS_DATA_DIR/data (local disk);
#   dead-node detection after 2 x HDFS_HEARTBEAT_RECHECK_MS + 30 s (90 s by default) and stale nodes (30 s without a
#   heartbeat) avoided for reads and writes;
#   systemd units hadoop-namenode / hadoop-datanode running as the system user 'hdfs' (the HDFS superuser),
#   logs in /var/log/hadoop; a wrapper /usr/local/bin/hdfs that sets JAVA_HOME / HADOOP_CONF_DIR for any user.
# Idempotent: download / unpack only when the version is missing; config and units rewritten only when they change;
# a service is restarted only when its config changed.
#
#     sudo bash infra/cluster/50_hdfs.sh                 # on an HDFS VM
#     sudo bash infra/cluster/50_hdfs.sh --init-dirs     # on the NameNode VM, DataNodes up: leave safe mode, create
#                                                        # HDFS_CHECKPOINT_ROOT owned by spark (deploy.sh does this)
#     bash infra/cluster/50_hdfs.sh --print-config        # resolved settings (key=value), touches nothing (tests)
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

read -r -a dn_hosts <<< "${HDFS_DATANODE_HOSTS}"
[[ -n "${HDFS_NAMENODE_HOST}" ]] || die "HDFS_NAMENODE_HOST is empty"
[[ "${#dn_hosts[@]}" -ge 1 ]] || die "HDFS_DATANODE_HOSTS is empty"
[[ "${HDFS_REPLICATION}" =~ ^[1-9]$ ]] || die "HDFS_REPLICATION must be 1..9"
[[ "${HDFS_REPLICATION}" -le "${#dn_hosts[@]}" ]] \
    || die "HDFS_REPLICATION=${HDFS_REPLICATION} exceeds the ${#dn_hosts[@]} DataNode(s) in HDFS_DATANODE_HOSTS"
HADOOP_DIST="hadoop-${HADOOP_VERSION}"
HADOOP_URL="${APACHE_ARCHIVE}/hadoop/common/${HADOOP_DIST}/${HADOOP_DIST}.tar.gz"
CONF_DIR="${HADOOP_INSTALL_DIR}/etc/hadoop"
NAME_DIR="${HDFS_DATA_DIR}/name"
DATA_DIR="${HDFS_DATA_DIR}/data"
LOG_DIR=/var/log/hadoop
# Hadoop 3.3 is built for Java 8/11; on Java 17 a few reflective accesses need the packages opened.
JAVA17_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED"

if [[ "${1:-}" == "--print-config" ]]; then
    cat <<EOF
HDFS_NAMENODE_HOST=${HDFS_NAMENODE_HOST}
HDFS_DATANODE_HOSTS=${dn_hosts[*]}
HADOOP_VERSION=${HADOOP_VERSION}
HADOOP_URL=${HADOOP_URL}
HADOOP_INSTALL_DIR=${HADOOP_INSTALL_DIR}
HDFS_URI=${HDFS_URI}
HDFS_NAMENODE_HEAP=${HDFS_NAMENODE_HEAP}
HDFS_DATANODE_HEAP=${HDFS_DATANODE_HEAP}
HDFS_REPLICATION=${HDFS_REPLICATION}
NAME_DIR=${NAME_DIR}
DATA_DIR=${DATA_DIR}
HDFS_CLUSTER_ID=${HDFS_CLUSTER_ID}
HDFS_CHECKPOINT_ROOT=${HDFS_CHECKPOINT_ROOT}
HDFS_HEARTBEAT_RECHECK_MS=${HDFS_HEARTBEAT_RECHECK_MS}
EOF
    exit 0
fi

require_root
me="$(hostname -s)"
is_nn=0; is_dn=0
[[ "${HDFS_NAMENODE_HOST%%.*}" == "${me}" ]] && is_nn=1
for h in "${dn_hosts[@]}"; do
    [[ "${h%%.*}" == "${me}" ]] && is_dn=1
done

as_hdfs() { runuser -u hdfs -- "$@"; }

# ---- --init-dirs: after the DataNodes registered (NameNode VM only) ---------------------------
if [[ "${1:-}" == "--init-dirs" ]]; then
    [[ "${is_nn}" -eq 1 ]] || die "--init-dirs runs on the NameNode VM ${HDFS_NAMENODE_HOST}, not ${me}"
    log "waiting for the NameNode to leave safe mode"
    timeout 300 runuser -u hdfs -- /usr/local/bin/hdfs dfsadmin -safemode wait > /dev/null \
        || die "NameNode still in safe mode after 300 s; see: hdfs dfsadmin -report, journalctl -u hadoop-namenode"
    as_hdfs /usr/local/bin/hdfs dfs -mkdir -p "${HDFS_CHECKPOINT_ROOT}"
    as_hdfs /usr/local/bin/hdfs dfs -chown spark:spark "${HDFS_CHECKPOINT_ROOT}"
    as_hdfs /usr/local/bin/hdfs dfs -chmod 0775 "${HDFS_CHECKPOINT_ROOT}"
    log "HDFS checkpoint root ${HDFS_URI}${HDFS_CHECKPOINT_ROOT} (owner spark)"
    exit 0
fi

[[ "${is_nn}" -eq 1 || "${is_dn}" -eq 1 ]] \
    || die "${me} is neither HDFS_NAMENODE_HOST (${HDFS_NAMENODE_HOST}) nor in HDFS_DATANODE_HOSTS (${HDFS_DATANODE_HOSTS})"
use_outbound_proxy

# ---- Java 17 (installed by 00_prereqs.sh) ---------------------------------------------------
command -v java > /dev/null 2>&1 || die "java not found; run 00_prereqs.sh first"
java_major="$(java -version 2>&1 | awk -F'"' '/version/ {split($2, v, "."); print v[1]; exit}')"
[[ "${java_major}" == "17" ]] || die "expected Java 17 (contract), found Java ${java_major}"
java_home="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"

# ---- memory: the heaps must fit next to what already runs here ----------------------------------
need_mb=0
[[ "${is_nn}" -eq 1 ]] && need_mb=$((need_mb + $(mem_to_mb "${HDFS_NAMENODE_HEAP}") + 256))
[[ "${is_dn}" -eq 1 ]] && need_mb=$((need_mb + $(mem_to_mb "${HDFS_DATANODE_HEAP}") + 128))
running=0
systemctl is-active --quiet hadoop-namenode 2> /dev/null && running=1
systemctl is-active --quiet hadoop-datanode 2> /dev/null && running=1
avail_mb="$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
if [[ "${running}" -eq 0 && "${avail_mb}" -lt "${need_mb}" ]]; then
    die "${me} has ${avail_mb} MB available, the HDFS daemons here need ~${need_mb} MB (pick another VM or lower the heaps)"
fi
log "hadoop ${HADOOP_VERSION} on ${me}: namenode=${is_nn} datanode=${is_dn}, fs ${HDFS_URI}, replication ${HDFS_REPLICATION} (${avail_mb} MB available)"

# ---- user, directories ----------------------------------------------------------------------
if ! id hdfs > /dev/null 2>&1; then
    useradd --system --home-dir "${HDFS_DATA_DIR}" --shell /usr/sbin/nologin hdfs
fi
install -d -o hdfs -g hdfs -m 0755 "${HDFS_DATA_DIR}" "${LOG_DIR}" "${HDFS_DATA_DIR}/tmp"
[[ "${is_nn}" -eq 1 ]] && install -d -o hdfs -g hdfs -m 0700 "${NAME_DIR}"
[[ "${is_dn}" -eq 1 ]] && install -d -o hdfs -g hdfs -m 0700 "${DATA_DIR}"

# ---- binaries -------------------------------------------------------------------------------
if [[ ! -x "/opt/${HADOOP_DIST}/bin/hdfs" ]]; then
    tmp="$(mktemp -d)"
    log "downloading ${HADOOP_URL} (~700 MB)"
    fetch "${HADOOP_URL}" "${tmp}/${HADOOP_DIST}.tar.gz"
    verify_sha512 "${tmp}/${HADOOP_DIST}.tar.gz" "${HADOOP_URL}.sha512"
    tar -xzf "${tmp}/${HADOOP_DIST}.tar.gz" -C /opt --exclude="${HADOOP_DIST}/share/doc"
    rm -rf "${tmp}"
    chown -R root:root "/opt/${HADOOP_DIST}"
fi
ln -sfn "/opt/${HADOOP_DIST}" "${HADOOP_INSTALL_DIR}"
printf '#!/bin/sh\n# Generated by infra/cluster/50_hdfs.sh\nexport JAVA_HOME=%s HADOOP_HOME=%s HADOOP_CONF_DIR=%s\nexec %s/bin/hdfs "$@"\n' \
    "${java_home}" "${HADOOP_INSTALL_DIR}" "${CONF_DIR}" "${HADOOP_INSTALL_DIR}" | put_file /usr/local/bin/hdfs 0755 || true

# ---- configuration --------------------------------------------------------------------------
prop() { printf '  <property><name>%s</name><value>%s</value></property>\n' "$1" "$2"; }
conf_changed=0
if { echo '<?xml version="1.0"?>'; echo '<!-- Generated by infra/cluster/50_hdfs.sh -->'; echo '<configuration>'
     prop fs.defaultFS "${HDFS_URI}"
     prop hadoop.tmp.dir "${HDFS_DATA_DIR}/tmp"
     echo '</configuration>'; } | put_file "${CONF_DIR}/core-site.xml" 0644; then
    conf_changed=1
fi
if { echo '<?xml version="1.0"?>'; echo '<!-- Generated by infra/cluster/50_hdfs.sh -->'; echo '<configuration>'
     prop dfs.replication "${HDFS_REPLICATION}"
     prop dfs.namenode.name.dir "file://${NAME_DIR}"
     prop dfs.datanode.data.dir "file://${DATA_DIR}"
     prop dfs.namenode.rpc-address "${HDFS_NAMENODE_HOST}:${HDFS_NAMENODE_PORT}"
     prop dfs.namenode.rpc-bind-host 0.0.0.0
     prop dfs.namenode.servicerpc-bind-host 0.0.0.0
     prop dfs.namenode.http-bind-host 0.0.0.0
     # Azure: VMs in the peered VNet may not have reverse DNS for each other
     prop dfs.namenode.datanode.registration.ip-hostname-check false
     prop dfs.namenode.heartbeat.recheck-interval "${HDFS_HEARTBEAT_RECHECK_MS}"
     prop dfs.namenode.stale.datanode.interval 30000
     prop dfs.namenode.avoid.read.stale.datanode true
     prop dfs.namenode.avoid.write.stale.datanode true
     prop dfs.permissions.superusergroup hdfs
     echo '</configuration>'; } | put_file "${CONF_DIR}/hdfs-site.xml" 0644; then
    conf_changed=1
fi

# ---- format the NameNode once ---------------------------------------------------------------
if [[ "${is_nn}" -eq 1 && ! -f "${NAME_DIR}/current/VERSION" ]]; then
    log "formatting the NameNode (cluster id ${HDFS_CLUSTER_ID}); this happens only once"
    as_hdfs env HADOOP_LOG_DIR="${LOG_DIR}" /usr/local/bin/hdfs namenode -format -nonInteractive \
        -clusterid "${HDFS_CLUSTER_ID}" > "${LOG_DIR}/format.log" 2>&1 \
        || { tail -n 20 "${LOG_DIR}/format.log" >&2; die "namenode -format failed"; }
fi

# ---- systemd --------------------------------------------------------------------------------
# unit <name> <daemon> <heap> <description>   returns 0 if the unit file changed
unit() {
    put_file "/etc/systemd/system/$1.service" 0644 <<EOF
# Generated by infra/cluster/50_hdfs.sh
[Unit]
Description=$4 (Apache Hadoop ${HADOOP_VERSION}, ${HDFS_URI})
After=network-online.target
Wants=network-online.target

[Service]
User=hdfs
Group=hdfs
Environment=JAVA_HOME=${java_home}
Environment=HADOOP_HOME=${HADOOP_INSTALL_DIR}
Environment=HADOOP_CONF_DIR=${CONF_DIR}
Environment=HADOOP_LOG_DIR=${LOG_DIR}
Environment="HADOOP_OPTS=${JAVA17_OPTS}"
Environment="HDFS_$(echo "$2" | tr '[:lower:]' '[:upper:]')_OPTS=-Xms$3 -Xmx$3"
ExecStart=${HADOOP_INSTALL_DIR}/bin/hdfs $2
Restart=on-failure
RestartSec=10
TimeoutStopSec=60
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
}

# start_unit <name> <changed 0|1>
start_unit() {
    systemctl enable "$1" > /dev/null 2>&1
    if systemctl is-active --quiet "$1"; then
        if [[ "$2" -eq 1 ]]; then
            log "config changed: restarting $1"
            systemctl restart "$1"
        fi
    else
        systemctl start "$1"
    fi
}

if [[ "${is_nn}" -eq 1 ]]; then
    changed="${conf_changed}"
    unit hadoop-namenode namenode "${HDFS_NAMENODE_HEAP}" "HDFS NameNode" && changed=1
    systemctl daemon-reload
    start_unit hadoop-namenode "${changed}"
    log "waiting for the NameNode RPC on ${HDFS_NAMENODE_HOST}:${HDFS_NAMENODE_PORT}"
    wait_port "${HDFS_NAMENODE_HOST}" "${HDFS_NAMENODE_PORT}" 180 \
        || die "NameNode did not open :${HDFS_NAMENODE_PORT}; see: sudo journalctl -u hadoop-namenode, ${LOG_DIR}"
fi
if [[ "${is_dn}" -eq 1 ]]; then
    changed="${conf_changed}"
    unit hadoop-datanode datanode "${HDFS_DATANODE_HEAP}" "HDFS DataNode" && changed=1
    systemctl daemon-reload
    start_unit hadoop-datanode "${changed}"
    log "waiting for the DataNode on :9866"
    wait_port 127.0.0.1 9866 180 || die "DataNode did not open :9866; see: sudo journalctl -u hadoop-datanode, ${LOG_DIR}"
fi
log "hdfs ready on ${me}"
