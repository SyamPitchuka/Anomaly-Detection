#!/usr/bin/env bash
# Run on every CASSANDRA_HOSTS VM (sudo), after 00_prereqs.sh (Java 17). deploy.sh phase 'cassandra' does it.
# Installs Apache Cassandra ${CASSANDRA_VERSION} (5.0.x binary tarball from the Apache archive, via the outbound proxy
# on VMs without internet) under /opt, with:
#   cluster_name ${CASSANDRA_CLUSTER_NAME}; every CASSANDRA_HOSTS node is a seed (their private IPs, port 7000);
#   listen_address = rpc_address = this VM's private IP (CQL on 9042, gossip on 7000, JMX on 127.0.0.1:7199 only);
#   heap CASSANDRA_HEAP (MAX_HEAP_SIZE; G1 on Java 17, so HEAP_NEWSIZE is ignored by cassandra-env.sh);
#   data, commit log, hints and saved caches under CASSANDRA_DATA_DIR, logs under /var/log/cassandra;
#   a systemd unit 'cassandra' running as the system user 'cassandra'.
# The serving keyspace (anomaly, RF 2) is created by src/serving/cassandra_store.py, not here.
# Idempotent: download / unpack only when the version is missing; config and unit rewritten only when they change;
# the service is restarted only when one of them changed (a running node with unchanged config is left alone).
#
#     sudo bash infra/cluster/40_cassandra.sh               # on a Cassandra VM
#     bash infra/cluster/40_cassandra.sh --print-config      # resolved settings (key=value), touches nothing (tests)
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

read -r -a cass_hosts <<< "${CASSANDRA_HOSTS}"
[[ "${#cass_hosts[@]}" -ge 1 ]] || die "CASSANDRA_HOSTS is empty"
CASSANDRA_DIST="apache-cassandra-${CASSANDRA_VERSION}"
CASSANDRA_URL="${APACHE_ARCHIVE}/cassandra/${CASSANDRA_VERSION}/${CASSANDRA_DIST}-bin.tar.gz"

# host_ip <name>   first IPv4 address of a cluster hostname (DNS or /etc/hosts)
host_ip() { getent ahostsv4 "$1" | awk 'NR == 1 {print $1}'; }

if [[ "${1:-}" == "--print-config" ]]; then
    cat <<EOF
CASSANDRA_HOSTS=${cass_hosts[*]}
CASSANDRA_VERSION=${CASSANDRA_VERSION}
CASSANDRA_URL=${CASSANDRA_URL}
CASSANDRA_HOME=${CASSANDRA_HOME}
CASSANDRA_HEAP=${CASSANDRA_HEAP}
CASSANDRA_CLUSTER_NAME=${CASSANDRA_CLUSTER_NAME}
CASSANDRA_DATA_DIR=${CASSANDRA_DATA_DIR}
CASSANDRA_CQL_PORT=${CASSANDRA_CQL_PORT}
EOF
    exit 0
fi

require_root
use_outbound_proxy

me="$(hostname -s)"
is_member=0
for h in "${cass_hosts[@]}"; do
    [[ "${h%%.*}" == "${me}" ]] && is_member=1
done
[[ "${is_member}" -eq 1 ]] || die "${me} is not in CASSANDRA_HOSTS (${CASSANDRA_HOSTS})"

# ---- Java 17 (installed by 00_prereqs.sh) ---------------------------------------------------
command -v java > /dev/null 2>&1 || die "java not found; run 00_prereqs.sh first"
java_major="$(java -version 2>&1 | awk -F'"' '/version/ {split($2, v, "."); print v[1]; exit}')"
[[ "${java_major}" == "17" ]] || die "Cassandra ${CASSANDRA_VERSION} is set up for Java 17 here, found Java ${java_major}"
java_home="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"

# ---- addresses ------------------------------------------------------------------------------
# The private IP: CASSANDRA_LISTEN_ADDRESS if set, else what this VM's own hostname resolves to, else hostname -I.
my_ip="${CASSANDRA_LISTEN_ADDRESS:-$(host_ip "${me}")}"
if [[ -z "${my_ip}" || "${my_ip}" == 127.* ]]; then
    my_ip="$(hostname -I | awk '{print $1}')"
fi
[[ "${my_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "cannot work out the private IP of ${me} (set CASSANDRA_LISTEN_ADDRESS)"
seeds=""
for h in "${cass_hosts[@]}"; do
    ip="$(host_ip "${h}")"
    [[ -n "${ip}" ]] || die "Cassandra host '${h}' does not resolve on ${me} (fix DNS or /etc/hosts)"
    seeds="${seeds:+${seeds},}${ip}:7000"
done
log "cassandra ${CASSANDRA_VERSION} on ${me}: listen ${my_ip}, seeds ${seeds}, heap ${CASSANDRA_HEAP}"

# ---- user, directories, kernel settings -----------------------------------------------------
if ! id cassandra > /dev/null 2>&1; then
    useradd --system --home-dir "${CASSANDRA_DATA_DIR}" --shell /usr/sbin/nologin cassandra
fi
install -d -o cassandra -g cassandra -m 0750 "${CASSANDRA_DATA_DIR}" "${CASSANDRA_DATA_DIR}/data" \
    "${CASSANDRA_DATA_DIR}/commitlog" "${CASSANDRA_DATA_DIR}/hints" "${CASSANDRA_DATA_DIR}/saved_caches" \
    "${CASSANDRA_DATA_DIR}/cdc_raw"
install -d -o cassandra -g cassandra -m 0755 /var/log/cassandra
# Recommended production settings: many memory-mapped SSTable segments, no swapping of the heap.
if printf 'vm.max_map_count = 1048575\nvm.swappiness = 1\n' | put_file /etc/sysctl.d/60-cassandra.conf 0644; then
    sysctl -q -p /etc/sysctl.d/60-cassandra.conf || warn "sysctl could not apply /etc/sysctl.d/60-cassandra.conf"
fi

# ---- binaries -------------------------------------------------------------------------------
if [[ ! -x "/opt/${CASSANDRA_DIST}/bin/cassandra" ]]; then
    tmp="$(mktemp -d)"
    log "downloading ${CASSANDRA_URL}"
    fetch "${CASSANDRA_URL}" "${tmp}/${CASSANDRA_DIST}-bin.tar.gz"
    verify_sha512 "${tmp}/${CASSANDRA_DIST}-bin.tar.gz" "${CASSANDRA_URL}.sha512"
    tar -xzf "${tmp}/${CASSANDRA_DIST}-bin.tar.gz" -C /opt
    rm -rf "${tmp}"
    chown -R cassandra:cassandra "/opt/${CASSANDRA_DIST}"
fi
ln -sfn "/opt/${CASSANDRA_DIST}" "${CASSANDRA_HOME}"
# keep the shipped config once: cassandra.yaml is regenerated from it on every run (idempotent edits)
[[ -f "${CASSANDRA_HOME}/conf/cassandra.yaml.dist" ]] \
    || cp "${CASSANDRA_HOME}/conf/cassandra.yaml" "${CASSANDRA_HOME}/conf/cassandra.yaml.dist"
# Wrappers, not symlinks: the scripts find cassandra.in.sh / cqlsh.py relative to their own path ($0).
for tool in nodetool cqlsh; do
    printf '#!/bin/sh\n# Generated by infra/cluster/40_cassandra.sh\nexec %s/bin/%s "$@"\n' "${CASSANDRA_HOME}" "${tool}" \
        | put_file "/usr/local/bin/${tool}" 0755 || true
done

# ---- cassandra.yaml (generated from the shipped file) ---------------------------------------
d="${CASSANDRA_DATA_DIR}"
yaml="$(sed \
    -e "s|^cluster_name:.*|cluster_name: '${CASSANDRA_CLUSTER_NAME}'|" \
    -e "s|^\([[:space:]]*- seeds:\).*|\1 \"${seeds}\"|" \
    -e "s|^listen_address:.*|listen_address: ${my_ip}|" \
    -e "s|^rpc_address:.*|rpc_address: ${my_ip}|" \
    -e "s|^native_transport_port:.*|native_transport_port: ${CASSANDRA_CQL_PORT}|" \
    -e "s|^# data_file_directories:|data_file_directories:|" \
    -e "s|^#     - /var/lib/cassandra/data\$|    - ${d}/data|" \
    -e "s|^# commitlog_directory:.*|commitlog_directory: ${d}/commitlog|" \
    -e "s|^# hints_directory:.*|hints_directory: ${d}/hints|" \
    -e "s|^# saved_caches_directory:.*|saved_caches_directory: ${d}/saved_caches|" \
    -e "s|^# cdc_raw_directory:.*|cdc_raw_directory: ${d}/cdc_raw|" \
    "${CASSANDRA_HOME}/conf/cassandra.yaml.dist")"
# The shipped layout can change between releases: fail loudly rather than start a node with the defaults.
for want in "cluster_name: '${CASSANDRA_CLUSTER_NAME}'" "- seeds: \"${seeds}\"" "listen_address: ${my_ip}" \
    "rpc_address: ${my_ip}" "data_file_directories:" "    - ${d}/data" "commitlog_directory: ${d}/commitlog" \
    "hints_directory: ${d}/hints" "saved_caches_directory: ${d}/saved_caches"; do
    grep -qF -- "${want}" <<< "${yaml}" || die "could not set '${want}' in cassandra.yaml (layout of ${CASSANDRA_VERSION} changed?)"
done

conf_changed=0
if printf '%s\n' "${yaml}" | put_file "${CASSANDRA_HOME}/conf/cassandra.yaml" 0644; then
    conf_changed=1
fi

# ---- systemd --------------------------------------------------------------------------------
if put_file /etc/systemd/system/cassandra.service 0644 <<EOF
# Generated by infra/cluster/40_cassandra.sh
[Unit]
Description=Apache Cassandra ${CASSANDRA_VERSION} (${CASSANDRA_CLUSTER_NAME})
After=network-online.target
Wants=network-online.target

[Service]
User=cassandra
Group=cassandra
Environment=JAVA_HOME=${java_home}
Environment=MAX_HEAP_SIZE=${CASSANDRA_HEAP}
Environment=HEAP_NEWSIZE=${CASSANDRA_HEAP_NEWSIZE}
Environment=CASSANDRA_LOG_DIR=/var/log/cassandra
ExecStart=${CASSANDRA_HOME}/bin/cassandra -f
Restart=on-failure
RestartSec=10
TimeoutStopSec=120
LimitNOFILE=100000
LimitNPROC=32768
LimitMEMLOCK=infinity
LimitAS=infinity

[Install]
WantedBy=multi-user.target
EOF
then
    conf_changed=1
    systemctl daemon-reload
fi

systemctl enable cassandra > /dev/null 2>&1
if systemctl is-active --quiet cassandra; then
    if [[ "${conf_changed}" -eq 1 ]]; then
        log "config changed: restarting cassandra"
        systemctl restart cassandra
    fi
else
    systemctl start cassandra
fi

log "waiting for CQL on ${my_ip}:${CASSANDRA_CQL_PORT}"
wait_port "${my_ip}" "${CASSANDRA_CQL_PORT}" 240 \
    || die "cassandra did not open ${my_ip}:${CASSANDRA_CQL_PORT}; see: sudo journalctl -u cassandra, /var/log/cassandra/system.log"
log "cassandra ready on ${me} (${my_ip}:${CASSANDRA_CQL_PORT})"
