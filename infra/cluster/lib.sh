#!/usr/bin/env bash
# Shared helpers for infra/cluster. Source this file; do not execute it.
# It does not call 'set -e', so it is safe to source from an interactive shell:
#     source infra/cluster/lib.sh && echo "$KAFKA_BOOTSTRAP"
# shellcheck shell=bash

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_ENV="${CLUSTER_ENV:-${CLUSTER_DIR}/cluster.env}"

if [[ ! -f "${CLUSTER_ENV}" ]]; then
    echo "ERROR: ${CLUSTER_ENV} not found. Copy cluster.env.example to cluster.env and edit it." >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "${CLUSTER_ENV}"

# VM-size settings (memory, cores, heaps) come from profiles/<VM_PROFILE>.env. The profile only fills in values
# that cluster.env has not set (its lines are ': "${VAR:=default}"'), so cluster.env can override single values.
VM_PROFILE="${VM_PROFILE:-vm_8gb_4cpu}"
PROFILE_FILE="${CLUSTER_DIR}/profiles/${VM_PROFILE}.env"
if [[ ! -f "${PROFILE_FILE}" ]]; then
    echo "ERROR: VM profile '${VM_PROFILE}' not found. Available: $(cd "${CLUSTER_DIR}/profiles" && ls ./*.env | sed 's|^\./||; s|\.env$||' | tr '\n' ' ')" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "${PROFILE_FILE}"

SSH_USER="${SSH_USER:-ubuntu}"
APACHE_ARCHIVE="${APACHE_ARCHIVE:-https://archive.apache.org/dist}"

# ---- fixed locations ----------------------------------------------------------------------
KAFKA_IMAGE="${KAFKA_IMAGE:-apache/kafka:${KAFKA_VERSION}}"
SPARK_DIST="spark-${SPARK_VERSION}-bin-${SPARK_HADOOP_PROFILE}"
SPARK_HOME="/opt/spark"

# ---- derived values -----------------------------------------------------------------------
# Number of Kafka brokers (containers on KAFKA_HOST): 1 for the small trial cluster, 3 for the full layout.
# contracts/topics.yaml asks for replication_cluster = 3; with fewer brokers the topic tools cap the replication
# factor at the broker count (contracts/ itself is never edited).
KAFKA_BROKER_COUNT="${KAFKA_BROKER_COUNT:-3}"
KAFKA_REPLICATION=$((KAFKA_BROKER_COUNT < 3 ? KAFKA_BROKER_COUNT : 3))   # default.replication.factor of the brokers
KAFKA_MIN_ISR=$((KAFKA_REPLICATION >= 3 ? 2 : 1))                        # min.insync.replicas (2 only when RF is 3)
KAFKA_PORT_BASE="${KAFKA_PORT_BASE:-9092}"
SPARK_WORKER_COUNT="${#SPARK_WORKER_HOSTS[@]}"
EXECUTORS_PER_WORKER=$((SPARK_WORKER_CORES / SPARK_EXECUTOR_CORES))
SPARK_CORES_MAX=$((SPARK_WORKER_COUNT * EXECUTORS_PER_WORKER * SPARK_EXECUTOR_CORES))
SPARK_MASTER_URL="spark://${SPARK_MASTER_HOST}:7077"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints"

# Broker n (1-based): clients on BASE+100*(n-1), controller on BASE+1+100*(n-1).
KAFKA_CLIENT_PORTS=()
KAFKA_CONTROLLER_PORTS=()
KAFKA_BOOTSTRAP=""
KAFKA_QUORUM_VOTERS=""
for ((_i = 0; _i < KAFKA_BROKER_COUNT; _i++)); do
    KAFKA_CLIENT_PORTS+=("$((KAFKA_PORT_BASE + 100 * _i))")
    KAFKA_CONTROLLER_PORTS+=("$((KAFKA_PORT_BASE + 1 + 100 * _i))")
    KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:+${KAFKA_BOOTSTRAP},}${KAFKA_HOST}:${KAFKA_CLIENT_PORTS[_i]}"
    KAFKA_QUORUM_VOTERS="${KAFKA_QUORUM_VOTERS:+${KAFKA_QUORUM_VOTERS},}$((_i + 1))@${KAFKA_HOST}:${KAFKA_CONTROLLER_PORTS[_i]}"
done
unset _i
export KAFKA_BOOTSTRAP

# ---- helpers ------------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(hostname -s)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(hostname -s)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(hostname -s)" "$*" >&2; exit 1; }

require_root() {
    [[ "${EUID}" -eq 0 ]] || die "run as root (sudo)"
}

all_hosts()   { printf '%s\n' "${KAFKA_HOST}" "${SPARK_MASTER_HOST}" "${SPARK_WORKER_HOSTS[@]}" "${NFS_SERVER_HOST}" | awk '!seen[$0]++'; }
spark_hosts() { printf '%s\n' "${SPARK_MASTER_HOST}" "${SPARK_WORKER_HOSTS[@]}" | awk '!seen[$0]++'; }

mem_to_mb() { # 3g -> 3072, 512m -> 512
    local v="$1"
    case "${v}" in
        *[gG]) echo $(( ${v%[gG]} * 1024 )) ;;
        *[mM]) echo $(( ${v%[mM]} )) ;;
        *) die "memory value '${v}' must end in g or m" ;;
    esac
}

# ---- memory budget -------------------------------------------------------------------------
# VM_MEMORY, OS_RESERVE_MB, DRIVER_OVERHEAD_MB and all Spark / Kafka sizes come from the VM profile
# (profiles/<VM_PROFILE>.env), optionally overridden in cluster.env.

# Sets VM_MB, WORKER_NEED_MB, KAFKA_NEED_MB, MASTER_NEED_MB (all in MB).
compute_budget() {
    local exec_mb overhead_mb daemon_mb driver_mb kafka_mb
    exec_mb="$(mem_to_mb "${SPARK_EXECUTOR_MEMORY}")"
    overhead_mb="$(mem_to_mb "${SPARK_EXECUTOR_MEMORY_OVERHEAD}")"
    daemon_mb="$(mem_to_mb "${SPARK_DAEMON_MEMORY}")"
    driver_mb="$(mem_to_mb "${SPARK_DRIVER_MEMORY}")"
    kafka_mb="$(mem_to_mb "${KAFKA_CONTAINER_MEMORY}")"
    VM_MB="$(mem_to_mb "${VM_MEMORY}")"
    WORKER_NEED_MB=$((EXECUTORS_PER_WORKER * (exec_mb + overhead_mb) + daemon_mb + OS_RESERVE_MB))
    KAFKA_NEED_MB=$((KAFKA_BROKER_COUNT * kafka_mb))
    MASTER_NEED_MB=$((daemon_mb + driver_mb + DRIVER_OVERHEAD_MB + OS_RESERVE_MB))
    if [[ "${KAFKA_HOST%%.*}" == "${SPARK_MASTER_HOST%%.*}" ]]; then
        MASTER_NEED_MB=$((MASTER_NEED_MB + KAFKA_NEED_MB))
    fi
}

show_budget() {
    compute_budget
    log "memory budget per VM (profile ${VM_PROFILE}, limit ${VM_MB} MB):"
    log "  worker : ${EXECUTORS_PER_WORKER} executors x ($(mem_to_mb "${SPARK_EXECUTOR_MEMORY}") + $(mem_to_mb "${SPARK_EXECUTOR_MEMORY_OVERHEAD}")) + daemon $(mem_to_mb "${SPARK_DAEMON_MEMORY}") + OS ${OS_RESERVE_MB} = ${WORKER_NEED_MB} MB"
    log "  master : ${KAFKA_BROKER_COUNT} broker(s) x $(mem_to_mb "${KAFKA_CONTAINER_MEMORY}") (if on the master) + daemon + driver + overhead + OS = ${MASTER_NEED_MB} MB"
}

validate_layout() {
    local worker_mem_mb
    [[ "${KAFKA_PORT_BASE}" =~ ^[0-9]+$ ]] || die "KAFKA_PORT_BASE must be a number"
    [[ "${KAFKA_BROKER_COUNT}" =~ ^[1-9][0-9]*$ ]] || die "KAFKA_BROKER_COUNT must be a positive number (1 or 3)"
    [[ $((KAFKA_BROKER_COUNT % 2)) -eq 1 ]] || warn "KAFKA_BROKER_COUNT=${KAFKA_BROKER_COUNT} is even: a KRaft quorum of N tolerates no more failures than N-1 (use 1 or 3)"
    [[ "${SPARK_WORKER_COUNT}" -ge 1 ]] || die "SPARK_WORKER_HOSTS is empty"
    [[ $((SPARK_WORKER_CORES % SPARK_EXECUTOR_CORES)) -eq 0 ]] || die "SPARK_WORKER_CORES must be a multiple of SPARK_EXECUTOR_CORES"
    worker_mem_mb="$(mem_to_mb "${SPARK_WORKER_MEMORY}")"
    [[ $((EXECUTORS_PER_WORKER * $(mem_to_mb "${SPARK_EXECUTOR_MEMORY}"))) -le "${worker_mem_mb}" ]] \
        || die "${EXECUTORS_PER_WORKER} executors x ${SPARK_EXECUTOR_MEMORY} does not fit in SPARK_WORKER_MEMORY=${SPARK_WORKER_MEMORY}"
    compute_budget
    [[ "${WORKER_NEED_MB}" -le "${VM_MB}" ]] || die "worker VM needs ${WORKER_NEED_MB} MB but VM_MEMORY is ${VM_MB} MB (lower SPARK_EXECUTOR_MEMORY / _OVERHEAD)"
    [[ "${MASTER_NEED_MB}" -le "${VM_MB}" ]] || die "master VM needs ${MASTER_NEED_MB} MB but VM_MEMORY is ${VM_MB} MB (lower KAFKA_CONTAINER_MEMORY / SPARK_DRIVER_MEMORY)"
    local w
    for w in "${SPARK_WORKER_HOSTS[@]}"; do
        [[ "${w%%.*}" != "${KAFKA_HOST%%.*}" ]] || die "KAFKA_HOST must not be a Spark worker (its memory budget has no room for the brokers)"
    done
}

# put_file <dest> <mode>   content on stdin. Returns 0 if the file changed, 1 if it was already identical.
put_file() {
    local dest="$1" mode="$2" tmp
    tmp="$(mktemp)"
    cat > "${tmp}"
    if [[ -f "${dest}" ]] && cmp -s "${tmp}" "${dest}"; then
        rm -f "${tmp}"
        return 1
    fi
    install -D -m "${mode}" "${tmp}" "${dest}"
    rm -f "${tmp}"
    return 0
}

fetch() { # fetch <url> <dest>
    curl -fSL --retry 3 --retry-delay 2 -o "$2" "$1"
}

# verify_sha512 <file> <checksum-url>   works for both the 'name: AAAA BBBB ...' and 'hash  name' layouts
verify_sha512() {
    local file="$1" url="$2" expected actual
    expected="$(curl -fsSL --retry 3 "${url}" | tr -d ' \n\r' | tr 'A-F' 'a-f')"
    actual="$(sha512sum "${file}" | cut -d' ' -f1)"
    [[ -n "${expected}" && "${expected}" == *"${actual}"* ]] || die "sha512 mismatch for ${file}"
}

resolve_check() { # fail early if a cluster hostname does not resolve on this VM
    local h bad=0
    while read -r h; do
        if ! getent hosts "${h}" > /dev/null; then
            warn "hostname '${h}' does not resolve (fix DNS or /etc/hosts)"
            bad=1
        fi
    done < <(all_hosts)
    [[ "${bad}" -eq 0 ]] || die "unresolvable cluster hostnames"
}

is_nfs_server() {
    [[ "$(hostname -s)" == "${NFS_SERVER_HOST%%.*}" ]]
}

# wait_port <host> <port> [seconds]   pure-bash TCP probe (works in Git Bash and on Ubuntu)
wait_port() {
    local host="$1" port="$2" secs="${3:-120}" i
    for ((i = 0; i < secs; i += 2)); do
        if timeout 2 bash -c "exec 3<>/dev/tcp/${host}/${port}" 2> /dev/null; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# Key=value file read by Python tools (cluster_common.py). No arrays, no quotes; lists are comma-separated.
render_runtime_env() {
    local spark_workers
    spark_workers="$(IFS=,; echo "${SPARK_WORKER_HOSTS[*]}")"
    cat <<EOF
# Generated by infra/cluster/deploy.sh (phase: env). Do not edit.
KAFKA_BOOTSTRAP=${KAFKA_BOOTSTRAP}
KAFKA_HOST=${KAFKA_HOST}
KAFKA_BROKER_COUNT=${KAFKA_BROKER_COUNT}
SPARK_MASTER_HOST=${SPARK_MASTER_HOST}
SPARK_MASTER_URL=${SPARK_MASTER_URL}
SPARK_WORKER_HOSTS=${spark_workers}
SPARK_WORKER_COUNT=${SPARK_WORKER_COUNT}
SPARK_WORKER_CORES=${SPARK_WORKER_CORES}
SPARK_EXECUTOR_CORES=${SPARK_EXECUTOR_CORES}
SPARK_EXECUTOR_MEMORY=${SPARK_EXECUTOR_MEMORY}
SPARK_CORES_MAX=${SPARK_CORES_MAX}
EXECUTORS_PER_WORKER=${EXECUTORS_PER_WORKER}
SHARED_DIR=${SHARED_DIR}
REPO_DIR=${REPO_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
VENV_DIR=${VENV_DIR}
SSH_USER=${SSH_USER}
EOF
}

# ---- ssh helpers (admin machine) -----------------------------------------------------------
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
# ssh reads stdin, which would swallow the rest of a 'while read h' host list. rsh / rsudo therefore use
# 'ssh -n'; rsh_in keeps stdin and is only for commands that receive a pipe (tar streams in deploy.sh).
rsh()   { local host="$1"; shift; ssh -n "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"; }
rsh_in() { local host="$1"; shift; ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"; }
rsudo() { local host="$1"; shift; ssh -n "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "sudo -n $*"; }

# Repo root on the admin machine (infra/cluster/../..) and a Python that has the repo requirements.
REPO_SRC="$(cd "${CLUSTER_DIR}/../.." && pwd)"
admin_python() {
    local c
    if [[ -n "${PYTHON_ADMIN:-}" ]]; then
        echo "${PYTHON_ADMIN}"
        return 0
    fi
    for c in "${REPO_SRC}/.venv/bin/python" "${REPO_SRC}/.venv/Scripts/python.exe"; do
        if [[ -x "${c}" ]]; then
            echo "${c}"
            return 0
        fi
    done
    command -v python3 || command -v python
}
