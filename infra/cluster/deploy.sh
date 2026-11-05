#!/usr/bin/env bash
# Orchestrates the whole cluster from an admin machine (Git Bash, WSL or Linux) over ssh.
# Needs: key-based ssh to every VM as SSH_USER, passwordless sudo on the VMs, tar, and a Python with
# the repo requirements (for the topic step). Every phase is idempotent and can be re-run.
#
#   bash infra/cluster/deploy.sh all          # everything below, in order
#   bash infra/cluster/deploy.sh <phase>...   # one or more phases
#
# Phases: ssh  env  prereqs  nfs  repo  kafka  spark  topics  health
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# push_scripts <host>   copy infra/cluster (including cluster.env) to /tmp/cluster-setup on <host>
push_scripts() {
    local host="$1"
    tar -C "${CLUSTER_DIR}" --exclude=__pycache__ -cf - . \
        | rsh_in "${host}" "rm -rf /tmp/cluster-setup && mkdir -p /tmp/cluster-setup && tar -C /tmp/cluster-setup -xf -"
}

# run_script <host> <script>
run_script() {
    local host="$1" script="$2"
    push_scripts "${host}"
    log "---- ${host}: ${script}"
    rsudo "${host}" "bash /tmp/cluster-setup/${script}"
}

phase_ssh() {
    local h
    while read -r h; do
        if rsh "${h}" "sudo -n true" > /dev/null 2>&1; then
            log "ssh + sudo ok: ${h}"
        else
            die "cannot ssh to ${SSH_USER}@${h} with passwordless sudo"
        fi
    done < <(all_hosts)
}

phase_env() {
    render_runtime_env > "${CLUSTER_DIR}/cluster.runtime.env"
    log "wrote ${CLUSTER_DIR}/cluster.runtime.env"
    show_budget
}

phase_prereqs() {
    local h
    while read -r h; do
        run_script "${h}" 00_prereqs.sh
    done < <(all_hosts)
}

phase_nfs() {
    local h
    run_script "${NFS_SERVER_HOST}" 10_nfs_server.sh
    while read -r h; do
        run_script "${h}" 11_nfs_client.sh
    done < <(spark_hosts)
}

# Copy the repo checkout to the share (via the NFS server). checkpoints/ is never touched.
# The tar is extracted as 'spark' so every node sees spark-owned files. Nothing is deleted on the target.
phase_repo() {
    phase_env
    log "copying ${REPO_SRC} -> ${NFS_SERVER_HOST}:${REPO_DIR}"
    tar -C "${REPO_SRC}" -cf - \
        --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache \
        --exclude=checkpoints --exclude=cluster.env . \
        | rsh_in "${NFS_SERVER_HOST}" "sudo -n -u spark tar -C ${REPO_DIR} -xf -"
    rsudo "${NFS_SERVER_HOST}" "install -d -o spark -g spark -m 0775 ${CHECKPOINT_DIR}"
}

# All brokers are containers on KAFKA_HOST, so the script runs once.
phase_kafka() {
    local n
    if [[ -z "${KAFKA_CLUSTER_ID:-}" ]]; then
        bash "${CLUSTER_DIR}/gen_cluster_id.sh"
        # shellcheck disable=SC1090
        source "${CLUSTER_ENV}"
    fi
    run_script "${KAFKA_HOST}" 20_kafka_broker.sh
    for ((n = 1; n <= KAFKA_BROKER_COUNT; n++)); do
        log "waiting for broker ${n} on ${KAFKA_HOST}:${KAFKA_CLIENT_PORTS[n - 1]}"
        wait_port "${KAFKA_HOST}" "${KAFKA_CLIENT_PORTS[n - 1]}" 240 \
            || die "broker ${n} did not open :${KAFKA_CLIENT_PORTS[n - 1]}; see: ssh ${KAFKA_HOST} sudo docker logs kafka-${n}"
    done
    log "${KAFKA_BROKER_COUNT} broker(s) listening"
}

phase_spark() {
    local h
    while read -r h; do
        run_script "${h}" 30_spark_node.sh
    done < <(spark_hosts)
    wait_port "${SPARK_MASTER_HOST}" 8080 120 || die "master web UI did not open"
}

phase_topics() {
    local py; py="$(admin_python)"
    [[ -n "${py}" ]] || die "no Python found on the admin machine; set PYTHON_ADMIN"
    (cd "${REPO_SRC}" && "${py}" infra/cluster/create_topics_cluster.py --bootstrap "${KAFKA_BOOTSTRAP}" --brokers "${KAFKA_BROKER_COUNT}")
}

phase_health() {
    bash "${CLUSTER_DIR}/health_check.sh" all
}

run_phase() {
    case "$1" in
        ssh) phase_ssh ;;
        env) phase_env ;;
        prereqs) phase_prereqs ;;
        nfs) phase_nfs ;;
        repo) phase_repo ;;
        kafka) phase_kafka ;;
        spark) phase_spark ;;
        topics) phase_topics ;;
        health) phase_health ;;
        *) die "unknown phase '$1' (ssh env prereqs nfs repo kafka spark topics health all)" ;;
    esac
}

validate_layout
phases=("$@")
[[ "${#phases[@]}" -gt 0 ]] || die "usage: deploy.sh all | <phase>...   (phases: ssh env prereqs nfs repo kafka spark topics health)"
if [[ "${phases[0]}" == "all" ]]; then
    phases=(ssh env prereqs nfs repo kafka spark topics health)
fi
for p in "${phases[@]}"; do
    log "=== phase: ${p}"
    run_phase "${p}"
done
log "done: ${phases[*]}"
