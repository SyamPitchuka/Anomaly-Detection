#!/usr/bin/env bash
# Cluster health checks. Run from the admin machine (ssh to the VMs) - or on any VM that can ssh to the others.
#
#   bash infra/cluster/health_check.sh [kafka|spark|storage|memory|all|cassandra|hdfs]
#
# Prints OK / FAIL lines and ends with PASS or FAIL (exit 0 / 1), like the tools/check_*.py gauges.
# The deeper end-to-end checks are Python:  check_kafka_cluster.py  and  check_spark_cluster.py (see README).
# 'memory' is most meaningful while a job is running (the VMs have only 8 GB each).
# KAFKA_BROKER_COUNT (1 or 3) decides how many containers / ports / KRaft voters are expected.
# 'cassandra' (optional store, not part of 'all'): every CASSANDRA_HOSTS node runs the unit, accepts CQL, and
# 'nodetool status' on each node lists all of them as UN (Up / Normal).
# 'hdfs' (optional checkpoint store, E7, not part of 'all'): NameNode unit + RPC port, not in safe mode, every
# HDFS_DATANODE_HOSTS DataNode unit active and live at the NameNode, and a test file written by the spark user with
# replication HDFS_REPLICATION, read back, checked by fsck (replicated, nothing missing) and deleted.
set -uo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
validate_layout

MEM_MIN_AVAILABLE_MB=400
fails=0
ok()  { printf 'OK   %s\n' "$*"; }
bad() { printf 'FAIL %s\n' "$*"; fails=$((fails + 1)); }

# kcli <kafka cli script> <args...>   run a Kafka CLI tool inside the kafka-1 container on KAFKA_HOST
kcli() {
    local tool="$1"; shift
    rsudo "${KAFKA_HOST}" "docker exec -e LOG_DIR=/tmp/kcli kafka-1 /opt/kafka/bin/${tool} $*" 2> /dev/null
}

check_kafka() {
    local n id out leader voters state
    echo "--- Kafka (${KAFKA_BOOTSTRAP}; ${KAFKA_BROKER_COUNT} container(s) on ${KAFKA_HOST})"
    for ((n = 1; n <= KAFKA_BROKER_COUNT; n++)); do
        state="$(rsudo "${KAFKA_HOST}" "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' kafka-${n}" 2> /dev/null || true)"
        case "${state}" in
            "true false 0") ok "container kafka-${n} running, no OOM kill, no restarts" ;;
            "true true "*)  bad "container kafka-${n} was OOM-killed; raise KAFKA_CONTAINER_MEMORY / lower KAFKA_HEAP, or free memory on ${KAFKA_HOST}" ;;
            "true false "*) bad "container kafka-${n} is running but restarted ${state##* } time(s); see: sudo docker logs kafka-${n}" ;;
            *)              bad "container kafka-${n} is not running (state: ${state:-missing})" ;;
        esac
        if wait_port "${KAFKA_HOST}" "${KAFKA_CLIENT_PORTS[n - 1]}" 6; then
            ok "broker ${n} ${KAFKA_HOST}:${KAFKA_CLIENT_PORTS[n - 1]} accepts connections"
        else
            bad "broker ${n} ${KAFKA_HOST}:${KAFKA_CLIENT_PORTS[n - 1]} unreachable"
        fi
        if wait_port "${KAFKA_HOST}" "${KAFKA_CONTROLLER_PORTS[n - 1]}" 6; then
            ok "controller ${n} ${KAFKA_HOST}:${KAFKA_CONTROLLER_PORTS[n - 1]} accepts connections"
        else
            bad "controller ${n} ${KAFKA_HOST}:${KAFKA_CONTROLLER_PORTS[n - 1]} unreachable"
        fi
    done

    if out="$(kcli kafka-metadata-quorum.sh --bootstrap-server "${KAFKA_BOOTSTRAP}" describe --status)"; then
        leader="$(echo "${out}" | sed -n 's/^LeaderId:[[:space:]]*//p')"
        voters="$(echo "${out}" | sed -n 's/^CurrentVoters:[[:space:]]*//p' | tr -cd '0-9,' | tr ',' '\n' | grep -c . || true)"
        if [[ -n "${leader}" && "${leader}" != "-1" ]]; then ok "KRaft quorum has leader node ${leader}"; else bad "KRaft quorum has no leader"; fi
        if [[ "${voters}" -eq "${KAFKA_BROKER_COUNT}" ]]; then ok "KRaft voters: ${voters}"; else bad "KRaft voters: ${voters}, want ${KAFKA_BROKER_COUNT}"; fi
    else
        bad "kafka-metadata-quorum.sh failed inside kafka-1 on ${KAFKA_HOST}"
    fi

    # --under-replicated-partitions / --unavailable-partitions print nothing when everything is healthy
    for id in under-replicated unavailable; do
        if out="$(kcli kafka-topics.sh --bootstrap-server "${KAFKA_BOOTSTRAP}" --describe "--${id}-partitions")"; then
            if [[ -z "${out//[[:space:]]/}" ]]; then ok "no ${id} partitions"; else bad "${id} partitions:"; echo "${out}"; fi
        else
            bad "kafka-topics.sh --describe --${id}-partitions failed inside kafka-1"
        fi
    done

    local py; py="$(admin_python)"
    if [[ -n "${py}" ]] && "${py}" -c 'import confluent_kafka, yaml' 2> /dev/null; then
        if (cd "${REPO_SRC}" && "${py}" infra/cluster/create_topics_cluster.py --bootstrap "${KAFKA_BOOTSTRAP}" --brokers "${KAFKA_BROKER_COUNT}" | sed 's/^/     /'); then
            ok "topics match contracts/topics.yaml (partitions_cluster; replication_cluster capped at ${KAFKA_BROKER_COUNT} broker(s))"
        else
            bad "topics do not match contracts/topics.yaml (run: deploy.sh topics)"
        fi
    else
        warn "skipped topic check: no Python with confluent-kafka + pyyaml on this machine"
    fi
}

check_spark() {
    local json out py
    echo "--- Spark (${SPARK_MASTER_URL})"
    py="$(admin_python)"
    if ! json="$(curl -fsS --max-time 8 "http://${SPARK_MASTER_HOST}:8080/json/")"; then
        bad "master UI http://${SPARK_MASTER_HOST}:8080 unreachable"
        return
    fi
    if [[ -z "${py}" ]]; then
        warn "no Python on this machine to parse the master status"
        return
    fi
    if out="$(echo "${json}" | "${py}" -c '
import json, sys
want_workers, want_cores = int(sys.argv[1]), int(sys.argv[2])
d = json.load(sys.stdin)
alive = [w for w in d["workers"] if w["state"] == "ALIVE"]
print("master status=%s alive_workers=%d total_cores=%d" % (d["status"], len(alive), sum(w["cores"] for w in alive)))
for w in d["workers"]:
    print("  %-18s %-6s cores=%d memory=%dMB" % (w["host"], w["state"], w["cores"], w["memory"]))
bad = d["status"] != "ALIVE" or len(alive) != want_workers or any(w["cores"] != want_cores for w in alive)
sys.exit(1 if bad else 0)
' "${SPARK_WORKER_COUNT}" "${SPARK_WORKER_CORES}")"; then
        echo "${out}" | sed 's/^/     /'
        ok "master ALIVE with ${SPARK_WORKER_COUNT} ALIVE workers x ${SPARK_WORKER_CORES} cores"
    else
        echo "${out}" | sed 's/^/     /'
        bad "master/worker registration is not ${SPARK_WORKER_COUNT} ALIVE workers x ${SPARK_WORKER_CORES} cores"
    fi
}

check_storage() {
    local h f n tag="health-$$-$(date +%s)" total
    total="$(spark_hosts | wc -l)"
    echo "--- Shared storage (${SHARED_DIR}, NFS)"
    while read -r h; do
        f="$(rsh "${h}" "findmnt -n -o FSTYPE --target ${SHARED_DIR}" 2> /dev/null || true)"
        if [[ "${f}" == nfs* ]] || { [[ "${h%%.*}" == "${NFS_SERVER_HOST%%.*}" ]] && [[ -n "${f}" ]]; }; then
            ok "${h}: ${SHARED_DIR} is ${f}"
        else
            bad "${h}: ${SHARED_DIR} is '${f:-not mounted}'"
        fi
        if rsudo "${h}" "-u spark bash -c 'mkdir -p ${CHECKPOINT_DIR}/_health && echo ${h} > ${CHECKPOINT_DIR}/_health/${tag}-${h%%.*}'" 2> /dev/null; then
            ok "${h}: spark user can write ${CHECKPOINT_DIR}"
        else
            bad "${h}: spark user cannot write ${CHECKPOINT_DIR}"
        fi
    done < <(spark_hosts)

    sleep 2 # NFS attribute cache (actimeo=1)
    while read -r h; do
        # 'grep -c' prints 0 and exits 1 when nothing matches, so do not append '|| echo 0' (it would print "0 0")
        n="$(rsh "${h}" "ls ${CHECKPOINT_DIR}/_health 2> /dev/null | grep -c '^${tag}-'" 2> /dev/null || true)"
        n="${n:-0}"
        if [[ "${n}" -eq "${total}" ]]; then ok "${h}: sees all ${total} test files written by the nodes"; else bad "${h}: sees ${n}/${total} test files"; fi
    done < <(spark_hosts)
    rsudo "${NFS_SERVER_HOST}" "-u spark bash -c 'rm -f ${CHECKPOINT_DIR}/_health/${tag}-*; rmdir ${CHECKPOINT_DIR}/_health 2> /dev/null || true'" > /dev/null 2>&1 || true
}

check_memory() {
    local h total avail
    echo "--- Memory (${VM_MEMORY} per VM; fails below ${MEM_MIN_AVAILABLE_MB} MB available)"
    while read -r h; do
        total="$(rsh "${h}" "awk '/MemTotal/ {print int(\$2/1024)}' /proc/meminfo" 2> /dev/null || true)"
        avail="$(rsh "${h}" "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo" 2> /dev/null || true)"
        if [[ -z "${total}" || -z "${avail}" ]]; then
            bad "${h}: cannot read /proc/meminfo"
        elif [[ "${avail}" -ge "${MEM_MIN_AVAILABLE_MB}" ]]; then
            ok "${h}: ${avail} MB available of ${total} MB"
        else
            bad "${h}: only ${avail} MB available of ${total} MB (risk of the OOM killer)"
        fi
    done < <(all_hosts)
}

# count_un <nodetool status output>   number of nodes in state UN (Up, Normal)
count_un() { grep -cE '^UN[[:space:]]' <<< "$1" || true; }

check_cassandra() {
    local h hosts want state out n
    read -r -a hosts <<< "${CASSANDRA_HOSTS}"
    want="${#hosts[@]}"
    echo "--- Cassandra (${CASSANDRA_HOSTS}; cluster ${CASSANDRA_CLUSTER_NAME}; CQL ${CASSANDRA_CQL_PORT})"
    [[ "${want}" -ge 1 ]] || { bad "CASSANDRA_HOSTS is empty"; return; }
    for h in "${hosts[@]}"; do
        state="$(rsh "${h}" "systemctl is-active cassandra" 2> /dev/null || true)"
        if [[ "${state}" == "active" ]]; then ok "${h}: cassandra.service active"; else bad "${h}: cassandra.service is '${state:-unknown}'"; fi
        if wait_port "${h}" "${CASSANDRA_CQL_PORT}" 6; then
            ok "${h}:${CASSANDRA_CQL_PORT} accepts CQL connections"
        else
            bad "${h}:${CASSANDRA_CQL_PORT} unreachable"
        fi
        if out="$(rsh "${h}" "${CASSANDRA_HOME}/bin/nodetool status" 2> /dev/null)"; then
            n="$(count_un "${out}")"
            echo "${out}" | grep -E '^[UD][NLJM][[:space:]]' | sed 's/^/     /'
            if [[ "${n}" -eq "${want}" ]]; then ok "${h}: nodetool status: ${n} UN (want ${want})"; else bad "${h}: nodetool status: ${n} UN, want ${want}"; fi
        else
            bad "${h}: nodetool status failed"
        fi
    done
}

# live_datanodes <dfsadmin -report output>   the N of 'Live datanodes (N):'
live_datanodes() { sed -n 's/^Live datanodes (\([0-9]*\)):.*/\1/p' <<< "$1" | head -n 1; }
# fsck_count <fsck output> <label>   first number after '<label>:' (e.g. 'Under-replicated blocks'), '' if absent
fsck_count() { grep -m1 -E "^[[:space:]]*$2:" <<< "$1" | sed -E 's/^[^:]*:[[:space:]]*([0-9]+).*/\1/'; }

check_hdfs() {
    local h dn want out n state tag path got rep i under missing
    read -r -a dn <<< "${HDFS_DATANODE_HOSTS}"
    want="${#dn[@]}"
    echo "--- HDFS (${HDFS_URI}; DataNodes ${HDFS_DATANODE_HOSTS}; replication ${HDFS_REPLICATION})"
    state="$(rsh "${HDFS_NAMENODE_HOST}" "systemctl is-active hadoop-namenode" 2> /dev/null || true)"
    if [[ "${state}" == "active" ]]; then ok "${HDFS_NAMENODE_HOST}: hadoop-namenode.service active"; else bad "${HDFS_NAMENODE_HOST}: hadoop-namenode.service is '${state:-unknown}'"; fi
    if wait_port "${HDFS_NAMENODE_HOST}" "${HDFS_NAMENODE_PORT}" 6; then
        ok "NameNode RPC ${HDFS_NAMENODE_HOST}:${HDFS_NAMENODE_PORT} accepts connections"
    else
        bad "NameNode RPC ${HDFS_NAMENODE_HOST}:${HDFS_NAMENODE_PORT} unreachable"
        return
    fi
    for h in "${dn[@]}"; do
        state="$(rsh "${h}" "systemctl is-active hadoop-datanode" 2> /dev/null || true)"
        if [[ "${state}" == "active" ]]; then ok "${h}: hadoop-datanode.service active"; else bad "${h}: hadoop-datanode.service is '${state:-unknown}'"; fi
    done
    out="$(hdfs_on hdfs dfsadmin -safemode get 2> /dev/null || true)"
    if grep -q "Safe mode is OFF" <<< "${out}"; then ok "NameNode not in safe mode"; else bad "NameNode safe mode: '${out:-no answer}'"; fi
    out="$(hdfs_on hdfs dfsadmin -report -live 2> /dev/null || true)"
    n="$(live_datanodes "${out}")"
    if [[ "${n:-0}" -eq "${want}" ]]; then ok "live DataNodes: ${n} (want ${want})"; else bad "live DataNodes: ${n:-0}, want ${want}"; fi

    tag="health-$$-$(date +%s)"
    path="${HDFS_CHECKPOINT_ROOT}/_health/${tag}"
    if rsh "${HDFS_NAMENODE_HOST}" "echo ${tag} | sudo -n -u spark /usr/local/bin/hdfs dfs -D dfs.replication=${HDFS_REPLICATION} -put - ${path}" 2> /dev/null; then
        ok "spark user wrote ${HDFS_URI}${path}"
    else
        bad "spark user cannot write ${HDFS_URI}${path} (deploy.sh hdfs creates ${HDFS_CHECKPOINT_ROOT})"
        return
    fi
    got="$(hdfs_on spark dfs -cat "${path}" 2> /dev/null || true)"
    if [[ "${got}" == "${tag}" ]]; then ok "read back the test file"; else bad "read back '${got}', want '${tag}'"; fi
    rep="$(hdfs_on spark dfs -stat %r "${path}" 2> /dev/null || true)"
    if [[ "${rep}" == "${HDFS_REPLICATION}" ]]; then ok "test file replication factor ${rep}"; else bad "test file replication '${rep}', want ${HDFS_REPLICATION}"; fi
    for ((i = 0; i < 10; i++)); do     # block reports reach the NameNode a moment after the write
        out="$(hdfs_on hdfs fsck "${path}" 2> /dev/null || true)"
        under="$(fsck_count "${out}" "Under-replicated blocks")"
        missing="$(fsck_count "${out}" "Missing blocks")"
        [[ "${under}" == "0" && "${missing}" == "0" ]] && break
        sleep 1
    done
    if [[ "${under}" == "0" && "${missing}" == "0" ]] && grep -q "Status: HEALTHY" <<< "${out}"; then
        ok "fsck: HEALTHY, 0 under-replicated, 0 missing (${HDFS_REPLICATION} replicas)"
    else
        bad "fsck of the test file: under-replicated '${under}', missing '${missing}'"
        echo "${out}" | grep -E "Status|replicated|Missing|Corrupt" | sed 's/^/     /'
    fi
    if hdfs_on spark dfs -rm -skipTrash "${path}" > /dev/null 2>&1 && ! hdfs_on spark dfs -test -e "${path}" 2> /dev/null; then
        ok "deleted the test file"
    else
        bad "could not delete ${path}"
    fi
}

target="${1:-all}"
case "${target}" in
    kafka) check_kafka ;;
    spark) check_spark ;;
    storage) check_storage ;;
    memory) check_memory ;;
    all) check_kafka; check_spark; check_storage; check_memory ;;
    cassandra) check_cassandra ;;
    hdfs) check_hdfs ;;
    *) die "usage: health_check.sh [kafka|spark|storage|memory|all|cassandra|hdfs]" ;;
esac

if [[ "${fails}" -eq 0 ]]; then
    echo "PASS"
    exit 0
fi
echo "FAIL (${fails} problem(s))"
exit 1
