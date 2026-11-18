#!/usr/bin/env bash
# E7: HDFS as the Spark checkpoint store. Does the streaming job survive the loss of a storage node, and what does
# HDFS cost in throughput compared with the NFS checkpoints?
#
# Run ON THE SPARK MASTER VM as the 'spark' user, after 'deploy.sh hdfs' + 'deploy.sh repo' (the share needs this
# branch's score_job.py / score_entry.py), topics created and data/raw/source=smd present:
#     sudo -u spark bash experiments/e7_hdfs.sh
#     sudo -u spark bash experiments/e7_hdfs.sh --steps c --nfs-repeats 3
#
#   -e, --executors N / -c, --cores N   score_job size (default 5 x 2, as the E2 5-worker baseline)
#   -n, --records N          records for (a) and (b) (default 1416825 = the whole SMD set)
#   -m, --max-offsets-per-trigger N   records per micro-batch (default 200000)
#       --stop-host H        DataNode stopped in (b) (default spark-w5; must be in HDFS_DATANODE_HOSTS)
#       --stop-after K       stop it once K micro-batches with input rows have been logged (default 3)
#       --after-dead S       keep it down S more seconds once the NameNode declared it dead (default 60)
#       --dead-timeout S     give up waiting for the NameNode to declare it dead after S s (default 600)
#       --heal-timeout S     give up waiting for fsck HEALTHY after the restart after S s (default 900)
#       --tput-records N / --tput-replicate R / --repeats K   (c): K drains of N records, replay.py --replicate R
#                            (default 2000000 / 4 / 3: the backlog of docs/results/e2_cluster_5w.json)
#       --nfs-repeats K      (c) also K drains with the default NFS checkpoints, interleaved (default 0: compare
#                            with the E2 baseline numbers only)
#       --steps LIST         any of a, b, c (default abc); b needs a's snapshot (same --out)
#       --run-id ID / -o, --out DIR   results (default $REPO_DIR/data/e7/<run-id>, run-id e7_<UTC timestamp>)
#       --keep-hdfs          keep the HDFS checkpoints of the runs (default: delete them at the end)
#
# Steps (checkpoint root of every HDFS run: $HDFS_URI$HDFS_CHECKPOINT_ROOT/e7/<run_id>)
#   (a) clean: recreate 'anomalies', drain_test.sh -e E -c C -n N -k hdfs://.../<run_id>; snapshot
#       data/scores_stream/source=smd and the anomalies topic into <out>/clean; fsck the checkpoint.
#   (b) fault: the same drain in the background; once K batches are logged, 'systemctl stop hadoop-datanode' on
#       the stop host, kept down until the drain has finished AND the NameNode has declared the node dead (+ S s).
#       'dfsadmin -report -live' + 'fsck <checkpoint>' are sampled every ~5 s into <out>/fault/samples. Then the
#       DataNode is started again and sampled until fsck is HEALTHY with 0 under-replicated blocks. Snapshot into
#       <out>/fault; e7_report.py compare (e3_compare.py definitions) -> <out>/compare.json.
#   (c) throughput: K drains of the standard backlog with HDFS checkpoints (+ optional NFS re-runs).
#   e7_report.py report -> docs/results/e7_hdfs.md / .json of this checkout (and a copy in <out>). Exit 0 = PASS.
# If the script stops early, the EXIT trap starts the stopped DataNode again.
set -euo pipefail
E7_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../infra/cluster/lib.sh
source "${E7_DIR}/../infra/cluster/lib.sh"

EXECUTORS=5
CORES=2
RECORDS=1416825
MAX_OFFSETS=200000
STOP_HOST=spark-w5
STOP_AFTER=3
AFTER_DEAD_S=60
DEAD_TIMEOUT_S=600
HEAL_TIMEOUT_S=900
TPUT_RECORDS=2000000
TPUT_REPLICATE=4
REPEATS=3
NFS_REPEATS=0
STEPS=abc
RUN_ID=""
OUT=""
KEEP_HDFS=0
SAMPLE_EVERY_S=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--executors) EXECUTORS="${2:?needs a value}"; shift 2 ;;
        -c|--cores)     CORES="${2:?needs a value}"; shift 2 ;;
        -n|--records)   RECORDS="${2:?needs a value}"; shift 2 ;;
        -m|--max-offsets-per-trigger) MAX_OFFSETS="${2:?needs a value}"; shift 2 ;;
        --stop-host)    STOP_HOST="${2:?needs a value}"; shift 2 ;;
        --stop-after)   STOP_AFTER="${2:?needs a value}"; shift 2 ;;
        --after-dead)   AFTER_DEAD_S="${2:?needs a value}"; shift 2 ;;
        --dead-timeout) DEAD_TIMEOUT_S="${2:?needs a value}"; shift 2 ;;
        --heal-timeout) HEAL_TIMEOUT_S="${2:?needs a value}"; shift 2 ;;
        --tput-records) TPUT_RECORDS="${2:?needs a value}"; shift 2 ;;
        --tput-replicate) TPUT_REPLICATE="${2:?needs a value}"; shift 2 ;;
        --repeats)      REPEATS="${2:?needs a value}"; shift 2 ;;
        --nfs-repeats)  NFS_REPEATS="${2:?needs a value}"; shift 2 ;;
        --steps)        STEPS="${2:?needs a value}"; shift 2 ;;
        --run-id)       RUN_ID="${2:?needs a value}"; shift 2 ;;
        -o|--out)       OUT="${2:?needs a value}"; shift 2 ;;
        --keep-hdfs)    KEEP_HDFS=1; shift ;;
        -h|--help)      sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *)              die "unknown option '$1'" ;;
    esac
done
for v in EXECUTORS CORES RECORDS MAX_OFFSETS STOP_AFTER DEAD_TIMEOUT_S HEAL_TIMEOUT_S TPUT_RECORDS TPUT_REPLICATE REPEATS; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "${v} must be a positive integer, got '${!v}'"
done
for v in AFTER_DEAD_S NFS_REPEATS; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || die "${v} must be a non-negative integer, got '${!v}'"
done
[[ "${STEPS}" =~ ^[abc]+$ ]] || die "--steps takes letters a, b, c (e.g. abc, bc), got '${STEPS}'"
read -r -a DN_HOSTS <<< "${HDFS_DATANODE_HOSTS}"
EXPECTED_DN="${#DN_HOSTS[@]}"
in_dn=0
for h in "${DN_HOSTS[@]}"; do [[ "${h%%.*}" == "${STOP_HOST%%.*}" ]] && in_dn=1; done
[[ "${in_dn}" -eq 1 ]] || die "--stop-host ${STOP_HOST} is not in HDFS_DATANODE_HOSTS (${HDFS_DATANODE_HOSTS})"
[[ "${EXPECTED_DN}" -gt "${HDFS_REPLICATION}" ]] \
    || die "with ${EXPECTED_DN} DataNode(s) and replication ${HDFS_REPLICATION}, stopping one leaves too few for the experiment"

PY="${VENV_DIR}/bin/python"
[[ "$(hostname -s)" == "${SPARK_MASTER_HOST%%.*}" ]] || die "run this on ${SPARK_MASTER_HOST}"
[[ -x "${PY}" ]] || die "${PY} not found; run 30_spark_node.sh"
cd "${REPO_DIR}"
compgen -G "${REPO_DIR}/data/raw/source=smd/*.parquet" > /dev/null || die "no raw data in ${REPO_DIR}/data/raw/source=smd"
grep -q "def delete_hadoop_path" "${REPO_DIR}/src/streaming/score_job.py" \
    || die "the share's score_job.py predates E7 (no HDFS --reset): run deploy.sh repo from this checkout"

ANOM_TOPIC="$("${PY}" -c "from contracts import schemas as S; print(S.TOPIC_ANOMALIES)")"
RUN_ID="${RUN_ID:-e7_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-${REPO_DIR}/data/e7/${RUN_ID}}"
mkdir -p "${OUT}/runs"
OUT="$(cd "${OUT}" && pwd)"
RUNS="${OUT}/runs"
EVENTS="${OUT}/events.jsonl"
SCORES_DIR="${REPO_DIR}/data/scores_stream/source=smd"
PROGRESS="${REPO_DIR}/data/latency_stream/progress.jsonl"
HDFS_E7_ROOT="${HDFS_URI}${HDFS_CHECKPOINT_ROOT}/e7"
REPORT=("${PY}" "${E7_DIR}/e7_report.py")
DRAIN=(bash "${CLUSTER_DIR}/drain_test.sh" -e "${EXECUTORS}" -c "${CORES}" -m "${MAX_OFFSETS}" -o "${RUNS}")

log "E7 ${RUN_ID}: steps ${STEPS}, ${EXECUTORS} x ${CORES}, checkpoints under ${HDFS_E7_ROOT}; results in ${OUT}"

event() { "${REPORT[@]}" event --file "${EVENTS}" --name "$@"; }
ckpt_path() { echo "$(hdfs_uri_path "${HDFS_E7_ROOT}")/$1"; }      # HDFS path (no scheme) of a run's checkpoint root

DN_STOPPED=0
DRAIN_PID=""
cleanup() {
    if [[ -n "${DRAIN_PID}" ]] && kill -0 "${DRAIN_PID}" 2> /dev/null; then
        kill -TERM "${DRAIN_PID}" 2> /dev/null || true        # drain_test.sh's own trap stops score_job
        wait "${DRAIN_PID}" 2> /dev/null || true
    fi
    if [[ "${DN_STOPPED}" -eq 1 ]]; then
        warn "starting the DataNode on ${STOP_HOST} again"
        rsudo "${STOP_HOST}" "systemctl start hadoop-datanode" || warn "could not start hadoop-datanode on ${STOP_HOST}"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# sample <phase>   dfsadmin -report -live + fsck of the fault run's checkpoint into <out>/fault/samples/<ms>_<phase>.txt
sample() {
    local ms; ms="$(date +%s%3N)"
    {
        echo "# e7-sample ts_ms=${ms} phase=$1"
        hdfs_on hdfs dfsadmin -report -live 2> /dev/null || true
        echo "==== fsck"
        hdfs_on hdfs fsck "${FAULT_CK}" 2> /dev/null || true      # exit 1 when unhealthy: the text is what counts
    } > "${OUT}/fault/samples/${ms}_$1.txt"
    LAST_SAMPLE="${OUT}/fault/samples/${ms}_$1.txt"
}
field() { "${REPORT[@]}" sample "${LAST_SAMPLE}" --field "$1"; }

recreate_anomalies() {
    "${PY}" "${CLUSTER_DIR}/create_topics_cluster.py" --bootstrap "${KAFKA_BOOTSTRAP}" --recreate "${ANOM_TOPIC}" \
        > "${OUT}/topics.log" 2>&1 || { cat "${OUT}/topics.log" >&2; die "could not recreate ${ANOM_TOPIC}"; }
}

snapshot() { # snapshot <dir>
    local d="$1"
    rm -rf "${d}/scores" "${d}/anomalies.jsonl"; mkdir -p "${d}"
    cp -r "${SCORES_DIR}" "${d}/scores"
    "${PY}" "${E7_DIR}/e3_compare.py" dump-topic --bootstrap "${KAFKA_BOOTSTRAP}" --topic "${ANOM_TOPIC}" \
        --out "${d}/anomalies.jsonl" > /dev/null
    log "snapshot ${d}: $(ls "${d}/scores" | wc -l) score files, $(wc -l < "${d}/anomalies.jsonl") anomalies records"
}

drain_hdfs() { # drain_hdfs <run_id> [drain_test options]: one drain with HDFS checkpoints
    local rid="$1"; shift
    "${DRAIN[@]}" --run-id "${rid}" -k "${HDFS_E7_ROOT}/{run_id}" --tag e7 "$@"
}

log "--- HDFS health"
if ! bash "${CLUSTER_DIR}/health_check.sh" hdfs | sed 's/^/    /'; then
    die "health_check.sh hdfs failed: fix HDFS first"
fi

# ================= (a) clean drain ==========================================================================
if [[ "${STEPS}" == *a* ]]; then
    log "(a) clean drain with HDFS checkpoints: ${RECORDS} records"
    recreate_anomalies
    drain_hdfs "${RUN_ID}_clean" -n "${RECORDS}" > "${OUT}/clean_drain.log" 2>&1 \
        || { tail -n 30 "${OUT}/clean_drain.log" >&2; die "(a) drain failed (see ${OUT}/clean_drain.log)"; }
    snapshot "${OUT}/clean"
    hdfs_on hdfs fsck "$(ckpt_path "${RUN_ID}_clean")" > "${OUT}/clean/fsck.txt" 2>&1 || true
    log "    fsck: $(grep -m1 '^Status' "${OUT}/clean/fsck.txt" || echo '?')"
fi

# ================= (b) DataNode stopped during the drain ====================================================
if [[ "${STEPS}" == *b* ]]; then
    [[ -d "${OUT}/clean/scores" ]] || die "(b) needs (a)'s snapshot in ${OUT}/clean (run --steps ab, or the same --out)"
    log "(b) the same drain; DataNode on ${STOP_HOST} stopped after ${STOP_AFTER} micro-batches"
    rm -rf "${OUT}/fault"; mkdir -p "${OUT}/fault/samples"
    : > "${EVENTS}"
    FAULT_CK="$(ckpt_path "${RUN_ID}_fault")"
    recreate_anomalies
    rm -f "${PROGRESS}"
    drain_hdfs "${RUN_ID}_fault" -n "${RECORDS}" > "${OUT}/fault_drain.log" 2>&1 &
    DRAIN_PID=$!
    event drain_started "run_id=${RUN_ID}_fault"
    t0=$SECONDS
    until (( $("${REPORT[@]}" batches --progress "${PROGRESS}") >= STOP_AFTER - 1 )); do
        kill -0 "${DRAIN_PID}" 2> /dev/null || { tail -n 30 "${OUT}/fault_drain.log" >&2; die "(b) drain ended before batch ${STOP_AFTER}"; }
        (( SECONDS - t0 < 1800 )) || die "(b) no ${STOP_AFTER} batches after 1800 s"
        sleep 0.5
    done
    sample before
    until (( $("${REPORT[@]}" batches --progress "${PROGRESS}") >= STOP_AFTER )); do
        kill -0 "${DRAIN_PID}" 2> /dev/null || die "(b) drain ended before batch ${STOP_AFTER}"
        sleep 0.2
    done
    rsudo "${STOP_HOST}" "systemctl stop hadoop-datanode" || die "could not stop hadoop-datanode on ${STOP_HOST}"
    DN_STOPPED=1
    event datanode_stopped "host=${STOP_HOST}" "after_batches=${STOP_AFTER}"
    log "    DataNode on ${STOP_HOST} stopped after batch ${STOP_AFTER}; sampling fsck every ~${SAMPLE_EVERY_S} s"

    drain_rc=""
    dead_at=""
    t0=$SECONDS
    while :; do
        sample outage
        if [[ -z "${drain_rc}" ]] && ! kill -0 "${DRAIN_PID}" 2> /dev/null; then
            drain_rc=0; wait "${DRAIN_PID}" || drain_rc=$?
            DRAIN_PID=""
            event drain_finished "rc=${drain_rc}"
            log "    drain finished (rc ${drain_rc})"
        fi
        live="$(field live)"
        if [[ -z "${dead_at}" && -n "${live}" && "${live}" -lt "${EXPECTED_DN}" ]]; then
            dead_at=$SECONDS
            event datanode_dead "live=${live}" "under=$(field under)"
            log "    NameNode lists ${live} live DataNodes: ${STOP_HOST} declared dead ($((SECONDS - t0)) s); under-replicated $(field under), missing $(field missing)"
        fi
        if [[ -n "${drain_rc}" && -n "${dead_at}" ]] && (( SECONDS - dead_at >= AFTER_DEAD_S )); then
            break
        fi
        if [[ -z "${dead_at}" ]] && (( SECONDS - t0 > DEAD_TIMEOUT_S )) && [[ -n "${drain_rc}" ]]; then
            warn "the NameNode did not declare ${STOP_HOST} dead within ${DEAD_TIMEOUT_S} s"
            break
        fi
        sleep "${SAMPLE_EVERY_S}"
    done
    [[ "${drain_rc}" == "0" ]] || { tail -n 30 "${OUT}/fault_drain.log" >&2; warn "(b) drain did not complete (rc ${drain_rc})"; }

    rsudo "${STOP_HOST}" "systemctl start hadoop-datanode" || die "could not start hadoop-datanode on ${STOP_HOST}"
    DN_STOPPED=0
    event datanode_started "host=${STOP_HOST}"
    log "    DataNode on ${STOP_HOST} started again; waiting for fsck HEALTHY with 0 under-replicated blocks"
    t0=$SECONDS
    while :; do
        sleep "${SAMPLE_EVERY_S}"
        sample restarted
        if [[ "$(field live)" == "${EXPECTED_DN}" && "$(field under)" == "0" && "$(field missing)" == "0" \
              && "$(field status)" == "HEALTHY" ]]; then
            event healed
            log "    healed after $((SECONDS - t0)) s: ${EXPECTED_DN} live DataNodes, fsck HEALTHY, 0 under-replicated"
            break
        fi
        (( SECONDS - t0 < HEAL_TIMEOUT_S )) || { warn "not healed within ${HEAL_TIMEOUT_S} s"; break; }
    done

    if [[ "${drain_rc}" == "0" ]]; then
        snapshot "${OUT}/fault"
        "${REPORT[@]}" compare --clean "${OUT}/clean" --fault "${OUT}/fault" --out "${OUT}/compare.json" \
            || warn "(b) output differs from (a): see ${OUT}/compare.json"
    fi
fi

# ================= (c) throughput ===========================================================================
if [[ "${STEPS}" == *c* ]]; then
    log "(c) throughput: ${REPEATS} x ${TPUT_RECORDS} records (replicate ${TPUT_REPLICATE}) with HDFS checkpoints, ${NFS_REPEATS} with NFS"
    for ((k = 1; k <= REPEATS || k <= NFS_REPEATS; k++)); do
        if (( k <= REPEATS )); then
            rid="${RUN_ID}_hdfs_rep${k}"
            drain_hdfs "${rid}" -n "${TPUT_RECORDS}" -r "${TPUT_REPLICATE}" > "${OUT}/${rid}.log" 2>&1 \
                || warn "${rid} did not drain (see ${OUT}/${rid}.log)"
            [[ "${KEEP_HDFS}" -eq 1 ]] || hdfs_on spark dfs -rm -r -skipTrash "$(ckpt_path "${rid}")" > /dev/null 2>&1 || true
        fi
        if (( k <= NFS_REPEATS )); then
            rid="${RUN_ID}_nfs_rep${k}"
            "${DRAIN[@]}" --run-id "${rid}" -n "${TPUT_RECORDS}" -r "${TPUT_REPLICATE}" --tag e7_nfs \
                > "${OUT}/${rid}.log" 2>&1 || warn "${rid} did not drain (see ${OUT}/${rid}.log)"
        fi
    done
fi

if [[ "${KEEP_HDFS}" -eq 0 ]]; then
    for rid in "${RUN_ID}_clean" "${RUN_ID}_fault"; do
        hdfs_on spark dfs -rm -r -skipTrash "$(ckpt_path "${rid}")" > /dev/null 2>&1 || true
    done
fi

# ================= report ===================================================================================
rc=0
"${REPORT[@]}" report --out-dir "${OUT}" --expected-datanodes "${EXPECTED_DN}" || rc=$?
cp "${E7_DIR}/../docs/results/e7_hdfs.md" "${E7_DIR}/../docs/results/e7_hdfs.json" "${OUT}/" 2> /dev/null || true
log "report: ${E7_DIR}/../docs/results/e7_hdfs.md (copy in ${OUT})"
exit "${rc}"
