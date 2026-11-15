#!/usr/bin/env bash
# E3 fault tolerance: does the shared checkpoint give exactly-once Parquet output across an executor kill and a
# driver restart on another VM?
#
# Run ON THE SPARK MASTER VM as the 'spark' user, topics created and data/raw/source=smd present:
#     sudo -u spark bash experiments/e3_faults.sh -e 2 -c 1
#     sudo -u spark bash experiments/e3_faults.sh -n 1416825 -S 3000 --driver-b spark-w1 -- --tier1 v1
#
#   -e, --executors N / -c, --cores N   passed to submit.sh (default: submit.sh defaults)
#   -n, --records N          records per run (default 1416825 = the whole SMD set: train rows come first, so a short prefix has no test
#                            windows, no Parquet and nothing to compare); the SAME N records are replayed in both runs
#   -S, --speedup S          steady replay: event time runs S x faster than wall time (default 3000, ~1400 records/s, ~17 min per run; 0 = max speed,
#                            which is not steady and makes the kill window hard to hit)
#   -m, --max-offsets-per-trigger N   records per micro-batch (default 20000)
#   -t, --trigger T          score_job processing-time trigger (default '5 seconds', so batches follow each other quickly)
#       --driver-b HOST      VM for the restarted driver (default: first SPARK_WORKER_HOSTS entry that is not the master;
#                            with none, the master itself, and the report says so)
#       --warmup-batches K   committed batches before the executor kill (default 4)
#       --gap-batches K      committed batches after the executor recovered before the driver kill (default 4)
#       --scores-dir DIR     output to compare (default $REPO_DIR/data/scores_stream/source=smd, what score_job writes;
#                            the contract layer data/scores is the offline batch output, not the stream's)
#       --timeout S          give up a phase after S seconds (default 3600)
#       --run-id ID / -o, --out DIR   result directory (default $REPO_DIR/data/e3/<run-id>, run-id e3_<UTC timestamp>)
#       --skip-clean         reuse <out>/clean from an earlier run with the same --out (same N, speedup, partitions)
#   --                       everything after goes to score_job.py (e.g. --tier1 v1); both runs get the same options
#
# Steps
#   1. clean run: recreate telemetry.smd + anomalies, score_job --reset, steady replay of N records, wait until the
#      checkpoint has committed every record, snapshot data/scores_stream and the anomalies topic into <out>/clean.
#   2. fault run: same input, fresh topics and checkpoint. After --warmup-batches commits, wait for a batch in flight
#      (checkpoint offsets/N written, commits/N not yet) and kill -9 one executor JVM. Recovery time = kill -> commits/N.
#      After --gap-batches more commits, wait for a batch in flight again, kill -9 the driver (spark-submit, python) and
#      start a NEW driver on VM B with the SAME checkpoint dir and NO --reset. Recovery time = kill -> commits/N (the
#      batch the old driver had planned is re-run by the new one); also kill -> 'score_job started' (restart latency).
#   3. wait for the checkpoint to catch up with the topic, snapshot scores + anomalies into <out>/fault.
#   4. e3_compare.py: lost rows, duplicate Parquet rows, topic duplicates and losses (de-duplicated on
#      entity_id, window_start), value mismatches; writes <out>/report.md + report.json. Exit 0 = PASS.
#
# Caveat: both runs replay in real time, so micro-batch boundaries (and with them the event-time watermark when a batch
# is cut) differ between runs. A row that is late in one run only would show up as lost/extra: if the report fails
# check lost_sample first. The fault run's own duplicates are independent of that.
# The new driver on VM B needs: Spark, the venv and the NFS share (same paths), the shared repo copy of submit.sh from
# this branch (deploy.sh repo), and executors that resolve B's hostname.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../infra/cluster/lib.sh"

RECORDS=1416825
SPEEDUP=3000
MAX_OFFSETS=20000
TRIGGER_EVERY="5 seconds"
DRIVER_B=""
WARMUP=4
GAP=4
SCORES_DIR=""
TIMEOUT_S=3600
RUN_ID=""
OUT=""
SKIP_CLEAN=0
SUBMIT_ARGS=()
JOB_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--executors) SUBMIT_ARGS+=(--executors "${2:?--executors needs a value}"); shift 2 ;;
        -c|--cores)     SUBMIT_ARGS+=(--cores "${2:?--cores needs a value}"); shift 2 ;;
        -n|--records)   RECORDS="${2:?--records needs a value}"; shift 2 ;;
        -S|--speedup)   SPEEDUP="${2:?--speedup needs a value}"; shift 2 ;;
        -m|--max-offsets-per-trigger) MAX_OFFSETS="${2:?needs a value}"; shift 2 ;;
        -t|--trigger)   TRIGGER_EVERY="${2:?--trigger needs a value}"; shift 2 ;;
        --driver-b)     DRIVER_B="${2:?--driver-b needs a value}"; shift 2 ;;
        --warmup-batches) WARMUP="${2:?needs a value}"; shift 2 ;;
        --gap-batches)  GAP="${2:?needs a value}"; shift 2 ;;
        --scores-dir)   SCORES_DIR="${2:?--scores-dir needs a value}"; shift 2 ;;
        --timeout)      TIMEOUT_S="${2:?--timeout needs a value}"; shift 2 ;;
        --run-id)       RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
        -o|--out)       OUT="${2:?--out needs a value}"; shift 2 ;;
        --skip-clean)   SKIP_CLEAN=1; shift ;;
        -h|--help)      sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        --)             shift; JOB_ARGS=("$@"); break ;;
        *)              die "unknown option '$1' (pass score_job.py options after --)" ;;
    esac
done
for v in RECORDS MAX_OFFSETS WARMUP GAP TIMEOUT_S; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "${v} must be a positive integer, got '${!v}'"
done
[[ "${SPEEDUP}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--speedup must be a number"

PY="${VENV_DIR}/bin/python"
MASTER_SHORT="${SPARK_MASTER_HOST%%.*}"
[[ "$(hostname -s)" == "${MASTER_SHORT}" ]] || die "run this on ${SPARK_MASTER_HOST}"
[[ -x "${PY}" ]] || die "${PY} not found; run 30_spark_node.sh"
cd "${REPO_DIR}"
compgen -G "${REPO_DIR}/data/raw/source=smd/*.parquet" > /dev/null || die "no raw data in ${REPO_DIR}/data/raw/source=smd"

if [[ -z "${DRIVER_B}" ]]; then
    for w in "${SPARK_WORKER_HOSTS[@]}"; do
        [[ "${w%%.*}" != "${MASTER_SHORT}" ]] && { DRIVER_B="${w}"; break; }
    done
fi
if [[ -z "${DRIVER_B}" || "${DRIVER_B%%.*}" == "${MASTER_SHORT}" ]]; then
    DRIVER_B="${SPARK_MASTER_HOST}"
    SAME_VM=1
    warn "no second VM available: the driver is restarted on ${DRIVER_B} itself (a process restart, NOT a different-VM restart)"
else
    SAME_VM=0
fi

SMD_TOPIC="$("${PY}" -c "from contracts import schemas as S; print(S.TOPIC_SMD)")"
ANOM_TOPIC="$("${PY}" -c "from contracts import schemas as S; print(S.TOPIC_ANOMALIES)")"
PASS_N="$("${PY}" src/producer/replay.py --source smd --print-count)"
[[ "${RECORDS}" -le "${PASS_N}" ]] || die "--records ${RECORDS} exceeds one pass of the data set (${PASS_N})"

RUN_ID="${RUN_ID:-e3_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-${REPO_DIR}/data/e3/${RUN_ID}}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
case "${OUT}/" in "${SHARED_DIR}"/*) ;; *) die "--out must be under ${SHARED_DIR} (the driver on ${DRIVER_B} writes its log there)" ;; esac
SCORES_DIR="${SCORES_DIR:-${REPO_DIR}/data/scores_stream/source=smd}"
CK="${CHECKPOINT_DIR}/score/source=smd"
cp "${CLUSTER_ENV}" "${OUT}/cluster.env"          # lets the driver on another VM source the same settings
EXTRA_ENV=""
[[ "${VM_PROFILE:-}" == "" ]] || EXTRA_ENV="VM_PROFILE=${VM_PROFILE}"

log "E3 ${RUN_ID}: ${RECORDS} records at speedup ${SPEEDUP}, batches <= ${MAX_OFFSETS}, trigger ${TRIGGER_EVERY}; driver B = ${DRIVER_B}; results in ${OUT}"

now_ms() { date +%s%3N; }
fmt_s()  { awk -v ms="$1" 'BEGIN { printf "%.1f s", ms / 1000 }'; }
num_max() { # num_max <dir>: highest numeric file name in dir, or -1
    local m=-1 f n
    for f in "$1"/*; do
        n="${f##*/}"
        [[ "${n}" =~ ^[0-9]+$ ]] && (( n > m )) && m="${n}"
    done
    echo "${m}"
}
last_commit() { num_max "${CK}/commits"; }
last_offset() { num_max "${CK}/offsets"; }

REPLAY_PID=""; DRIVER_PID=""
stop_drivers() { # stop_drivers <signal>: signal every score_job driver process (spark-submit JVM + python) on A and B
    local h
    for h in "${SPARK_MASTER_HOST}" "${DRIVER_B}"; do
        rsh "${h}" "pkill -${1} -f '[s]core_entry.py'" 2> /dev/null || true
    done
}
cleanup() {
    [[ -z "${REPLAY_PID}" ]] || kill "${REPLAY_PID}" 2> /dev/null || true
    stop_drivers INT
    sleep 3
    stop_drivers KILL
}
trap cleanup EXIT INT TERM

wait_for() { # wait_for <description> <timeout s> <command...>: poll every 0.2 s
    local what="$1" limit="$2" t0; shift 2
    t0=$SECONDS
    until "$@"; do
        (( SECONDS - t0 < limit )) || die "timeout (${limit} s) waiting for ${what}"
        [[ -z "${DRIVER_PID}" ]] || kill -0 "${DRIVER_PID}" 2> /dev/null || [[ -n "${DRIVER_KILLED:-}" ]] \
            || die "score_job exited while waiting for ${what} (see ${OUT})"
        sleep 0.2
    done
}
commits_ge()   { (( $(last_commit) >= $1 )); }
scores_files_ge() { (( $(find "${SCORES_DIR}" -maxdepth 1 -name 'batch-*.parquet' 2> /dev/null | wc -l) >= $1 )); }
in_flight()    { (( $(last_offset) > $(last_commit) )); }
log_has()      { grep -q "$2" "$1" 2> /dev/null; }

# submit.sh takes --max-offsets-per-trigger as its own -m option; reuse it so score_job gets it exactly once
start_job_local() {
    local lg="$1"; shift
    bash "${CLUSTER_DIR}/submit.sh" "${SUBMIT_ARGS[@]}" -m "${MAX_OFFSETS}" -- --trigger "${TRIGGER_EVERY}" \
        --max-seconds "$((TIMEOUT_S * 4))" "$@" "${JOB_ARGS[@]}" > "${lg}" 2>&1 &
    DRIVER_PID=$!
    wait_for "'score_job started'" 240 log_has "${lg}" "score_job started"
}

start_job_on_b() { # start_job_on_b <log>: new driver on VM B, same checkpoint, no --reset
    local lg="$1" cmd
    cmd="cd $(printf %q "${REPO_DIR}") && ${EXTRA_ENV} CLUSTER_ENV=$(printf %q "${OUT}/cluster.env") SUBMIT_ANY_HOST=1 exec bash infra/cluster/submit.sh"
    cmd+=" $(printf '%q ' "${SUBMIT_ARGS[@]}") --skip-model-check -m ${MAX_OFFSETS} -- --trigger $(printf %q "${TRIGGER_EVERY}")"
    cmd+=" --max-seconds $((TIMEOUT_S * 4)) $(printf '%q ' "${JOB_ARGS[@]}")"
    rsh "${DRIVER_B}" "${cmd}" > "${lg}" 2>&1 &
    DRIVER_PID=$!
}

reset_topics() {
    "${PY}" "${CLUSTER_DIR}/create_topics_cluster.py" --bootstrap "${KAFKA_BOOTSTRAP}" \
        --recreate "${SMD_TOPIC}" --recreate "${ANOM_TOPIC}" > "${OUT}/topics.log" 2>&1 \
        || { cat "${OUT}/topics.log" >&2; die "could not recreate the topics"; }
}

start_replay() { # start_replay <log>
    "${PY}" src/producer/replay.py --source smd --speedup "${SPEEDUP}" --limit "${RECORDS}" \
        --bootstrap "${KAFKA_BOOTSTRAP}" > "$1" 2>&1 &
    REPLAY_PID=$!
}

drained() { # replay finished and every record is in a committed batch
    kill -0 "${REPLAY_PID}" 2> /dev/null && return 1
    [[ "$("${PY}" experiments/e3_compare.py lag --checkpoint "${CK}" --bootstrap "${KAFKA_BOOTSTRAP}" --topic "${SMD_TOPIC}")" == "0" ]]
}

snapshot() { # snapshot <dir>: scores output + anomalies topic after the job is stopped
    local d="$1"
    rm -rf "${d}"; mkdir -p "${d}"
    cp -r "${SCORES_DIR}" "${d}/scores"
    "${PY}" experiments/e3_compare.py dump-topic --bootstrap "${KAFKA_BOOTSTRAP}" --topic "${ANOM_TOPIC}" --out "${d}/anomalies.jsonl" \
        | xargs -I{} log "{} anomalies records in the topic" 2> /dev/null || true
    log "snapshot ${d}: $(ls "${d}/scores" | wc -l) score files, $(wc -l < "${d}/anomalies.jsonl") anomalies records"
}

finish_job() { # stop the driver(s) after the last commit
    stop_drivers INT
    local i
    for ((i = 0; i < 30; i++)); do
        pgrep -f '[s]core_entry.py' > /dev/null || break
        sleep 1
    done
    stop_drivers KILL
    DRIVER_PID=""
}

# ================= 1. clean run ===========================================================================
if [[ "${SKIP_CLEAN}" -eq 1 && -d "${OUT}/clean/scores" ]]; then
    log "1. clean run: reusing ${OUT}/clean"
else
    log "1. clean run"
    reset_topics
    start_job_local "${OUT}/clean_score_job.log" --reset
    start_replay "${OUT}/clean_replay.log"
    wait_for "the clean run to consume ${RECORDS} records" "${TIMEOUT_S}" drained
    REPLAY_PID=""
    sleep "$(( ${TRIGGER_EVERY%% *} + 5 ))"       # let the last batch's anomalies reach the topic
    finish_job
    snapshot "${OUT}/clean"
fi

# ================= 2. fault run ===========================================================================
log "2. fault run (executor kill, then driver restart on ${DRIVER_B})"
reset_topics
start_job_local "${OUT}/fault_score_job_A.log" --reset
start_replay "${OUT}/fault_replay.log"

wait_for "${WARMUP} committed batches" "${TIMEOUT_S}" commits_ge "$((WARMUP - 1))"
# SMD replays every machine's train rows first, which write no Parquet; fault only once test windows are being written
log "   warm-up done; waiting for test windows (Parquet files in ${SCORES_DIR}) ..."
wait_for "test windows to reach the scores output" "${TIMEOUT_S}" scores_files_ge 3

# --- executor kill, mid-batch
wait_for "a batch in flight" 120 in_flight
N1="$(last_offset)"
EXEC_HOST=""; EXEC_PID=""
for w in "${SPARK_WORKER_HOSTS[@]}"; do
    pid="$(rsh "${w}" "pgrep -f '[C]oarseGrainedExecutorBackend' | head -n 1" 2> /dev/null || true)"
    [[ -z "${pid}" ]] || { EXEC_HOST="${w}"; EXEC_PID="${pid}"; break; }
done
[[ -n "${EXEC_PID}" ]] || die "no executor JVM found on ${SPARK_WORKER_HOSTS[*]}"
T_EXEC="$(now_ms)"
rsh "${EXEC_HOST}" "kill -9 ${EXEC_PID}"
log "   killed executor pid ${EXEC_PID} on ${EXEC_HOST} while batch ${N1} was in flight (last commit $(last_commit))"
wait_for "batch ${N1} to commit after the executor kill" "${TIMEOUT_S}" commits_ge "${N1}"
EXEC_RECOVERY_MS=$(($(now_ms) - T_EXEC))
log "   executor recovery: $(fmt_s "${EXEC_RECOVERY_MS}") (kill -> commit of batch ${N1})"

# --- driver kill, mid-batch, restart on VM B
wait_for "${GAP} more commits" "${TIMEOUT_S}" commits_ge "$((N1 + GAP))"
wait_for "a batch in flight" 120 in_flight
N2="$(last_offset)"
T_DRV="$(now_ms)"
DRIVER_KILLED=1
stop_drivers KILL
log "   killed the driver on ${SPARK_MASTER_HOST} while batch ${N2} was in flight (last commit $(last_commit))"
start_job_on_b "${OUT}/fault_score_job_B.log"
wait_for "'score_job started' on ${DRIVER_B}" 300 log_has "${OUT}/fault_score_job_B.log" "score_job started"
DRV_START_MS=$(($(now_ms) - T_DRV))
DRIVER_KILLED=""
wait_for "batch ${N2} to commit on the new driver" "${TIMEOUT_S}" commits_ge "${N2}"
DRV_RECOVERY_MS=$(($(now_ms) - T_DRV))
log "   driver restart: started after $(fmt_s "${DRV_START_MS}"), batch ${N2} committed after $(fmt_s "${DRV_RECOVERY_MS}")"

# --- finish
wait_for "the fault run to consume ${RECORDS} records" "${TIMEOUT_S}" drained
REPLAY_PID=""
sleep "$(( ${TRIGGER_EVERY%% *} + 5 ))"
finish_job
snapshot "${OUT}/fault"

# ================= 3. report ==============================================================================
TIMINGS="${OUT}/timings.json"
"${PY}" - "${TIMINGS}" <<EOF
import json, sys
json.dump({
    "executor kill (batch ${N1} in flight), recovery kill -> batch committed": "$(fmt_s "${EXEC_RECOVERY_MS}")",
    "driver kill (batch ${N2} in flight) on ${SPARK_MASTER_HOST}, new driver on ${DRIVER_B}": "restart $(fmt_s "${DRV_START_MS}") to 'score_job started', $(fmt_s "${DRV_RECOVERY_MS}") to batch committed",
    "driver restart on a different VM": "$([[ "${SAME_VM}" -eq 0 ]] && echo yes || echo 'NO, same VM (no second VM available)')",
    "records per run / speedup": "${RECORDS} / ${SPEEDUP}",
}, open(sys.argv[1], "w"), indent=1)
EOF
rc=0
"${PY}" experiments/e3_compare.py report \
    --clean-scores "${OUT}/clean/scores" --fault-scores "${OUT}/fault/scores" \
    --clean-topic "${OUT}/clean/anomalies.jsonl" --fault-topic "${OUT}/fault/anomalies.jsonl" \
    --timings "${TIMINGS}" --out-json "${OUT}/report.json" --out-md "${OUT}/report.md" || rc=$?
log "report: ${OUT}/report.md"
exit "${rc}"
