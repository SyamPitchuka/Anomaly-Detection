#!/usr/bin/env bash
# E2 strong scaling, "backlog drain": put N records in telemetry.smd with score_job NOT running, then start
# score_job on E executors x C cores and time how long it takes to consume the whole backlog.
#
# Run ON THE SPARK MASTER VM as the 'spark' user, with the topics created and data/raw/source=smd present:
#     sudo -u spark bash infra/cluster/drain_test.sh -e 2 -c 2
#     sudo -u spark bash infra/cluster/drain_test.sh -e 4 -c 2 -n 5000000 -m 200000 --run-id e4c2_rep1
#     sudo -u spark bash infra/cluster/drain_test.sh -e 4 -c 2 -r 4 -- --tier1 v1     # after -- : score_job options
#
#   -e, --executors N      executors for score_job (default: all the cluster can hold)   } passed to submit.sh
#   -c, --cores N          cores per executor (default: profile value)                   }
#   -p, --shuffle-partitions N  spark.sql.shuffle.partitions (default 2 x executors x cores, so every core gets
#                          two stateful tasks per batch whatever the config)
#   -P, --topic-partitions P  partitions of the recreated telemetry.smd (default 32; contracts/topics.yaml keeps 8 for
#                          the pipeline). The Kafka read stage runs one task per partition, so 8 would cap it at 8 cores
#   -n, --records N        backlog size (default 2000000). More than one pass of the data set is sent as
#                          several passes, each shifted in event time (see Looping)
#   -m, --max-offsets-per-trigger N  records per micro-batch (default 200000), so the drain is many equal batches
#   -r, --replicate K      weak scaling: replay.py --replicate K (K renamed copies of every machine)
#   -t, --trigger T        score_job processing-time trigger (default '0 seconds': next batch as soon as one ends)
#       --timeout S        give up after S seconds of draining (default 3600)
#       --run-id ID        name of the result (default e<E>c<C>_<UTC timestamp>)
#   -o, --out DIR          result directory (default $REPO_DIR/data/scaling)
#       --ui URL           driver Spark UI for the executor count (default $SPARK_UI_URL or http://localhost:4040)
#       --keep-topic       do not recreate the topic or replay: drain what is already there (a re-run of the
#                          same backlog). score_job still runs with --reset; the topic must still have P partitions
#       --print-config     print the resolved settings (key=value) and exit, touching nothing (used by the tests)
#
# Steps
#   (a) recreate telemetry.smd with P partitions (create_topics_cluster.py --recreate --partitions telemetry.smd=P)
#       so the backlog is exactly N records, then check that the broker reports P partitions (else stop);
#   (b) replay N records at max speed (replay.py --speedup 0 --limit), score_job not running, and wait until the
#       broker's end offsets total N (drain_monitor.py wait-written);
#   (c) start score_job via submit.sh -e E -c C -m <maxOffsetsPerTrigger> -- --reset --trigger '0 seconds';
#   (d) poll data/latency_stream/progress.jsonl until the consumer offset reaches the topic end
#       (drain_monitor.py watch, which also polls the driver UI at localhost:4040 for the executors);
#   (e) write <out>/<run-id>.json and append one row to <out>/drain_runs.csv: drain seconds, records/s, number of
#       batches, mean / p95 batch duration, executors seen (definitions in drain_monitor.py), then stop score_job:
#       SIGINT to its process group, after 60 s SIGTERM, after 20 s more SIGKILL (job_group.py stop), then wait
#       <= 60 s until the Spark master lists no active app. A run whose JSON says 'drained' exits 0 whatever
#       happens on stop.
#
# Looping: when N is more than one pass of the data set (replay.py --print-count), pass k (0-based) is sent with
# --time-offset k * (data span + 1 h), as in soak_test.sh, so no pass lies behind the watermark of the one before.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RECORDS=2000000
MAX_OFFSETS=200000
REPLICATE=""
TRIGGER_EVERY="0 seconds"
TIMEOUT_S=3600
RUN_ID=""
OUT=""
KEEP_TOPIC=0
UI_URL="${SPARK_UI_URL:-http://localhost:4040}"
EXECUTORS=""
CORES=""
SHUFFLE=""
TOPIC_PARTITIONS=32
PRINT_CONFIG=0
LOOP_GAP_S=3600
JOB_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--executors)          EXECUTORS="${2:?--executors needs a value}"; shift 2 ;;
        -c|--cores)              CORES="${2:?--cores needs a value}"; shift 2 ;;
        -p|--shuffle-partitions) SHUFFLE="${2:?--shuffle-partitions needs a value}"; shift 2 ;;
        -P|--topic-partitions)   TOPIC_PARTITIONS="${2:?--topic-partitions needs a value}"; shift 2 ;;
        --print-config)          PRINT_CONFIG=1; shift ;;
        -n|--records)            RECORDS="${2:?--records needs a value}"; shift 2 ;;
        -m|--max-offsets-per-trigger) MAX_OFFSETS="${2:?--max-offsets-per-trigger needs a value}"; shift 2 ;;
        -r|--replicate)          REPLICATE="${2:?--replicate needs a value}"; shift 2 ;;
        -t|--trigger)            TRIGGER_EVERY="${2:?--trigger needs a value}"; shift 2 ;;
        --timeout)               TIMEOUT_S="${2:?--timeout needs a value}"; shift 2 ;;
        --run-id)                RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
        -o|--out)                OUT="${2:?--out needs a value}"; shift 2 ;;
        --keep-topic)            KEEP_TOPIC=1; shift ;;
        --ui)                    UI_URL="${2:?--ui needs a value}"; shift 2 ;;
        -h|--help)               sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        --)                      shift; JOB_ARGS=("$@"); break ;;
        *)                       die "unknown option '$1' (pass score_job.py options after --)" ;;
    esac
done
EXECUTORS="${EXECUTORS:-$((SPARK_WORKER_COUNT * EXECUTORS_PER_WORKER))}"
CORES="${CORES:-${SPARK_EXECUTOR_CORES}}"
for v in EXECUTORS CORES; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "${v} must be a positive integer, got '${!v}'"
done
SHUFFLE="${SHUFFLE:-$((2 * EXECUTORS * CORES))}"
for v in SHUFFLE TOPIC_PARTITIONS RECORDS MAX_OFFSETS TIMEOUT_S; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "${v} must be a positive integer, got '${!v}'"
done
[[ -z "${REPLICATE}" || "${REPLICATE}" =~ ^[1-9][0-9]*$ ]] || die "--replicate must be a positive integer"
if [[ "${PRINT_CONFIG}" -eq 1 ]]; then
    printf '%s\n' "executors=${EXECUTORS}" "cores=${CORES}" "shuffle_partitions=${SHUFFLE}" \
        "topic_partitions=${TOPIC_PARTITIONS}" "records=${RECORDS}" "max_offsets_per_trigger=${MAX_OFFSETS}" \
        "replicate=${REPLICATE}"
    exit 0
fi

PY="${VENV_DIR}/bin/python"
[[ "$(hostname -s)" == "${SPARK_MASTER_HOST%%.*}" ]] || die "run this on ${SPARK_MASTER_HOST}"
[[ -x "${PY}" ]] || die "${PY} not found; run 30_spark_node.sh"
cd "${REPO_DIR}"
compgen -G "${REPO_DIR}/data/raw/source=smd/*.parquet" > /dev/null || die "no raw data in ${REPO_DIR}/data/raw/source=smd"

TOPIC="$("${PY}" -c "from contracts import schemas as S; print(S.TOPIC_SMD)")"
RUN_ID="${RUN_ID:-e${EXECUTORS}c${CORES}_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-${REPO_DIR}/data/scaling}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
LOGS="${OUT}/logs/${RUN_ID}"
mkdir -p "${LOGS}"
PROGRESS="${REPO_DIR}/data/latency_stream/progress.jsonl"
REPL_ARGS=()
[[ -z "${REPLICATE}" ]] || REPL_ARGS=(--replicate "${REPLICATE}")

log "drain ${RUN_ID}: ${RECORDS} records, ${EXECUTORS} executors x ${CORES} cores, ${TOPIC_PARTITIONS} topic partitions, ${SHUFFLE} shuffle partitions, maxOffsetsPerTrigger ${MAX_OFFSETS}, replicate ${REPLICATE:-none}, logs in ${LOGS}"

# score_job runs in its own process group (job_group.py start: setsid, output only to score_job.log), so a Ctrl+C
# in this terminal does not reach it and stopping it never depends on the caller's terminal or pipe.
JOB_PID=""          # = the job's process group id
MASTER_JSON="http://${SPARK_MASTER_HOST}:8080/json/"
cleanup() {         # bounded: SIGINT 60 s -> SIGTERM 20 s -> SIGKILL, then <= 60 s for the master to drop the app
    [[ -n "${JOB_PID}" ]] || return 0
    local pgid="${JOB_PID}"
    JOB_PID=""
    trap '' INT TERM                                   # a second Ctrl+C must not abandon the job half-stopped
    "${PY}" "${CLUSTER_DIR}/job_group.py" stop --pgid "${pgid}" --int-wait 60 --term-wait 20 --kill-wait 10 \
        || warn "score_job (process group ${pgid}) could not be stopped; kill it by hand: kill -KILL -- -${pgid}"
    "${PY}" "${CLUSTER_DIR}/job_group.py" wait-apps --master "${MASTER_JSON}" --timeout 60 \
        || warn "an application is still active on ${MASTER_JSON}; the next run may get no executors"
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT     # exit runs the EXIT trap (cleanup) once; without 'exit' the script would carry on
trap 'exit 143' TERM

if [[ "${KEEP_TOPIC}" -eq 0 ]]; then
    # (a) empty topic with TOPIC_PARTITIONS partitions
    log "(a) recreating ${TOPIC} with ${TOPIC_PARTITIONS} partitions"
    "${PY}" "${CLUSTER_DIR}/create_topics_cluster.py" --bootstrap "${KAFKA_BOOTSTRAP}" --recreate "${TOPIC}" \
        --partitions "${TOPIC}=${TOPIC_PARTITIONS}" \
        > "${LOGS}/topics.log" 2>&1 || { cat "${LOGS}/topics.log" >&2; die "could not recreate ${TOPIC}"; }
fi
# the broker must really hold TOPIC_PARTITIONS partitions (also with --keep-topic)
got_parts="$("${PY}" "${CLUSTER_DIR}/drain_monitor.py" partitions --bootstrap "${KAFKA_BOOTSTRAP}" --topic "${TOPIC}")" \
    || die "cannot read the partition count of ${TOPIC}"
[[ "${got_parts}" == "${TOPIC_PARTITIONS}" ]] \
    || die "${TOPIC} has ${got_parts} partitions, expected ${TOPIC_PARTITIONS}"
log "    ${TOPIC}: ${got_parts} partitions"

if [[ "${KEEP_TOPIC}" -eq 0 ]]; then
    # (b) backlog of exactly RECORDS records, job not running
    PASS="$("${PY}" src/producer/replay.py --source smd --print-count "${REPL_ARGS[@]}")"
    SPAN_S="$("${PY}" src/producer/replay.py --source smd --print-span)"
    [[ "${PASS}" =~ ^[1-9][0-9]*$ ]] || die "replay.py --print-count said '${PASS}'"
    [[ "${SPAN_S}" =~ ^[0-9]+$ ]] || die "replay.py --print-span said '${SPAN_S}'"
    passes=$(((RECORDS + PASS - 1) / PASS))
    log "(b) replaying ${RECORDS} records at max speed: ${passes} pass(es) of up to ${PASS} (span ${SPAN_S} s)"
    remaining="${RECORDS}"
    : > "${LOGS}/replay.log"
    for ((k = 0; remaining > 0; k++)); do
        limit=$((remaining < PASS ? remaining : PASS))
        offset_s=$((k * (SPAN_S + LOOP_GAP_S)))
        log "    pass $((k + 1))/${passes}: ${limit} records, event times +${offset_s} s"
        "${PY}" src/producer/replay.py --source smd --speedup 0 --limit "${limit}" --time-offset "${offset_s}" \
            "${REPL_ARGS[@]}" --bootstrap "${KAFKA_BOOTSTRAP}" >> "${LOGS}/replay.log" 2>&1 \
            || { tail -n 20 "${LOGS}/replay.log" >&2; die "replay pass $((k + 1)) failed"; }
        remaining=$((remaining - limit))
    done
    "${PY}" "${CLUSTER_DIR}/drain_monitor.py" wait-written --bootstrap "${KAFKA_BOOTSTRAP}" --topic "${TOPIC}" \
        --expect "${RECORDS}" --timeout 300 || die "the broker does not hold all ${RECORDS} records"
else
    warn "--keep-topic: draining whatever ${TOPIC} holds now"
fi

# (c) score_job on the cluster, from a clean checkpoint, as fast as it can
rm -f "${PROGRESS}"
T0_MS="$(date +%s%3N)"
log "(c) submitting score_job"
"${PY}" "${CLUSTER_DIR}/job_group.py" start --log "${LOGS}/score_job.log" -- \
    bash "${CLUSTER_DIR}/submit.sh" --executors "${EXECUTORS}" --cores "${CORES}" --shuffle-partitions "${SHUFFLE}" \
    --max-offsets-per-trigger "${MAX_OFFSETS}" -- --reset --trigger "${TRIGGER_EVERY}" "${JOB_ARGS[@]}" \
    < /dev/null > /dev/null 2>&1 &
JOB_PID=$!          # job_group.py setsid-s and execs in place: this pid is the process group id of the whole job

# (d) + (e) wait for the drain, write the JSON and the CSV row
log "(d) watching ${PROGRESS} (job pid ${JOB_PID}, log ${LOGS}/score_job.log)"
rc=0
"${PY}" "${CLUSTER_DIR}/drain_monitor.py" watch --bootstrap "${KAFKA_BOOTSTRAP}" --topic "${TOPIC}" \
    --progress "${PROGRESS}" --out "${OUT}" --run-id "${RUN_ID}" --job-pid "${JOB_PID}" --since-ms "${T0_MS}" \
    --timeout "${TIMEOUT_S}" --ui "${UI_URL}" \
    --meta "executors=${EXECUTORS}" "cores=${CORES}" "max_offsets_per_trigger=${MAX_OFFSETS}" \
           "replicate=${REPLICATE}" "topic_partitions=${TOPIC_PARTITIONS}" "shuffle_partitions=${SHUFFLE}" "trigger=${TRIGGER_EVERY// /_}" \
    || rc=$?
log "stopping score_job"
cleanup             # never fails, never blocks for more than ~150 s
# The watch above wrote the result JSON: a drained run (rc 0) stays a success even if score_job dies messily on
# stop (Py4J errors at the end of score_job.log when the JVM goes down under the Python driver are harmless then).
if [[ "${rc}" -ne 0 ]]; then
    tail -n 30 "${LOGS}/score_job.log" >&2
    die "drain ${RUN_ID} did not complete (see ${OUT}/${RUN_ID}.json and ${LOGS}/score_job.log)"
fi
log "drain done: ${OUT}/${RUN_ID}.json"
