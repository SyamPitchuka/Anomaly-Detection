#!/usr/bin/env bash
# 30-minute soak test: score_job on the cluster (via submit.sh) while the producer replays Alibaba at maximum
# speed (replay.py --speedup 0). Throughput is recorded every minute to <out>/throughput.csv.
#
# Run ON THE SPARK MASTER VM as the 'spark' user, with the topics created and data/raw/source=alibaba present:
#     sudo -u spark bash infra/cluster/soak_test.sh -e 2 -c 2
#     sudo -u spark bash infra/cluster/soak_test.sh -e 4 -c 2 -d 30 -- --tier1 v1      # after -- : score_job options
#
#   -e, --executors N   executors for score_job (default: all the cluster can hold)   } passed to submit.sh
#   -c, --cores N       cores per executor (default: profile value)                   }
#   -d, --minutes N     soak duration in minutes (default 30)
#   -s, --source S      alibaba (default) or smd: which raw layer / topic the replay writes
#   -o, --out DIR       result directory (default $REPO_DIR/data/soak/<UTC timestamp>)
#
# Files in the result directory: throughput.csv (one row per minute), score_job.log, replay.log, sampler.log.
#
# NOTE: src/streaming/score_job.py reads telemetry.smd only (it has the SMD autoencoder, 38 metrics). Replaying
# Alibaba therefore loads and measures Kafka + the producer, while the job's own rows/s column stays 0. Use
# --source smd to drive the job itself. The replay restarts from the first record whenever the data set is exhausted
# before the time is up, so the load lasts the full duration (event times repeat; the Alibaba job, if any, must not
# depend on them being unique).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MINUTES=30
SOURCE=alibaba
OUT=""
SUBMIT_ARGS=()
JOB_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--executors) SUBMIT_ARGS+=(--executors "${2:?--executors needs a value}"); shift 2 ;;
        -c|--cores)     SUBMIT_ARGS+=(--cores "${2:?--cores needs a value}"); shift 2 ;;
        -d|--minutes)   MINUTES="${2:?--minutes needs a value}"; shift 2 ;;
        -s|--source)    SOURCE="${2:?--source needs a value}"; shift 2 ;;
        -o|--out)       OUT="${2:?--out needs a value}"; shift 2 ;;
        -h|--help)      sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        --)             shift; JOB_ARGS=("$@"); break ;;
        *)              die "unknown option '$1'" ;;
    esac
done
[[ "${MINUTES}" =~ ^[1-9][0-9]*$ ]] || die "--minutes must be a positive integer"
[[ "${SOURCE}" == "alibaba" || "${SOURCE}" == "smd" ]] || die "--source must be alibaba or smd"

PY="${VENV_DIR}/bin/python"
[[ "$(hostname -s)" == "${SPARK_MASTER_HOST%%.*}" ]] || die "run this on ${SPARK_MASTER_HOST}"
[[ -x "${PY}" ]] || die "${PY} not found; run 30_spark_node.sh"
cd "${REPO_DIR}"
compgen -G "${REPO_DIR}/data/raw/source=${SOURCE}/*.parquet" > /dev/null \
    || die "no raw data in ${REPO_DIR}/data/raw/source=${SOURCE}"
TOPIC="$("${PY}" -c "from contracts import schemas as S; print({'smd': S.TOPIC_SMD, 'alibaba': S.TOPIC_ALIBABA}['${SOURCE}'])")"

OUT="${OUT:-${REPO_DIR}/data/soak/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
PROGRESS="${REPO_DIR}/data/latency_stream/progress.jsonl"
DURATION_S=$((MINUTES * 60))
JOB_STOP_S=$((DURATION_S + 180))          # job outlives the load by 3 min so the last batches are logged

if [[ "${SOURCE}" == "alibaba" ]]; then
    warn "score_job consumes telemetry.smd; with --source alibaba its rows/s column will be 0 (Kafka/producer throughput is still measured)"
fi
log "soak: ${MINUTES} min, replay ${SOURCE} -> ${TOPIC} at max speed, results in ${OUT}"

JOB_PID=""
SAMPLER_PID=""
cleanup() {
    local pid
    for pid in "${SAMPLER_PID}" "${JOB_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2> /dev/null; then
            kill -INT "${pid}" 2> /dev/null || true     # SIGINT: score_job stops its query, spark-submit exits
        fi
    done
}
trap cleanup EXIT INT TERM

# 1. score_job on the cluster; wait until its query is running before the load starts
bash "${CLUSTER_DIR}/submit.sh" "${SUBMIT_ARGS[@]}" -- --max-seconds "${JOB_STOP_S}" "${JOB_ARGS[@]}" \
    > "${OUT}/score_job.log" 2>&1 &
JOB_PID=$!
log "score_job submitted (pid ${JOB_PID}), waiting for 'score_job started' (log: ${OUT}/score_job.log)"
for ((i = 0; i < 180; i += 2)); do
    if grep -q "score_job started" "${OUT}/score_job.log" 2> /dev/null; then
        break
    fi
    kill -0 "${JOB_PID}" 2> /dev/null || { tail -n 20 "${OUT}/score_job.log" >&2; die "score_job exited before starting"; }
    sleep 2
done
grep -q "score_job started" "${OUT}/score_job.log" || die "score_job did not start within 180 s"

# 2. per-minute sampler
"${PY}" "${CLUSTER_DIR}/soak_sampler.py" --bootstrap "${KAFKA_BOOTSTRAP}" --topic "${TOPIC}" \
    --progress "${PROGRESS}" --out "${OUT}" --minutes "${MINUTES}" > "${OUT}/sampler.log" 2>&1 &
SAMPLER_PID=$!

# 3. replay at max speed for the whole duration; restart it if the data set runs out first
: > "${OUT}/replay.log"
deadline=$((SECONDS + DURATION_S))
runs=0
while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    runs=$((runs + 1))
    log "replay run ${runs} ($((deadline - SECONDS)) s left)"
    rc=0
    timeout -s INT -k 15 "$((deadline - SECONDS))" \
        "${PY}" src/producer/replay.py --source "${SOURCE}" --speedup 0 --bootstrap "${KAFKA_BOOTSTRAP}" \
        >> "${OUT}/replay.log" 2>&1 || rc=$?
    # 0 = data set exhausted (loop again), 124 = time is up, 130 = interrupted by timeout's SIGINT
    if [[ "${rc}" -ne 0 && "${rc}" -ne 124 && "${rc}" -ne 130 && "${rc}" -ne 137 ]]; then
        tail -n 20 "${OUT}/replay.log" >&2
        die "replay failed with exit code ${rc}"
    fi
done
log "replay finished after ${runs} run(s)"

# 4. wait for the sampler's last sample and print the table
wait "${SAMPLER_PID}" || warn "sampler exited with an error; see ${OUT}/sampler.log"
SAMPLER_PID=""
cat "${OUT}/sampler.log"
log "stopping score_job"
kill -INT "${JOB_PID}" 2> /dev/null || true
wait "${JOB_PID}" 2> /dev/null || true
JOB_PID=""
log "soak done: ${OUT}/throughput.csv"
