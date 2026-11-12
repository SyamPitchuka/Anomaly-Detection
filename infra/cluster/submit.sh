#!/usr/bin/env bash
# spark-submit src/streaming/score_job.py to the standalone cluster (client mode).
#
# Run ON THE SPARK MASTER VM (the driver must be reachable from the executors), as the 'spark' user:
#     sudo -u spark bash infra/cluster/submit.sh -e 2 -c 2
#     sudo -u spark bash infra/cluster/submit.sh -e 4 -c 2 -k /srv/spark-shared/Anomaly-Detection/checkpoints -- --max-seconds 600
#
#   -e, --executors N        number of executors (default: every executor the cluster can hold)
#   -c, --cores N            cores per executor   (default: SPARK_EXECUTOR_CORES of the VM profile)
#   -k, --checkpoint-dir D   checkpoint ROOT on shared storage (default: $CHECKPOINT_DIR = $REPO_DIR/checkpoints);
#                            the query checkpoint is D/score/source=smd. Must be under $SHARED_DIR.
#   -p, --shuffle-partitions N  spark.sql.shuffle.partitions (default 16). Keep it constant for the life of a
#                            checkpoint: Spark fixes the state-store partition count when the query first starts.
#       --skip-model-check   do not ssh to the workers to compare the model files
#       --dry-run            print the configuration and the spark-submit command, run nothing
#   --                       everything after it goes to score_job.py (--max-seconds, --reset, --tier1 v1, ...)
#
# What it does
#   * --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 (override: KAFKA_PACKAGE=...)
#   * --executor-cores C --total-executor-cores E*C  -> exactly E executors of C cores (standalone mode has no
#     --num-executors); refuses a request the workers cannot hold (cores or memory).
#   * checkpoints on the shared NFS path (refused if not under $SHARED_DIR or if the share is not mounted).
#   * models: every executor loads models/ from $REPO_DIR, the NFS share mounted at the SAME path on all nodes
#     (spark.executorEnv.PYTHONPATH=$REPO_DIR; score_job.py resolves models relative to its own file). Nothing is
#     copied per job; instead this script ssh-es to each worker and compares a checksum of models/ with the
#     master's, so a stale or missing model fails the submit instead of failing inside a Python worker.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

KAFKA_PACKAGE="${KAFKA_PACKAGE:-org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1}"
EXECUTORS="$((SPARK_WORKER_COUNT * EXECUTORS_PER_WORKER))"
CORES="${SPARK_EXECUTOR_CORES}"
CKPT_ROOT="${CHECKPOINT_DIR}"
SHUFFLE_PARTITIONS=16
MODEL_CHECK=1
DRY_RUN=0
JOB_ARGS=()

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--executors)          EXECUTORS="${2:?--executors needs a value}"; shift 2 ;;
        -c|--cores)              CORES="${2:?--cores needs a value}"; shift 2 ;;
        -k|--checkpoint-dir)     CKPT_ROOT="${2:?--checkpoint-dir needs a value}"; shift 2 ;;
        -p|--shuffle-partitions) SHUFFLE_PARTITIONS="${2:?--shuffle-partitions needs a value}"; shift 2 ;;
        --skip-model-check)      MODEL_CHECK=0; shift ;;
        --dry-run)               DRY_RUN=1; shift ;;
        -h|--help)               usage; exit 0 ;;
        --)                      shift; JOB_ARGS=("$@"); break ;;
        *)                       usage >&2; die "unknown option '$1' (pass score_job.py options after --)" ;;
    esac
done

[[ "${EXECUTORS}" =~ ^[1-9][0-9]*$ ]] || die "--executors must be a positive integer, got '${EXECUTORS}'"
[[ "${CORES}" =~ ^[1-9][0-9]*$ ]] || die "--cores must be a positive integer, got '${CORES}'"
[[ "${SHUFFLE_PARTITIONS}" =~ ^[1-9][0-9]*$ ]] || die "--shuffle-partitions must be a positive integer"

# ---- can the workers hold E executors x C cores? --------------------------------------------
PER_WORKER_CAP=$((SPARK_WORKER_CORES / CORES))
[[ "${PER_WORKER_CAP}" -ge 1 ]] || die "--cores ${CORES} exceeds SPARK_WORKER_CORES=${SPARK_WORKER_CORES}"
MAX_EXECUTORS=$((SPARK_WORKER_COUNT * PER_WORKER_CAP))
[[ "${EXECUTORS}" -le "${MAX_EXECUTORS}" ]] \
    || die "${EXECUTORS} executors x ${CORES} cores does not fit: ${SPARK_WORKER_COUNT} workers x ${SPARK_WORKER_CORES} cores hold at most ${MAX_EXECUTORS}"
TOTAL_CORES=$((EXECUTORS * CORES))

# Standalone spreads executors over the workers, so the busiest worker gets ceil(E / workers) of them.
WORST_PER_WORKER=$(((EXECUTORS + SPARK_WORKER_COUNT - 1) / SPARK_WORKER_COUNT))
exec_mb="$(mem_to_mb "${SPARK_EXECUTOR_MEMORY}")"
overhead_mb="$(mem_to_mb "${SPARK_EXECUTOR_MEMORY_OVERHEAD}")"
[[ $((WORST_PER_WORKER * exec_mb)) -le "$(mem_to_mb "${SPARK_WORKER_MEMORY}")" ]] \
    || die "${WORST_PER_WORKER} executors x ${SPARK_EXECUTOR_MEMORY} do not fit in SPARK_WORKER_MEMORY=${SPARK_WORKER_MEMORY}"
need_mb=$((WORST_PER_WORKER * (exec_mb + overhead_mb) + $(mem_to_mb "${SPARK_DAEMON_MEMORY}") + OS_RESERVE_MB))
[[ "${need_mb}" -le "$(mem_to_mb "${VM_MEMORY}")" ]] \
    || die "a worker VM would need ${need_mb} MB for ${WORST_PER_WORKER} executors but VM_MEMORY is ${VM_MEMORY}"

# ---- checkpoints must be on shared storage --------------------------------------------------
CKPT_ROOT="$(realpath -m "${CKPT_ROOT}")"
case "${CKPT_ROOT}/" in
    "${SHARED_DIR}"/*) ;;
    *) die "checkpoint dir ${CKPT_ROOT} is not under the shared storage ${SHARED_DIR}; executors and a restarted driver would not see it" ;;
esac
SCORE_CHECKPOINT_DIR="${CKPT_ROOT}/score/source=smd"

ENTRY="${CLUSTER_DIR}/score_entry.py"
PY="${VENV_DIR}/bin/python"
SUBMIT="${SPARK_HOME}/bin/spark-submit"

CMD=("${SUBMIT}"
     --master "${SPARK_MASTER_URL}"
     --deploy-mode client
     --packages "${KAFKA_PACKAGE}"
     --executor-cores "${CORES}"
     --total-executor-cores "${TOTAL_CORES}"
     "${ENTRY}"
     --input kafka
     --master "${SPARK_MASTER_URL}"          # score_job's own --master (its default local[*] would override spark-submit's)
     --shuffle-partitions "${SHUFFLE_PARTITIONS}"
     "${JOB_ARGS[@]}")

log "score_job: ${EXECUTORS} executors x ${CORES} cores = ${TOTAL_CORES} cores on ${SPARK_MASTER_URL}"
log "  Kafka        ${KAFKA_BOOTSTRAP}   package ${KAFKA_PACKAGE}"
log "  checkpoints  ${SCORE_CHECKPOINT_DIR}"
log "  models       ${REPO_DIR}/models (shared mount, checked on ${SPARK_WORKER_COUNT} workers)"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  %q' "${CMD[@]}"; printf '\n'
    exit 0
fi

# ---- preflight --------------------------------------------------------------------------------
[[ "$(hostname -s)" == "${SPARK_MASTER_HOST%%.*}" ]] || die "run this on ${SPARK_MASTER_HOST} (the driver must be reachable from the executors)"
[[ -x "${SUBMIT}" ]] || die "${SUBMIT} not found; run 30_spark_node.sh"
[[ -x "${PY}" ]] || die "${PY} not found; run 30_spark_node.sh"
mountpoint -q "${SHARED_DIR}" || is_nfs_server || die "${SHARED_DIR} is not mounted"
mkdir -p "${SCORE_CHECKPOINT_DIR}"
[[ -w "${SCORE_CHECKPOINT_DIR}" ]] || die "${SCORE_CHECKPOINT_DIR} is not writable by $(id -un)"

models_sum() { # models_sum: checksum of file names + contents under $REPO_DIR/models (run on a node)
    (cd "${REPO_DIR}/models" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}
MODELS_REL="$(cd "${REPO_DIR}" && "${PY}" -c "from contracts import schemas as S; print(S.MODELS['ae'].replace('<source>', 'smd'))")"
[[ -f "${REPO_DIR}/${MODELS_REL}/model.pt" ]] || die "no autoencoder at ${REPO_DIR}/${MODELS_REL}: train it and run deploy.sh repo"
if [[ "${MODEL_CHECK}" -eq 1 ]]; then
    want="$(models_sum)"
    for w in "${SPARK_WORKER_HOSTS[@]}"; do
        got="$(rsh "${w}" "cd '${REPO_DIR}/models' && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1")" \
            || die "cannot ssh to ${w} to check the models (use --skip-model-check to skip)"
        [[ "${got}" == "${want}" ]] || die "models/ on ${w} differ from the master's (is ${SHARED_DIR} mounted there? run deploy.sh repo)"
        log "  models identical on ${w} (${want:0:12})"
    done
fi

export PYSPARK_PYTHON="${PY}" PYSPARK_DRIVER_PYTHON="${PY}"
export CLUSTER_KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP}" SCORE_CHECKPOINT_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${REPO_DIR}"
exec "${CMD[@]}"
