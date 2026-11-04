#!/usr/bin/env bash
# Run ONCE on KAFKA_HOST (= spark-master) with sudo, after 00_prereqs.sh.
# Installs Docker and starts KAFKA_BROKER_COUNT Kafka 3.7.0 KRaft brokers (image apache/kafka:3.7.0, the same image as
# infra/docker-compose.yml) as containers kafka-1 .. kafka-N in combined broker+controller mode.
# KAFKA_BROKER_COUNT is 1 on the small trial cluster (one container, replication factor 1, min ISR 1) and 3 on the
# full layout (replication factor 3, min ISR 2).
#
#   host networking: broker n listens on KAFKA_PORT_BASE+100*(n-1) (clients) and +1 (controller),
#                    advertises KAFKA_HOST:<client port>, so clients and containers use the same address
#   data:            ${KAFKA_DATA_DIR}/broker-n on the host (survives container re-creation)
#   memory:          JVM heap KAFKA_HEAP, container limit KAFKA_CONTAINER_MEMORY (see the budget in cluster.env)
#
# Safe to re-run: a container is re-created only when its generated settings changed (the settings hash is
# stored as a container label), and the broker's storage is formatted by the image only on first start.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_root
validate_layout
[[ -n "${KAFKA_CLUSTER_ID:-}" ]] || die "KAFKA_CLUSTER_ID is empty; run gen_cluster_id.sh on the admin machine first"
[[ "$(hostname -s)" == "${KAFKA_HOST%%.*}" ]] || die "this VM is $(hostname -s) but KAFKA_HOST=${KAFKA_HOST}"

# ---- Docker --------------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
if ! command -v docker > /dev/null 2>&1; then
    log "installing docker.io"
    apt-get update -y
    apt-get install -y --no-install-recommends docker.io
fi
systemctl enable --now docker
docker info > /dev/null 2>&1 || die "docker daemon is not running"

if ! docker image inspect "${KAFKA_IMAGE}" > /dev/null 2>&1; then
    log "pulling ${KAFKA_IMAGE}"
    pulled=0
    for attempt in 1 2 3; do
        if docker pull "${KAFKA_IMAGE}"; then
            pulled=1
            break
        fi
        warn "docker pull failed (attempt ${attempt}/3); Docker Hub rate limit or no outbound access?"
        sleep 10
    done
    [[ "${pulled}" -eq 1 ]] || die "cannot pull ${KAFKA_IMAGE}"
fi

# ---- brokers -------------------------------------------------------------------------------
install -d -m 0755 "${KAFKA_DATA_DIR}"

for ((n = 1; n <= KAFKA_BROKER_COUNT; n++)); do
    name="kafka-${n}"
    client_port="${KAFKA_CLIENT_PORTS[n - 1]}"
    controller_port="${KAFKA_CONTROLLER_PORTS[n - 1]}"
    data_dir="${KAFKA_DATA_DIR}/broker-${n}"
    install -d -o 1000 -g 1000 -m 0750 "${data_dir}"       # the image runs as uid 1000 (appuser)

    # Environment variables of the apache/kafka image: KAFKA_FOO_BAR -> foo.bar in server.properties.
    env_args=(
        -e "CLUSTER_ID=${KAFKA_CLUSTER_ID}"
        -e "KAFKA_NODE_ID=${n}"
        -e "KAFKA_PROCESS_ROLES=broker,controller"
        -e "KAFKA_LISTENERS=PLAINTEXT://:${client_port},CONTROLLER://:${controller_port}"
        -e "KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://${KAFKA_HOST}:${client_port}"
        -e "KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER"
        -e "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT"
        -e "KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT"
        -e "KAFKA_CONTROLLER_QUORUM_VOTERS=${KAFKA_QUORUM_VOTERS}"
        -e "KAFKA_LOG_DIRS=/var/lib/kafka/data"
        -e "KAFKA_LOG_RETENTION_HOURS=${KAFKA_LOG_RETENTION_HOURS}"
        -e "KAFKA_AUTO_CREATE_TOPICS_ENABLE=false"
        # 3 brokers: RF 3 + min ISR 2, so one broker can be down and producers (acks=all) keep working.
        # 1 broker: RF 1 + min ISR 1 (no redundancy; the internal topics could not be created with more).
        -e "KAFKA_DEFAULT_REPLICATION_FACTOR=${KAFKA_REPLICATION}"
        -e "KAFKA_MIN_INSYNC_REPLICAS=${KAFKA_MIN_ISR}"
        -e "KAFKA_UNCLEAN_LEADER_ELECTION_ENABLE=false"
        -e "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=${KAFKA_REPLICATION}"
        -e "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=${KAFKA_REPLICATION}"
        -e "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=${KAFKA_MIN_ISR}"
        -e "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=3000"
        -e "KAFKA_HEAP_OPTS=-Xms${KAFKA_HEAP} -Xmx${KAFKA_HEAP}"
    )
    cfg_hash="$(printf '%s\n' "${KAFKA_IMAGE}" "${KAFKA_CONTAINER_MEMORY}" "${data_dir}" "${env_args[@]}" | sha256sum | cut -c1-16)"

    run_container() {
        docker run -d --name "${name}" --network host --restart unless-stopped \
            --memory "${KAFKA_CONTAINER_MEMORY}" --ulimit nofile=100000:100000 --stop-timeout 120 \
            --log-opt max-size=50m --log-opt max-file=3 \
            --label "cluster.cfg=${cfg_hash}" \
            -v "${data_dir}:/var/lib/kafka/data" \
            "${env_args[@]}" "${KAFKA_IMAGE}" > /dev/null
    }

    if ! docker inspect "${name}" > /dev/null 2>&1; then
        log "creating ${name} (clients :${client_port}, controller :${controller_port})"
        run_container
        # not waiting: with several voters the first broker only becomes ready once a second one is up; deploy.sh waits for all
    else
        current_hash="$(docker inspect -f '{{ index .Config.Labels "cluster.cfg" }}' "${name}")"
        if [[ "${current_hash}" != "${cfg_hash}" ]]; then
            log "${name}: settings changed, re-creating (data in ${data_dir} is kept)"
            docker rm -f "${name}" > /dev/null
            run_container
            wait_port "${KAFKA_HOST}" "${client_port}" 180 || die "${name} did not come back on :${client_port}; see: docker logs ${name}"
        elif [[ "$(docker inspect -f '{{.State.Running}}' "${name}")" != "true" ]]; then
            log "${name}: not running, starting"
            docker start "${name}" > /dev/null
            wait_port "${KAFKA_HOST}" "${client_port}" 180 || die "${name} did not start on :${client_port}; see: docker logs ${name}"
        else
            log "${name}: running, settings unchanged"
        fi
    fi
done

log "kafka containers installed on ${KAFKA_HOST}; bootstrap: ${KAFKA_BOOTSTRAP}"
