# Cluster setup (Ubuntu VMs): Kafka 3.7.0 KRaft (Docker), Spark 3.5.1 standalone, shared NFS storage

This is the multi-VM counterpart of the laptop setup (`infra/docker-compose.yml`, `infra/create_topics.py`).
Versions, topic names and paths are the frozen ones from `CLAUDE.md`, `contracts/schemas.py` and
`contracts/topics.yaml`. Nothing under `contracts/` or `tools/` is modified.

The default layout is the **small trial cluster that fits a 6 vCPU Azure quota: 3 VMs, 1 Kafka broker**. The same scripts also build the
bigger layout (4 workers, 3 brokers); see "Bigger layout" below.

> **Not yet run on real VMs.** The scripts pass `bash -n` and `py_compile` (checked before the last round of
> edits), but nobody has run them on Ubuntu VMs. Run `deploy.sh all` on throw-away VMs first and fix what you find.

Running this on Azure? Follow the runbook in [`AZURE.md`](AZURE.md) (existing `spark-rg` / `spark-vnet`, India South Central, VMs, deploy,
verify, cleanup).

## Layout (3 VMs, 2 vCPU / 8 GB RAM each)

| VM | Azure size | Runs |
| --- | --- | --- |
| `spark-master` | `Standard_D2as_v4` | Spark master (7077, UI 8080), **1 Kafka broker as a Docker container** `kafka-1`, the NFS server, the Spark **drivers**, and the admin work (`deploy.sh`, Python tools) |
| `spark-w1`, `spark-w2` | `Standard_D2s_v4` | one Spark worker each: 2 cores, **1 executor of `spark.executor.cores=2`** |
| shared storage | | `spark-master:/srv/spark-shared`, mounted at the same path on all 3 VMs (NFSv4.1) |

Spark totals: 2 workers x 1 executor = **2 executors, 4 cores**. This follows from `SPARK_WORKER_CORES=2` and
`SPARK_EXECUTOR_CORES=2`: standalone mode starts `floor(2 / 2) = 1` executor per worker.

**Kafka in Docker on the master.** The image is `apache/kafka:3.7.0`, the same as the laptop compose file. The broker is a KRaft node in
combined `broker,controller` mode with host networking:

| Broker (container) | Clients | Controller | Data dir on the VM |
| --- | --- | --- | --- |
| `kafka-1` (node 1) | `spark-master:9092` | 9093 | `/var/lib/kafka/data/broker-1` |

Bootstrap list: `spark-master:9092`.

> **Trade-off.** With one broker there is **no replication** (replication factor 1, min ISR 1) and no failover: restarting the container
> or losing `spark-master` stops Kafka, the Spark master and the NFS server together. `contracts/topics.yaml` asks for
> `replication_cluster: 3`; `create_topics_cluster.py` caps that at the number of brokers (here 1) without editing `contracts/`.
> That is acceptable for a trial and for latency / throughput measurements, not for production, and it means the replication
> overhead of a 3-broker cluster is not measured by this layout.

### Bigger layout (more quota)

Put these in `cluster.env` (nothing else in the scripts changes):

```bash
VM_PROFILE=vm_8gb_4cpu                                  # 4 vCPU / 8 GB VMs (Standard_F4s_v2)
KAFKA_BROKER_COUNT=3                                    # kafka-1..3 on spark-master: ports 9092/9192/9292, RF 3, min ISR 2
SPARK_WORKER_HOSTS=(spark-w1 spark-w2 spark-w3 spark-w4)
```

That gives 4 workers x 2 executors = 8 executors, 16 cores, and the bootstrap list
`spark-master:9092,spark-master:9192,spark-master:9292`. Broker n listens on client port `9092 + 100*(n-1)` and controller port `9093 + 100*(n-1)`.
`KAFKA_BROKER_COUNT` should be 1 or 3 (an even count gives no extra fault tolerance).
Do not run 3 brokers on the 2 vCPU profile: the master budget would be 7552 MB of 8192 MB, which leaves no room for a second driver.

### Memory budget (8 GB per VM)

`lib.sh` computes this from the VM profile (`profiles/vm_8gb_2cpu.env`, see "VM profiles" below) and `deploy.sh` refuses to continue if a VM would exceed `VM_MEMORY`
(`deploy.sh env` prints the numbers). The check compares against the nominal 8 GB; the kernel of an "8 GB" VM reports roughly
7.6-7.8 GB as `MemTotal`, so the worker (7168 MB) has about 0.5 GB of slack and the master (4480 MB) about 3 GB.

| VM | Component | MB |
| --- | --- | --- |
| each worker | 1 executor x (4 GB executor + 1536 MB `memoryOverhead`) | 5632 |
| | worker daemon | 512 |
| | OS reserve (kernel, sshd, page cache) | 1024 |
| | **total of 8192** | **7168** |
| `spark-master` | 1 Kafka container, 1536 MB limit (1 GB JVM heap + off-heap + page cache) | 1536 |
| | master daemon 512 + driver 1024 + driver overhead 384 | 1920 |
| | OS reserve | 1024 |
| | **total of 8192** | **4480** |

The memory overhead matters more than usual here: the Python workers (pandas, numpy, torch in `mapInPandas` for the autoencoder)
live outside the executor's heap, inside the overhead. If you see `Container killed` / OOM-killer messages, lower the load rather than
raising `SPARK_EXECUTOR_MEMORY` first.

### One thread per Python process (`OMP_NUM_THREADS=1`)

Every executor core already runs one Python worker, so a multi-threaded BLAS inside each would oversubscribe the cores of a worker
VM and thrash. `30_spark_node.sh` therefore sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS`
to `1` in `spark-env.sh` (daemons and local drivers) **and** as `spark.executorEnv.*` in `spark-defaults.conf` (executors and their Python
workers). `run_job.py` and `check_spark_cluster.py` set the same variables for the driver process. (`score_job.py` additionally calls
`torch.set_num_threads(1)`.)

### Ports

Kafka 9092/9093 on `spark-master` (9192/9193 and 9292/9293 with 3 brokers); Spark 7077 (master), 8080 / 8081 (UIs), plus random driver,
block-manager and executor ports (keep the VMs in one security group with all ports open between them); NFS 2049.

### Why NFS (and not HDFS or S3)

Your request left the storage as `<HDFS/S3/NFS>`. I chose NFS because it is the only option the repo supports without code changes:

* Every job builds its checkpoint path as `ROOT / "checkpoints" / ...` and passes `CHECKPOINT_DIR.as_posix()` to
  `checkpointLocation`. That is a plain filesystem path, which works on any POSIX mount and breaks on `hdfs://` or `s3a://`.
* `score_job.py` puts the repo root on the workers' `PYTHONPATH`, so the repo must be visible at the same path on every node.
  The NFS share holds the whole repo (`$REPO_DIR`), so `checkpoints/`, `data/` and `models/` are shared automatically.
* Nothing in the repo ships the Hadoop-AWS jars or an HDFS client config.

HDFS or S3 would mean changing the `*_DIR` constants in `src/streaming/*.py` (not covered here), adding `hadoop-aws`
or an HDFS client, and re-checking the Parquet `data/` layout. Tell me if you want that variant.

NFS caveats: it is a single point of failure (the master VM), and Spark's file-based checkpoint log needs NFSv4 with
`hard` mounts, which `11_nfs_client.sh` sets up. Don't run one query's checkpoint from two applications at once.

## Local checks before touching any VM

Run from the repo root with the venv active (nothing here needs a VM, Kafka or Spark):

```bash
python -m pytest tests/test_cluster_tools.py -q            # topic helper, env parsing, lib.sh layout and memory maths
for f in infra/cluster/*.sh; do bash -n "$f" && echo "syntax ok: $f"; done
python -m py_compile infra/cluster/*.py
```

## Prerequisites

* 3 Ubuntu 22.04 / 24.04 VMs with 2 vCPU / 8 GB RAM each (master + 2 workers). Internet access: Docker Hub (pull `apache/kafka:3.7.0`),
  Apache archive (Spark), PyPI, Maven Central (Kafka connector), the deadsnakes PPA if Python 3.11 is not in the Ubuntu repos.
* Hostnames resolve on every VM (DNS or `/etc/hosts`). Each VM's `hostname -s` must equal the short name in `cluster.env`.
* An admin machine with `ssh`, `tar`, `curl`, and key-based ssh to every VM as `SSH_USER`, with passwordless `sudo` on them. On Azure this is
  `spark-master` itself (only a machine inside the VNet resolves the broker's advertised name); otherwise Git Bash on Windows, WSL or Linux.
* On the admin machine, the repo venv (`.venv`, see the top-level README) for the topic step.

## Quick start

```bash
cd Anomaly-Detection
cp infra/cluster/cluster.env.example infra/cluster/cluster.env     # edit NFS_NETWORK (and SSH_USER, hostnames if yours differ)
bash infra/cluster/gen_cluster_id.sh                               # writes KAFKA_CLUSTER_ID once
bash infra/cluster/deploy.sh all                                   # ssh env prereqs nfs repo kafka spark topics health
```

`deploy.sh all` ends with the health check and prints `PASS` or `FAIL`. Each phase is idempotent; run one on its own with
`bash infra/cluster/deploy.sh <phase>`:

| Phase | What it does |
| --- | --- |
| `ssh` | checks ssh + passwordless sudo on every VM |
| `env` | writes `cluster.runtime.env` (bootstrap list, broker count, master URL, sizes) for the Python tools and prints the memory budget |
| `prereqs` | `00_prereqs.sh` on all VMs: Java 17, base tools, the `spark` user (fixed uid), hostname resolution check |
| `nfs` | `10_nfs_server.sh` on the NFS host, `11_nfs_client.sh` on the other Spark VMs |
| `repo` | copies the working tree (without `.venv`, `.git`, `checkpoints/`) to `$REPO_DIR` on the share |
| `kafka` | `20_kafka_broker.sh` once on `KAFKA_HOST`: installs Docker, starts the broker container(s), then waits for the client port(s) |
| `spark` | `30_spark_node.sh` on master and workers: Spark 3.5.1, Python 3.11 venv at `$VENV_DIR`, requirements, config, systemd |
| `topics` | `create_topics_cluster.py --brokers $KAFKA_BROKER_COUNT` |
| `health` | `health_check.sh all` |

Re-run `deploy.sh repo` after code changes. It never touches `checkpoints/`.

If you prefer to run by hand, copy `infra/cluster/` plus `cluster.env` to each VM and run, as root, the numbered scripts in
order (`00` everywhere, then `10` on the NFS host / `11` on the other Spark VMs, then `20` on `KAFKA_HOST`, then `30` on every Spark VM).
`30_spark_node.sh` reads `requirements.txt` from the share, so the repo must be copied there first.

## What the scripts configure

**Kafka** (`20_kafka_broker.sh`): installs `docker.io`, pulls `apache/kafka:3.7.0`, and runs `kafka-1..N` (N = `KAFKA_BROKER_COUNT`) with
`--network host`, `--restart unless-stopped`, a memory limit of `KAFKA_CONTAINER_MEMORY` and a JVM heap of `KAFKA_HEAP`. Settings are passed
as `KAFKA_*` environment variables: `controller.quorum.voters` lists all nodes, `advertised.listeners` is `spark-master:<port>`,
`auto.create.topics.enable=false` (as on laptops), `unclean.leader.election.enable=false`. Replication follows the broker count:

| `KAFKA_BROKER_COUNT` | `default.replication.factor` and internal topics | `min.insync.replicas` | What survives |
| --- | --- | --- | --- |
| 1 (default) | 1 | 1 | nothing: no redundancy |
| 3 | 3 | 2 | one broker (container) down, producers with `acks=all` keep working |

A hash of the settings is stored as a container label, so re-running re-creates a container only when its settings changed; the data
directory is kept. Listeners are `PLAINTEXT` (no TLS or auth), matching the laptop setup, so keep the VMs on a private network.

**Spark** (`30_spark_node.sh`): `/opt/spark`, `spark-master.service` / `spark-worker.service`, and `conf/spark-defaults.conf` containing
`spark.master`, `spark.executor.cores=2`, executor memory and memory overhead, driver memory, `spark.cores.max` (all worker cores: 4 here),
`spark.executorEnv.PYTHONPATH=$REPO_DIR`, the single-thread variables above, UTC session time zone and `TIMESTAMP_MICROS` (the repo's
Spark rules). The venv at `$VENV_DIR` (local disk, same path everywhere) has `requirements.txt`, and `requirements-ml.txt` (torch 2.3.1 CPU)
when `INSTALL_ML=yes`, because `score_job.py` loads the autoencoder on the workers. It fails if pip's PySpark version differs from the
cluster's Spark version. The Kafka connector (`org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1`) is still set by each job through
`spark.jars.packages`, so the driver needs Maven access; for an offline cluster, drop the connector and its dependency jars into
`/opt/spark/jars` on every node.

## Topics

```bash
python infra/cluster/create_topics_cluster.py                      # uses cluster.runtime.env (bootstrap and broker count)
python infra/cluster/create_topics_cluster.py --bootstrap spark-master:9092 --brokers 1
```

Reads `contracts/topics.yaml` and creates every topic with its `partitions_cluster` and a replication factor of
`min(replication_cluster, brokers)`: with the default single broker that is `telemetry.smd` 8 / 1, `telemetry.alibaba` 32 / 1, `anomalies` 8 / 1;
with three brokers 8 / 3, 32 / 3, 8 / 3. The broker count comes from `--brokers`, else `KAFKA_BROKER_COUNT` in `cluster.runtime.env`, else 3.
It is safe to re-run: existing topics are only verified (partitions, replication factor, every partition has a leader) and a mismatch prints
`WRONG` and exits 1. Kafka cannot lower a partition count; delete and recreate the topic to change it.

## Health checks

```bash
bash infra/cluster/health_check.sh            # all four groups; ends with PASS / FAIL, exit code 0 / 1
bash infra/cluster/health_check.sh kafka      # or: spark | storage | memory
```

| Group | Checks |
| --- | --- |
| `kafka` | each container running, not OOM-killed, 0 restarts; client and controller ports reachable; KRaft quorum has a leader and `KAFKA_BROKER_COUNT` voters; no under-replicated and no unavailable partitions; topics match `topics.yaml` (replication capped at the broker count) |
| `spark` | master UI answers, status `ALIVE`, all workers `ALIVE` with `SPARK_WORKER_CORES` cores each (2 workers x 2 cores) |
| `storage` | every Spark VM has the share mounted (`nfs4`), the `spark` user can write `checkpoints/`, and every node sees the files the other nodes wrote |
| `memory` | every VM has at least 400 MB `MemAvailable` (most useful while a job is running) |

Two deeper checks run real data / tasks:

```bash
# Kafka round trip on the cluster (needs data/sample/telemetry_smd.jsonl: python tools/fake_data.py --layers jsonl)
python infra/cluster/check_kafka_cluster.py

# Spark: run on spark-master, as the spark user, from the repo on the share
cd /srv/spark-shared/Anomaly-Detection
sudo -u spark /opt/anomaly-venv/bin/python infra/cluster/check_spark_cluster.py
```

`check_spark_cluster.py` passes when the expected executors register (2: one per worker on 2 hosts), executors use the driver's Python minor
version and can `import contracts.schemas`, and files written from every worker host are read back by the driver under `checkpoints/_gauge`.

Manual spot checks: Spark UI at `http://spark-master:8080`; on `spark-master`,
`sudo docker exec kafka-1 /opt/kafka/bin/kafka-metadata-quorum.sh --bootstrap-server spark-master:9092 describe --status`,
`sudo docker exec kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server spark-master:9092 --describe`,
`sudo docker logs --tail 100 kafka-1`, `sudo docker stats --no-stream`, and `journalctl -u spark-master|spark-worker`.

With the 3-broker layout a failover test is worth doing once: `sudo docker stop kafka-2` on `spark-master`, run `check_kafka_cluster.py`
(it should still PASS, `health_check.sh kafka` reports the stopped container and under-replicated partitions), then `sudo docker start kafka-2`
and wait for `health_check.sh kafka` to return to `PASS`. This tests one broker failing, not the loss of the VM. With one broker there is
nothing to fail over to.

## Pointing `tools/check_kafka.py` at the cluster

`tools/check_kafka.py` is a frozen gauge. It takes the broker address from the constant `contracts.schemas.KAFKA_BOOTSTRAP`
(`localhost:9092`) and compares partition counts with the `partitions_laptop` values. Neither can be set from the command line or an
environment variable, and editing `contracts/` or `tools/` needs Syam's approval. So use the wrapper, which sets both **in memory**
and then runs the real script, so the produce / consume / key checks and the final `PASS` / `FAIL` line are the original ones:

```bash
python infra/cluster/check_kafka_cluster.py                                       # bootstrap from cluster.runtime.env
python infra/cluster/check_kafka_cluster.py --bootstrap spark-master:9092
KAFKA_BOOTSTRAP=spark-master:9092 python infra/cluster/check_kafka_cluster.py
```

Run it from the repo root with the venv active. The machine must resolve `spark-master` and reach port 9092 (and 9192/9292 with three
brokers), because the brokers advertise `spark-master:<port>` (an ssh tunnel to `localhost:9092` will not work). Order of precedence:
`--bootstrap`, then `$KAFKA_BOOTSTRAP`, then `cluster.runtime.env`.

If the change is approved, the permanent fix is for `schemas.py` to read `KAFKA_BOOTSTRAP` from the environment and for the check
to accept a `--cluster` flag that uses the `*_cluster` values; the wrapper then becomes unnecessary.

## Running the pipeline on the cluster

The jobs hard-code `master("local[*]")` and `localhost:9092`, so don't launch them directly on the cluster. `run_job.py` replaces both
in memory and runs the unchanged script:

```bash
# on spark-master, from the shared repo, as the spark user
cd /srv/spark-shared/Anomaly-Detection
PY=/opt/anomaly-venv/bin/python
sudo -u spark $PY infra/cluster/run_job.py tools/check_spark_kafka.py                       # PASS = executors read telemetry.smd
sudo -u spark CLUSTER_CORES_MAX=2 $PY infra/cluster/run_job.py src/streaming/features_job.py
sudo -u spark CLUSTER_CORES_MAX=2 $PY infra/cluster/run_job.py src/streaming/score_job.py --input kafka
```

Checkpoints go to `$REPO_DIR/checkpoints/<job>/...` and Parquet to `$REPO_DIR/data/...`, both on the share.
One application takes all 4 cores (both executors) by default, so cap each job with `CLUSTER_CORES_MAX` (2 = one executor) when running two
at once. The budget above counts **one** driver (1 GB + 384 MB overhead); a second driver on the master adds about 1.4 GB and brings it to
roughly 5.9 GB of the 8 GB, which fits next to the single broker. (With the 3-broker layout the master is already at 6784 MB, so run one
driver at a time there, lower `SPARK_DRIVER_MEMORY`, or run the second driver from another VM; the driver must be reachable by the executors.)
Watch `health_check.sh memory`. The trace producer (`src/producer/`) runs outside Spark; give it the broker list `spark-master:9092`
through whatever bootstrap option it offers (the former `src/producer/replay.py` had `--bootstrap`; that file is not in the current working
tree, so check its `--help`).

`tools/check_spark_kafka.py` itself uses `local[2]`, so through `run_job.py` it checks that the cluster's executors can reach Kafka and that
the connector resolves.

## Configuration reference (`cluster.env`)

| Variable | Meaning |
| --- | --- |
| `KAFKA_HOST` | VM that runs the broker container(s) (default `spark-master`; must not be a worker) |
| `KAFKA_BROKER_COUNT` | number of brokers: 1 (default, RF 1) or 3 (RF 3, min ISR 2) |
| `SPARK_MASTER_HOST`, `SPARK_WORKER_HOSTS`, `NFS_SERVER_HOST` | topology (default: `spark-master`; `spark-w1 spark-w2`; `spark-master`) |
| `VM_PROFILE` | which file in `profiles/` supplies the VM-size settings (default `vm_8gb_2cpu`) |
| `SSH_USER`, `NFS_NETWORK` | admin ssh user (`spark` on Azure, created with `--admin-username spark`); CIDR allowed to mount the share (the VMs' subnet) |
| `KAFKA_CLUSTER_ID` | written by `gen_cluster_id.sh`, identical on all brokers; never change it after first start |
| `KAFKA_PORT_BASE` | 9092: broker n uses client port `BASE+100*(n-1)` and controller port `BASE+1+100*(n-1)` |
| `KAFKA_DATA_DIR`, `KAFKA_LOG_RETENTION_HOURS` | host data path (broker n in `broker-n/`), retention (24 h as on laptops) |
| `SHARED_DIR`, `REPO_DIR`, `SPARK_UID`, `SPARK_GID` | share path, repo location on it, fixed `spark` uid / gid on all VMs (1000 when `spark` is the Azure admin user; `00_prereqs.sh` fails if an existing `spark` user differs) |
| `PYTHON_BIN`, `VENV_DIR`, `INSTALL_ML` | Python 3.11 binary, local venv path (same on every Spark VM), install torch or not |

### VM profiles (`profiles/<VM_PROFILE>.env`)

Everything that depends on the size of the VM, rather than on your site, is in a profile file, so `cluster.env` stays short and
moving to a bigger VM is one line. Selected with `VM_PROFILE=` in `cluster.env`.

| Variable | `vm_8gb_2cpu` (default) | `vm_8gb_4cpu` | `vm_16gb_4cpu` | Meaning |
| --- | --- | --- | --- | --- |
| `VM_MEMORY` | 8g | 8g | 16g | RAM per VM; the budget check compares against it |
| `OS_RESERVE_MB`, `DRIVER_OVERHEAD_MB` | 1024, 384 | 1024, 384 | 1024, 384 | memory kept for the OS; driver off-heap + Python on top of `SPARK_DRIVER_MEMORY` |
| `KAFKA_HEAP`, `KAFKA_CONTAINER_MEMORY` | 1g, 1536m | 768m, 1280m | 1g, 1536m | JVM heap and Docker memory limit per broker |
| `SPARK_WORKER_CORES`, `SPARK_WORKER_MEMORY` | 2, 4g | 4, 4g | 4, 8g | per worker VM: what the executors may use |
| `SPARK_EXECUTOR_CORES` | 2 | 2 | 2 | `spark.executor.cores` (executors per worker = worker cores / executor cores) |
| `SPARK_EXECUTOR_MEMORY`, `SPARK_EXECUTOR_MEMORY_OVERHEAD` | 4g, 1536m | 2g, 768m | 4g, 1g | per executor |
| `SPARK_DAEMON_MEMORY`, `SPARK_DRIVER_MEMORY` | 512m, 1g | 512m, 1g | 512m, 2g | master / worker daemon heap; driver heap |
| Budget (worker / master, MB) | 7168 / 4480 (1 broker) | 7168 / 6784 (3 brokers) | 11776 / 8576 (3 brokers) | checked against `VM_MEMORY` by `lib.sh` |

Rules:

* A profile line is `: "${VAR:=default}"`, so it only fills in a variable that `cluster.env` (or your shell) has **not** set. To change one
  value, add `VAR=value` to `cluster.env`; you do not need to copy the profile.
* To support another VM size, copy a profile to `profiles/<name>.env`, change the numbers, and set `VM_PROFILE=<name>`. `deploy.sh env`
  prints the budget, and `deploy.sh` stops if a VM would not fit in `VM_MEMORY`.
* The master budget depends on `KAFKA_BROKER_COUNT`; the numbers above are for the broker count each profile was sized for.
* `vm_8gb_2cpu` and `vm_16gb_4cpu` are sizing guesses and, like everything here, untested.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `deploy.sh` stops at the memory check | a VM would exceed `VM_MEMORY`; lower `SPARK_EXECUTOR_MEMORY(_OVERHEAD)`, `KAFKA_CONTAINER_MEMORY` or `SPARK_DRIVER_MEMORY` |
| `deploy.sh kafka` times out on a client port | single broker: `sudo docker logs kafka-1` on `spark-master`, and check `spark-master` resolves to its own IP; three brokers: the quorum needs 2 of 3 nodes up |
| `cannot pull apache/kafka:3.7.0` | no outbound access or the Docker Hub anonymous rate limit; wait or `docker login` on `spark-master` |
| `kafka-N was OOM-killed` | the container hit `KAFKA_CONTAINER_MEMORY`; the VM is short of RAM (check drivers running next to the broker) |
| `cluster.id` mismatch in a broker log | the data dir was formatted by an earlier run with another id; wipe `$KAFKA_DATA_DIR/broker-N` only on a broker you are rebuilding |
| Switched `KAFKA_BROKER_COUNT` after the first start | the quorum voters and replication settings changed; wipe `$KAFKA_DATA_DIR` and recreate the topics (`deploy.sh kafka topics`), the data of the old cluster is not kept |
| Client connects, then hangs or times out | `spark-master` not resolvable from the client machine, or port 9092 (9192/9292) blocked |
| Fewer workers in the UI than in `SPARK_WORKER_HOSTS` | worker cannot reach `spark-master:7077`, or `hostname -s` differs from the name in `cluster.env` |
| Application stuck in `WAITING` | `spark.cores.max` or another application holds the cores; lower `CLUSTER_CORES_MAX` or stop it |
| Executors lost, or `Killed` in the worker logs | the Linux OOM killer; run `sudo dmesg -T` on the worker and look for "Killed process"; reduce parallel jobs |
| `Python in worker has different version` | `$VENV_DIR` differs between nodes; re-run `deploy.sh spark` |
| `No module named src` / `contracts` on executors | the share is not mounted at `$REPO_DIR` on that node, or the repo was not copied (`deploy.sh repo`) |
| Permission denied writing `checkpoints/` | `spark` uid / gid differ between VMs; they must match `SPARK_UID` / `SPARK_GID` |
