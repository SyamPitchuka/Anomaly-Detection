# Running the cluster on Azure VMs

Step-by-step runbook for building the cluster from `README.md` on Azure with what the subscription actually has:
**6 vCPUs of quota, so 3 VMs with 2 vCPU / 8 GiB each**, in the existing resource group `spark-rg` and virtual network `spark-vnet`,
in the region **India South Central** (the account is pinned to it).

| VM | Size | Runs |
| --- | --- | --- |
| `spark-master` | `Standard_D2as_v4` (2 vCPU / 8 GiB) | Spark master, **1 Kafka broker** (Docker), the NFS server, the Spark drivers, and the `deploy.sh` / admin work |
| `spark-w1` | `Standard_D2s_v4` (2 vCPU / 8 GiB) | one Spark worker: 2 cores, 1 executor of 2 cores |
| `spark-w2` | `Standard_D2s_v4` (2 vCPU / 8 GiB) | one Spark worker: 2 cores, 1 executor of 2 cores |

Spark totals: **2 executors, 4 cores**. Kafka: **1 broker, replication factor 1** (`contracts/topics.yaml` asks for 3; the topic tool caps it
at the number of brokers, see `README.md`). There is no separate admin VM, because 3 x 2 vCPUs already use the whole quota:
`spark-master` plays that role. Read `README.md` first for what the scripts do and for the memory budget; this file only covers the Azure side.

> **Untested.** The scripts have never run on real VMs, and neither has this runbook. Billing runs for as long as VMs are allocated;
> deallocate them when you stop working (section 9).

> **`spark-master` already exists.** It was created in the portal (admin user `spark`, uid 1000, ssh key `spark-master_key`) and is already set up:
> repo, SMD data and the AE / KMeans / PCA models in `~/Anomaly-Detection`, Python 3.11 (deadsnakes) and Docker. This runbook therefore
> creates only **`spark-w1` and `spark-w2`**, with the **same admin user `spark` and the same key**, so that the `spark` user has uid / gid 1000 on all
> three VMs (NFS ownership must match). Sections 3, 5 and 9 are written for that.
>
> The old standalone `kafka` container on `spark-master` (from the latency run) holds port 9092; `20_kafka_broker.sh` removes it before
> it starts `kafka-1`.

## Why `spark-master` is also the admin machine

The broker advertises `spark-master:9092`. Azure's built-in DNS resolves VM names only inside the same virtual network, so your laptop
cannot resolve `spark-master`. `deploy.sh`, `check_kafka_cluster.py`, the topic step and the producer must therefore run from a machine inside
`spark-vnet`. `spark-master` is that machine, and you reach it over ssh through its public IP.

```
laptop --ssh--> spark-master (public IP, in spark-vnet)  <- run deploy.sh and the python tools here
                    | ssh
        +-----------+-----------+
   spark-master             spark-w1, spark-w2
   (Kafka x1, Spark master,  (Spark workers)
    NFS, drivers)
```

All three VMs get a public IP, for two reasons: it gives them outbound internet (apt, Docker Hub, PyPI, Maven, Apache archive, deadsnakes PPA)
without changing the existing subnet, and it needs no NAT gateway. Inbound traffic is limited to ssh from your own address by a network security
group attached to the VM NICs, so the existing subnet is not touched.

## 0. Before you start

Work in a **bash** shell: Git Bash with the Azure CLI installed, or Azure Cloud Shell (bash). The commands below use bash syntax
(arrays, `$(...)`, `\` line continuations), not PowerShell.

```bash
az login
az account set --subscription "<your subscription name or id>"
```

### Region, sizes and quota

```bash
LOC=$(az group show -n spark-rg --query location -o tsv)     # the VMs must be in the region of spark-rg / spark-vnet
echo "$LOC"                                                  # expected: indiasouthcentral
az vm list-usage -l "$LOC" -o table | grep -Ei "Total Regional|standardDASv4Family|standardDSv4Family"
az vm list-skus -l "$LOC" --size Standard_D2as_v4 -o table     # "Restrictions" column must be empty
az vm list-skus -l "$LOC" --size Standard_D2s_v4 -o table
```

Needed: **2 vCPUs** in the `standardDASv4Family`, **4 vCPUs** in the `standardDSv4Family`, and **6** in `Total Regional vCPUs`, all with nothing
else running in those families. If a family quota is too low, use another 2 vCPU / 8 GiB size that you do have (for example `Standard_D2s_v5`
for all three); the profile `vm_8gb_2cpu` only depends on 2 vCPU / 8 GiB.

`$LOC` is read from `spark-rg` instead of being typed, so the VMs always land in the region of the existing group and VNet.

## 1. Variables and SSH key

The resource group and the virtual network already exist; they are not created here.

```bash
RG=spark-rg
VNET=spark-vnet
LOC=$(az group show -n "$RG" --query location -o tsv)   # region of the existing group (indiasouthcentral)
ADMIN_USER=spark   # same admin user as the existing spark-master (uid 1000)
IMAGE=Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest   # Ubuntu 22.04
KEY="$HOME/.ssh/spark-master_key" # private key of the existing spark-master (adjust the path / extension, e.g. .pem)

az group show -n "$RG" --query "{name:name,location:location}" -o table
az network vnet show -g "$RG" -n "$VNET" --query "{space:addressSpace.addressPrefixes,location:location}" -o json
az network vnet subnet list -g "$RG" --vnet-name "$VNET" --query "[].{name:name,prefix:addressPrefix,nat:natGateway.id,nsg:networkSecurityGroup.id}" -o table
```

Check that `spark-master` exists and use **its subnet** for the workers (it becomes `NFS_NETWORK` in section 6):

```bash
az vm show -d -g "$RG" -n spark-master --query "{state:powerState,size:hardwareProfile.vmSize,ip:publicIps,user:osProfile.adminUsername}" -o table
SUBNET=$(az vm show -g "$RG" -n spark-master --query "networkProfile.networkInterfaces[0].id" -o tsv \
  | xargs az network nic show --ids | python3 -c 'import json,sys; print(json.load(sys.stdin)["ipConfigurations"][0]["subnet"]["id"].split("/")[-1])')
SUBNET_CIDR=$(az network vnet subnet show -g "$RG" --vnet-name "$VNET" -n "$SUBNET" --query addressPrefix -o tsv)
echo "$SUBNET $SUBNET_CIDR"

# the workers get the public half of the SAME key as spark-master (the portal gave you only the private key)
chmod 600 "$KEY"
ssh-keygen -y -f "$KEY" > "$KEY.pub"
```

The `user` column must show `spark`. If `python3` is not available in your shell, read the subnet name from the table above and set `SUBNET` by hand.

Ubuntu 22.04 is used because the scripts install Python 3.11 from the deadsnakes PPA, which supports it.

If the subnet already has free addresses but a **custom DNS** is configured on `spark-vnet`, check in section 4 that `getent hosts spark-w1` works;
VM names must resolve between the three VMs.

## 2. Network security group for the workers

Only ssh from your address is allowed in; traffic between the VMs inside the VNet is allowed by Azure's default `AllowVnetInBound` rule.
`spark-master` keeps the NSG it already has; this one is attached to the new worker NICs only.

```bash
MYIP=$(curl -s https://api.ipify.org)              # SSH will be allowed from this address only
az network nsg show -g "$RG" -n spark-nsg --query name -o tsv 2>/dev/null \
  || az network nsg create -g "$RG" -l "$LOC" -n spark-nsg
az network nsg rule create -g "$RG" --nsg-name spark-nsg -n ssh-from-me --priority 100 \
  --access Allow --direction Inbound --protocol Tcp --destination-port-ranges 22 \
  --source-address-prefixes "$MYIP/32"
```

If your address changes later, update the rule: `az network nsg rule update -g "$RG" --nsg-name spark-nsg -n ssh-from-me --source-address-prefixes "<new ip>/32"`.
You do not need to open Kafka, Spark or NFS ports.

## 3. Create the worker VMs

**`spark-master` already exists; do not create it again.** Create only `spark-w1` and `spark-w2`. The VM names **must** match the hostnames in
`cluster.env.example`, because the scripts compare them with `hostname -s`.

`--admin-username spark` makes the Azure admin account `spark` (uid / gid 1000, passwordless sudo), the same as on `spark-master`, so
`00_prereqs.sh` finds an existing `spark` user and the NFS ownership is identical on all VMs.

```bash
az vm show -g "$RG" -n spark-master --query name -o tsv || { echo "spark-master is missing"; false; }   # must already exist

common=(--resource-group "$RG" --location "$LOC" --image "$IMAGE" --admin-username "$ADMIN_USER"
        --authentication-type ssh --ssh-key-values "$KEY.pub"
        --vnet-name "$VNET" --subnet "$SUBNET" --nsg spark-nsg --public-ip-sku Standard
        --os-disk-size-gb 64 --storage-sku Premium_LRS
        --os-disk-delete-option Delete --nic-delete-option Delete --no-wait)

az vm create "${common[@]}" --name spark-w1 --size Standard_D2s_v4 --public-ip-address spark-w1-pip
az vm create "${common[@]}" --name spark-w2 --size Standard_D2s_v4 --public-ip-address spark-w2-pip

for n in spark-w1 spark-w2; do
  az vm wait -g "$RG" -n "$n" --created
done
az vm list -g "$RG" -d -o table                     # the three VMs: "VM running" (spark-rg may hold other resources too)
```

Kafka data goes to `/var/lib/kafka/data` on the 64 GB OS disk of `spark-master`, together with the NFS share
(`/srv/spark-shared`, which holds the repo, checkpoints and Parquet output). That is enough for a trial with 24 h retention. For a long run
or the large Alibaba trace, attach a managed data disk to `spark-master` and mount it for `KAFKA_DATA_DIR` and `SHARED_DIR` before section 7.
The broker, the NFS server and the drivers share this one VM and disk, so Kafka throughput is limited by it.

## 4. Prepare `spark-master`

```bash
MASTER_IP=$(az vm show -d -g "$RG" -n spark-master --query publicIps -o tsv)

# spark-master needs the private key to reach itself and the two workers over ssh
# (skip these two lines if ~/.ssh/id_ed25519 / id_rsa is already the spark-master_key on spark-master)
scp -i "$KEY" "$KEY" "$ADMIN_USER@$MASTER_IP:.ssh/id_ed25519"
ssh -i "$KEY" "$ADMIN_USER@$MASTER_IP" "chmod 600 ~/.ssh/id_ed25519"

ssh -i "$KEY" "$ADMIN_USER@$MASTER_IP"               # you are now on spark-master
```

On `spark-master`:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip netcat-openbsd curl tar

# name resolution and ssh to every VM (first connection accepts the host key)
getent hosts spark-master spark-w1 spark-w2
for h in spark-master spark-w1 spark-w2; do
  ssh -o StrictHostKeyChecking=accept-new "$h" 'hostname -s && sudo -n true && echo sudo-ok && free -m | head -2'
done
```

Every host must resolve and print its own short name, `sudo-ok` and about 7.6-7.8 GB of total memory. If a name does not resolve, the VM name
or the VNet DNS is wrong. Do not continue until this works.

## 5. Copy the repo to `spark-master`: already done, skip

`spark-master` already has the repo, the SMD data and the AE / KMeans / PCA models in `~/Anomaly-Detection`, plus Python 3.11.18 (deadsnakes)
and Docker. Nothing to copy or install. Optional quick check on `spark-master`:

```bash
cd ~/Anomaly-Detection
ls models data | head; python3.11 --version; docker --version
sed -i 's/\r$//' infra/cluster/*.sh infra/cluster/cluster.env.example           # harmless if there is no CRLF
for f in infra/cluster/*.sh; do bash -n "$f" && echo "syntax ok: $f"; done
```

Continue with section 6. The rest of this section is kept only for building a new master from scratch.

<details><summary>Copy the repo to a new master (not needed here)</summary>

`deploy.sh repo` copies the repo from the machine that runs it (`spark-master`, home directory) to the NFS share, so `spark-master` needs a full
working copy. Pick one.

**a) From your laptop (Git Bash; not PowerShell, which corrupts binary pipes):**

```bash
cd "<path to the parent of Anomaly-Detection>"
tar --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache -cf - Anomaly-Detection \
  | ssh -i "$KEY" "$ADMIN_USER@$MASTER_IP" "tar -xf -"
```

This copies `data/` and `models/` too. `score_job.py` needs the trained models under `models/`, so keep them; add
`--exclude=Anomaly-Detection/data` if `data/` is large, and sync the parts you need later.

**b) From git**, on `spark-master`: `git clone <your repo url> Anomaly-Detection`. Models and data are git-ignored, so copy
`models/` (and `data/raw` for the producer) separately with the `tar` pipe above.

Then, on `spark-master`:

```bash
cd ~/Anomaly-Detection
sed -i 's/\r$//' infra/cluster/*.sh infra/cluster/cluster.env.example     # strip CRLF if the files came from Windows
python3 -m venv .venv && . .venv/bin/activate
pip install --quiet -r requirements.txt                                    # a few minutes; the topic step needs confluent-kafka, pyarrow, pyyaml
python -m pytest tests/test_cluster_tools.py -q                            # local sanity check, no cluster needed
for f in infra/cluster/*.sh; do bash -n "$f" && echo "syntax ok: $f"; done
```

This `~/Anomaly-Detection/.venv` is only for the client tools (Ubuntu's Python 3.10 is fine). The Spark VMs get Python 3.11 at
`/opt/anomaly-venv` from the deploy scripts.

</details>

## 6. Configure the cluster

```bash
cd ~/Anomaly-Detection
cp infra/cluster/cluster.env.example infra/cluster/cluster.env
# NFS_NETWORK = the CIDR of the subnet this VM is in, read from its own route table (the same subnet as spark-w1 and spark-w2)
SUBNET_CIDR=$(ip -4 route show dev eth0 proto kernel scope link | awk 'NR==1 {print $1}'); echo "$SUBNET_CIDR"
sed -i "s|^NFS_NETWORK=.*|NFS_NETWORK=${SUBNET_CIDR}|" infra/cluster/cluster.env
grep -E '^(VM_PROFILE|KAFKA_HOST|KAFKA_BROKER_COUNT|SPARK_MASTER_HOST|SPARK_WORKER_HOSTS|NFS_SERVER_HOST|SSH_USER|SPARK_UID|SPARK_GID|NFS_NETWORK)=' infra/cluster/cluster.env
id spark                                                                   # on every VM: uid=1000(spark) gid=1000(spark), matching SPARK_UID / SPARK_GID
bash infra/cluster/gen_cluster_id.sh                                       # writes KAFKA_CLUSTER_ID once
```

The example already matches this runbook: `VM_PROFILE=vm_8gb_2cpu`, `KAFKA_BROKER_COUNT=1`, `SPARK_WORKER_HOSTS=(spark-w1 spark-w2)`,
`KAFKA_HOST=spark-master`, `SSH_USER=spark` (the Azure admin account of all three VMs, with passwordless sudo), `SPARK_UID=1000`,
`SPARK_GID=1000`. Only `NFS_NETWORK` needs your subnet. `00_prereqs.sh` stops with an error if the `spark` user on a VM does not have
`SPARK_UID` / `SPARK_GID`.

## 7. Deploy

Wait about 5 minutes after the VMs were created. On a fresh Azure Ubuntu image, `unattended-upgrades` can hold the apt lock and make
`00_prereqs.sh` fail; if that happens just re-run the phase.

```bash
cd ~/Anomaly-Detection
sudo docker ps -a --filter name=kafka       # the old standalone 'kafka' container (port 9092) is removed by 20_kafka_broker.sh; or: sudo docker rm -f kafka
bash infra/cluster/deploy.sh ssh            # ssh + passwordless sudo on all three VMs
bash infra/cluster/deploy.sh env            # prints the memory budget per VM; stops if a VM would not fit
bash infra/cluster/deploy.sh all            # about 15-30 minutes; torch and pyspark are installed on three VMs
```

To find a failure faster, run the phases one at a time:
`bash infra/cluster/deploy.sh env prereqs nfs repo kafka spark topics health`. Every phase can be re-run safely.

`all` finishes with the health check and prints `PASS` or `FAIL`.

## 8. Verify and run

On `spark-master`, with the client venv active (`. .venv/bin/activate`), from `~/Anomaly-Detection`:

```bash
bash infra/cluster/health_check.sh                      # expect PASS (container, Kafka quorum, Spark workers, NFS, memory)

python tools/fake_data.py --layers jsonl                # only data/sample/*.jsonl; do not write fake Parquet layers
python infra/cluster/check_kafka_cluster.py             # tools/check_kafka.py against the broker, expect PASS
```

Use `--layers jsonl` on purpose. Without it `fake_data.py` also writes `fake.parquet` into `data/raw` and the other layers, and a later
`deploy.sh repo` would copy those fake files onto the share next to your real data.

Then the Spark check, still on `spark-master`:

```bash
cd /srv/spark-shared/Anomaly-Detection
sudo -u spark /opt/anomaly-venv/bin/python infra/cluster/check_spark_cluster.py      # expect PASS (2 executors, 1 per worker)
sudo -u spark /opt/anomaly-venv/bin/python infra/cluster/run_job.py tools/check_spark_kafka.py   # PASS needs records in telemetry.smd
```

`check_spark_kafka.py` only passes after something has been produced to `telemetry.smd`. Run `check_kafka_cluster.py` first (it produces the
sample), or start the producer.

### Run the pipeline

Producer, from `~/Anomaly-Detection` with the client venv active (check `python src/producer/replay.py --help` first; the file may not be
in your working tree):

```bash
python src/producer/replay.py --bootstrap spark-master:9092
```

Streaming jobs, on `spark-master` as the `spark` user. One application takes both executors (4 cores) unless capped. To run the features and
the score job at the same time, cap each at one executor (`CLUSTER_CORES_MAX=2`); with 2 cores each they will be slow, so for a first test run
one job at a time:

```bash
cd /srv/spark-shared/Anomaly-Detection
PY=/opt/anomaly-venv/bin/python
sudo -u spark CLUSTER_CORES_MAX=2 $PY infra/cluster/run_job.py src/streaming/features_job.py --once      # one-shot test
sudo -u spark CLUSTER_CORES_MAX=2 $PY infra/cluster/run_job.py src/streaming/score_job.py --input kafka --once
```

Drop `--once` for continuous streaming and run the job under `tmux` or `nohup` so it survives your ssh session.
Checkpoints land in `/srv/spark-shared/Anomaly-Detection/checkpoints/` and Parquet in `.../data/`.
The master has room for a second driver next to the single broker (see the budget in `README.md`).

After changing code on `spark-master`, run `bash infra/cluster/deploy.sh repo` to push it to the share (it never touches `checkpoints/`).

### Web UIs

Nothing is exposed publicly. Tunnel through `spark-master` from your laptop:

```bash
ssh -i "$KEY" -L 8080:spark-master:8080 -L 4040:spark-master:4040 "$ADMIN_USER@$MASTER_IP"
# then open http://localhost:8080 (master, 2 workers) and, while a job runs, http://localhost:4040
```

## 9. Stop, restart, delete

`spark-rg` and `spark-vnet` are shared and already existed, so **do not delete the resource group**. Deallocate to stop paying for compute,
or delete only the resources created here.

> **WARNING: `spark-master` was NOT created by this runbook** (portal-created, holds the repo, data, models and the Kafka data). The delete
> commands below remove **the workers only**. Never delete `spark-master`, its public IP or the key `spark-master_key` as part of this cleanup.

```bash
# stop paying for compute (disks and the public IPs still cost a little)
az vm deallocate -g "$RG" --no-wait --ids $(az vm show -g "$RG" -n spark-master --query id -o tsv) \
  $(az vm show -g "$RG" -n spark-w1 --query id -o tsv) $(az vm show -g "$RG" -n spark-w2 --query id -o tsv)

# start again: master first (it is the NFS server and runs the Kafka container), then the workers
az vm start -g "$RG" -n spark-master
az vm start -g "$RG" --no-wait -n spark-w1
az vm start -g "$RG" --no-wait -n spark-w2
# then on spark-master:  bash infra/cluster/health_check.sh
# the Kafka container uses --restart unless-stopped, so Docker brings it back with the VM

# remove only what this runbook created: the two workers (OS disks and NICs go with the VMs because of the *-delete-option flags)
for n in spark-w1 spark-w2; do az vm delete -g "$RG" -n "$n" --yes; done
for n in spark-w1 spark-w2; do az network public-ip delete -g "$RG" -n "$n-pip"; done
az network nsg delete -g "$RG" -n spark-nsg          # only if you created it in section 2 and nothing else (e.g. spark-master) uses it
```

The public IPs are Standard (static), so their addresses stay the same after a deallocate; the ssh rule only needs updating if your own address changes.
Keep the key `spark-master_key`: it is also the login key of `spark-master`. The workers are gone after the delete; `spark-master` keeps
running (and billing) until you deallocate it as above.

## Azure-specific troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `getent hosts spark-w1` fails on `spark-master` | VM names are wrong, the VMs are in different VNets, or `spark-vnet` has a custom DNS server that does not know the VM names |
| `apt-get`, `pip` or `docker pull` hangs | no outbound access; the VM has no public IP and the subnet has no NAT gateway (section 3 gives each VM a public IP) |
| `Could not get lock /var/lib/dpkg/lock-frontend` | `unattended-upgrades` is running on a new VM; wait a few minutes and re-run the phase |
| `SkuNotAvailable` / `QuotaExceeded` on `az vm create` | the size or the family quota is missing in `$LOC`; check section 0, use another 2 vCPU / 8 GiB size, or raise the quota. Remember that 6 vCPUs are fully used by these three VMs |
| `LocationNotAvailableForResourceType`, or the VM lands in another region | `$LOC` must be the account's region and the same as `spark-vnet`'s; `az network vnet show -g spark-rg -n spark-vnet --query location` |
| `deploy.sh env` says a VM needs more than `VM_MEMORY` | the VM is smaller than the profile expects, or you overrode memory settings in `cluster.env`; pick the matching `VM_PROFILE` (see `README.md`) |
| `bash\r: No such file or directory` | CRLF line endings; run the `sed -i 's/\r$//'` line from section 5 |
| Workers show the share unmounted after a restart | workers booted before the master; on each worker run `sudo mount -a && sudo systemctl restart spark-worker` |
| `ssh: Permission denied (publickey)` from `spark-master` | private key not copied or wrong mode; redo the `scp` and `chmod 600` in section 4 |
| ssh from the laptop times out | your public address changed; update the `ssh-from-me` rule (section 2) |
| Client connects to Kafka, then times out | the client is outside the VNet and cannot resolve `spark-master`; run it from `spark-master` |

Everything else (Kafka quorum, Spark registration, NFS permissions, OOM kills, `cluster.id` mismatch) is covered by the troubleshooting table
in `README.md`.
