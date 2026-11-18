# E6: CAP behaviour of the Cassandra serving store

Nodes spark-w2, spark-w5; keyspace `anomaly` (SimpleStrategy, RF 2); table `e6_cap`; 1000 rows written and read back per consistency level and phase; one synchronous statement per op, latency measured on the client (spark-master). Victim: spark-w5. Started 2026-11-18T00:04:02Z. Produced by `experiments/e6_cap.py`.

| phase | CL | replicas needed | write ok | write p50 ms | write p95 ms | read ok | read p50 ms | read p95 ms | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both_up (spark-w2+spark-w5 up) | ONE | 1 | 1000/1000 (100.0%) | 1.376 | 2.506 | 1000/1000 (100.0%) | 1.463 | 2.582 | - |
| both_up (spark-w2+spark-w5 up) | QUORUM | 2 | 1000/1000 (100.0%) | 2.1 | 3.347 | 1000/1000 (100.0%) | 2.361 | 3.47 | - |
| both_up (spark-w2+spark-w5 up) | ALL | 2 | 1000/1000 (100.0%) | 1.975 | 2.794 | 1000/1000 (100.0%) | 2.152 | 3.068 | - |
| one_down (spark-w2 up) | ONE | 1 | 1000/1000 (100.0%) | 1.138 | 2.128 | 1000/1000 (100.0%) | 1.277 | 2.141 | - |
| one_down (spark-w2 up) | QUORUM | 2 | 0/1000 (0.0%) | - | - | 0/1000 (0.0%) | - | - | write NoHostAvailable 1000, read NoHostAvailable 1000 |
| one_down (spark-w2 up) | ALL | 2 | 0/1000 (0.0%) | - | - | 0/1000 (0.0%) | - | - | write NoHostAvailable 1000, read NoHostAvailable 1000 |

## Recovery

- stop of spark-w5 seen as DN after 7.7 s; restart to UN in 20.4 s
- pending hints on spark-w2 before the restart: 0 host(s); delivered: yes (waited 5.6 s)
- rows written successfully in total (expected in every replica): 4000
- spark-w5 alone after hint delivery: 4000 rows
- `nodetool repair -full anomaly e6_cap` on spark-w5: ok in 2.5 s
- each replica alone after repair: spark-w2 4000, spark-w5 4000
- COUNT(*) at CL ALL after repair: 4000
- converged: **yes**

## CAP interpretation

- **ONE** (needs 1 of 2 replicas) stayed available with one node down (1000/1000 writes, 1000/1000 reads): it chooses **A over C** (AP). The down replica misses those writes until hints or repair copy them, so a read at ONE from it in that time can be stale (as expected).
- **QUORUM** (needs 2 of 2 replicas) refused every op with one node down (0 writes, 0 reads succeeded; errors {'NoHostAvailable': 1000}): it chooses **C over A** (CP), never acknowledging a write that not every required replica holds (as expected).
- **ALL** (needs 2 of 2 replicas) refused every op with one node down (0 writes, 0 reads succeeded; errors {'NoHostAvailable': 1000}): it chooses **C over A** (CP), never acknowledging a write that not every required replica holds (as expected).
- With RF 2, QUORUM = floor(2/2)+1 = 2 replicas = ALL: on two nodes a quorum cannot lose a node, so QUORUM behaves exactly like ALL. Reads and writes at QUORUM overlap (R + W = 4 > RF 2), so they are strongly consistent; ONE + ONE (2, not > 2) is not. RF 3 on 3 nodes would give a QUORUM (2 of 3) that is both consistent and survives one node.
- Recovery: the survivor held hints for 0 host(s) while spark-w5 was down; after the restart they were delivered in 5.6 s (victim alone then held 4000 of 4000 rows); after nodetool repair the replicas converged (4000 rows expected). That is the eventual consistency side of AP: availability during the partition, convergence after it.
- A stopped node is what a network partition looks like to the coordinator (no gossip, no replies), so these runs stand for a partition of one node; a real partition would also leave clients on the other side, which a 2-node test cannot show.
