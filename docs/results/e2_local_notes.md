**Where scaling flattens and why.** With the thread fix, throughput no longer falls as cores are added,
but it barely rises either: about 1.1x at 2 cores and 1.13x at 4 (efficiency 0.55 and 0.28), and at 8 cores
the run was stopped for memory before its first micro-batch finished. Three things cap it:
(1) **shuffle and keys.** Each micro-batch holds only 8 machines, i.e. 8 keys, hashed into 2n shuffle partitions, so
at most 8 tasks have work, some partitions get 2–3 machines while others get none, and the batch waits for
the busiest one, while every extra partition adds a state-store checkpoint per batch;
(2) **Python worker overhead.** Every core runs its own Python worker that loads PyTorch and the models (about
0.3–0.4 GB each: peak memory grows from 1.5 GB at 1 core to 2.6 GB at 4, and reached 3.85 GB at 8 before the stop), and every row
crosses the JVM↔Python Arrow boundary, which costs the same at any core count;
(3) **single-disk I/O.** Reading the raw Parquet, the state checkpoints and the score files all go to one laptop
SSD, which more cores cannot speed up.

**Caveats.** Run-to-run noise on this laptop is as large as the effect being measured: the same 1-core
setting gave 4,702 and 7,010 records/s, mostly because free RAM was only 3–5 GB with other apps open. The first
micro-batch of each run also includes model loading and JVM warm-up. Read the speed-ups as "flat
within noise", not as precise values. The second 8-core run did not complete: the host stopped the whole
experiment for low memory during that run. The cluster version of E2, with one worker per VM and more memory
per worker, replaces these numbers.
