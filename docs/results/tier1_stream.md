# Tier 1 in the stream: parity, Tier 1 on/off, and the v2 grid

Measured 2026-11-02 on all 28 SMD machines (141,708 test windows of 5 minutes, 326 labelled faults).
The AE is AE v2 (`models/ae_smd`, recentre 24, adaptive threshold). Tier 1 v1 is the robust z-score from
`src/detection/tier1_zscore.py`: z = |window mean − train median| / (1.4826 × train MAD), max over metrics,
flag if z > 3, metrics with MAD = 0 skipped (303 of 1,064 machine-metrics).

## Decision

`is_anomaly` stays AE only (`is_anomaly = ae_flag`). Tier 1 is kept as a baseline and an explainer: the stream
fills `tier1_score` / `tier1_flag` with `--tier1 v1` and leaves them null by default (`--tier1 off`).
Neither Tier 1 v1 (215 false alarms per machine per day) nor any Tier 1 v2 setting (best 9.4) met the
pre-registered limit of 2 false alarms per machine per day, and adding either to the AE lowers composite F1.

## Stream vs batch parity (Tier 1 v1 + AE)

`python src/streaming/score_job.py --input files --once --reset --tier1 v1`, then
`python tools/check_stream_parity.py` (AE: PASS) and `python experiments/tier1_compare.py`.

| quantity | value |
| --- | ---: |
| matched windows (stream vs batch) | 141,624 of 141,708 |
| missing in stream (last 15 min of each machine, still open in the state) | 84 |
| duplicates in stream | 0 |
| max \|tier1_score diff\| | 5.8e-11 |
| max \|ae_score diff\| | 4.1e-06 |
| tier1_flag identical | 100.00 % |
| ae_flag identical | 100.00 % |
| stream `is_anomaly` == `ae_flag` | yes, every window |
| windows tier1_flag adds on top of ae_flag | 110,516 (5,264 labelled anomalous, 105,252 normal) |

## Tier 1 v1 on vs off (stream, identical in batch)

Composite F1 = harmonic mean of event recall (faults with at least one flagged window) and alarm precision
(flagged windows that are anomalous); false alarms per machine per day count flagged normal windows; median
delay is over detected faults, in minutes from the first anomalous window (all from `src/detection/evaluate.py`,
scored as in `src/detection/autoencoder.py`).

| alarm | composite F1 | faults caught | alarm precision | false alarms / machine / day | median delay (min) | flagged windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tier 1 off: `ae_flag` | **0.525** | 180/326 | 0.501 | **1.18** | 0 | 1,158 |
| Tier 1 on: `ae_flag OR tier1_flag` | 0.099 | 323/326 | 0.052 | 215.2 | 0 | 111,674 |

Tier 1 v1 flags 79 % of the test windows: SMD machines drift away from their train median during the test
period, and a single metric beyond 3 robust standard deviations is enough to flag.

## Tier 1 v2 grid (batch only)

`python experiments/tier1_v2_grid.py`. Changes from v1, all at once:

- **Recentre:** z against a causal rolling median of the previous 24 windows (at least 12, current excluded,
  else the train median), the same as AE `--recenter 24`.
- **MAD floor:** mad = max(mad, 0.01 × |median| + 1e-3), so near-constant metrics cannot explode
  (and are no longer skipped).
- **Agreement:** flag only if at least k metrics have z > t in the same window, k ∈ {2, 3}, t ∈ {4, 6}.
- **Persistence:** 2 consecutive flagged windows (the flag is set on the second, so it costs one window of delay).

**Selection rule, fixed before the results were seen:** among the settings with Tier 1 v2 false alarms
≤ 2 per machine per day, take the highest composite F1; ties go to fewer false alarms. The rule is applied
to Tier 1 v2 alone; the `AE OR v2` columns are for reference.

| k | t | v2 composite F1 | v2 faults caught | v2 alarm precision | v2 false alarms / machine / day | v2 median delay (min) | windows added on top of AE | AE OR v2 composite F1 | AE OR v2 false alarms / machine / day | AE OR v2 median delay (min) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.266 | 194/326 | 0.171 | 28.1 | 5 | 15,955 | 0.286 | 28.7 | 0 |
| 2 | 6 | 0.316 | 166/326 | 0.229 | 15.2 | 5 | 9,038 | 0.354 | 15.8 | 0 |
| 3 | 4 | 0.291 | 172/326 | 0.201 | 17.8 | 5 | 10,357 | 0.324 | 18.5 | 0 |
| 3 | 6 | 0.338 | 150/326 | 0.267 | 9.4 | 5 | 5,743 | 0.396 | 10.1 | 0 |
| AE only (reference) | | | | | | | | 0.525 | 1.18 | 0 |

**Result: no setting meets the rule**, so no Tier 1 v2 setting is selected and there is no AE vs AE OR v2
comparison beyond the reference row. v2 cuts false alarms from 215 to 9–28 per machine per day, but even the
best setting would lower the combined composite F1 from 0.525 to 0.396. The grid was not widened after
the results were seen; a further grid (larger k or t, longer persistence) would need its own rule fixed first.

## Reproduce

```
python src/detection/tier1_zscore.py          # baselines + data/scores_tier1
# rename models/tier1_smd/baselines.parquet to baselines_v1.parquet (the file --tier1 v1 reads)
python src/streaming/score_job.py --input files --once --reset --tier1 v1
python tools/check_stream_parity.py
python experiments/tier1_compare.py
python experiments/tier1_v2_grid.py
```

Hardware: i5-13420H, 4 P-cores + 4 E-cores (12 threads), 16 GB RAM, Windows 11, Spark local[*],
one math thread per Python worker.
