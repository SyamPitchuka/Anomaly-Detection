# E5b – autoencoder vs Tier 1 on injected Alibaba faults (incl. correlation breaks)

Second dataset, no labels: faults are injected into real Alibaba v2018 machine metrics (as in [E5](e5.md)), with a fourth, multivariate fault type. Script: `experiments/e5b_alibaba_ae.py`; run 18 Nov 2026, 12.8 min on one laptop (CPU).

## Setup

- Data: machine_usage of machines m_1..m_40 (raw layer from `src/producer/alibaba_to_parquet.py`), 8 days.
- AE: `src/detection/autoencoder.py --source alibaba --train-from-hours 72`, 7 metrics x (mean, max) = 14 inputs, 14-32-8-32-14, production v2 settings (recenter 24, adaptive 288, global threshold = p99.5 of the train score x 1.1, seed 42). Trained on hours 72–192 only: 57600 windows of 40 machines, 47 epochs (best 42), threshold 0.1291. The test slice (hours 0–48) is never seen in training; hours 48–72 are a gap.
- On the uninjected hours 0–72 the AE raises 1.53 false alarms per machine per day (140 of 26303 windows; includes the first 12 h, where the adaptive limit has no history yet).
- Faults: `src/producer/inject_faults.py --types spike drift break corrbreak`, 40 machines, first 48 h, clean first 24 h, 10 per type (one per machine), seeds 7, 8, 9. spike/drift/break exactly as E5 (same machines, metrics and times per seed: corrbreak is planned last, on the 10 machines E5 left fault-free).
- **corrbreak**: for 15 minutes one metric with a correlated partner (|r| >= 0.5 of the clean 5-minute means; else the most correlated one) is replaced by the same metric's values from a random clean 15-minute stretch of that machine, whose mean lies between the clean p10 and p90 and on the other side of the clean median than the replaced stretch. Every value stays inside the machine's clean min–max.
- Both detectors see the replayed messages (`replay.py` into a capture producer). Tier 1 = E5's robust z-score (5-minute means, median/MAD of the clean 24 h). AE = the saved model on the whole 48 h (first 24 h = recenter/adaptive history). Evaluation with `inject_faults.report` (`src/detection/evaluate.py`) on windows after the clean part: recall = share of faults with a flagged window, delay = fault start to first flagged window, false alarms per machine per day (all 40 machines for 'all', the type's 10 machines in the per-type rows).
- Tier 1 is shown at threshold 20 (E5) and at a matched budget: the threshold (grid 3–200) whose all-machine false alarms per day are closest to the main AE row's, chosen per seed.
- Main result = AE with the production global threshold. Secondary, pre-specified row = the same scores with the per-machine thresholds saved at training time (p99.5 x 1.1 of each machine's own train-hour scores, `--threshold per-machine` of autoencoder.py) and the same adaptive limit. Both thresholds come from the training hours only; nothing is tuned on the injected slice or its labels.

## Mean over seeds (min–max)

| detector | false alarms / machine / day | spike | drift | break | corrbreak | all |
| --- | --- | --- | --- | --- | --- | --- |
| AE (Alibaba, v2), global threshold [main] | 1.37 (1.29–1.45) | 0.63 (0.40–0.80) | 0.60 (0.50–0.70) | 1.00 | 0.07 (0.00–0.10) | 0.57 (0.55–0.60), 69/120 |
| Tier 1, matched false alarms (thr 16.5) | 1.31 (1.29–1.33) | 0.63 (0.50–0.70) | 0.47 (0.30–0.70) | 1.00 | 0.00 | 0.53 (0.50–0.55), 63/120 |
| Tier 1, threshold 20 (E5) | 0.65 (0.62–0.66) | 0.53 (0.40–0.70) | 0.40 (0.20–0.60) | 1.00 | 0.00 | 0.48 (0.47–0.50), 58/120 |
| AE, per-machine train-only threshold [secondary] | 1.26 (1.20–1.33) | 0.63 (0.40–0.80) | 0.60 (0.50–0.70) | 1.00 | 0.10 (0.00–0.20) | 0.58 (0.55–0.62), 70/120 |

Recall per fault type. Mean detection delay in minutes (detected faults only):

| detector | spike | drift | break | corrbreak |
| --- | --- | --- | --- | --- |
| AE (Alibaba, v2), global threshold [main] | 0.0 | 4.6 (3.3–7.0) | 0.0 | 2.5 (0.0–5.0) |
| Tier 1, matched false alarms | 0.0 | 14.0 (6.2–20.0) | 0.0 | – |
| Tier 1, threshold 20 (E5) | 0.0 | 13.9 (10.0–17.5) | 0.0 | – |
| AE, per-machine train-only threshold [secondary] | 0.0 | 3.7 (2.9–5.0) | 0.0 | 2.5 (0.0–5.0) |

## Per seed

### Seed 7 (Tier 1 matched threshold 16.5)

| detector | FA / machine / day | spike | drift | break | corrbreak | all | delay spike / drift / break / corrbreak (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE (Alibaba, v2), global threshold [main] | 1.45 | 7/10 | 6/10 | 10/10 | 1/10 | 0.60 | 0.0 / 3.3 / 0.0 / 0.0 |
| Tier 1, matched false alarms | 1.33 | 7/10 | 4/10 | 10/10 | 0/10 | 0.53 | 0.0 / 6.2 / 0.0 / – |
| Tier 1, threshold 20 (E5) | 0.66 | 5/10 | 4/10 | 10/10 | 0/10 | 0.47 | 0.0 / 10.0 / 0.0 / – |
| AE, per-machine train-only threshold [secondary] | 1.33 | 7/10 | 6/10 | 10/10 | 2/10 | 0.62 | 0.0 / 3.3 / 0.0 / 0.0 |

### Seed 8 (Tier 1 matched threshold 16.5)

| detector | FA / machine / day | spike | drift | break | corrbreak | all | delay spike / drift / break / corrbreak (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE (Alibaba, v2), global threshold [main] | 1.37 | 4/10 | 7/10 | 10/10 | 1/10 | 0.55 | 0.0 / 3.6 / 0.0 / 5.0 |
| Tier 1, matched false alarms | 1.33 | 5/10 | 7/10 | 10/10 | 0/10 | 0.55 | 0.0 / 15.7 / 0.0 / – |
| Tier 1, threshold 20 (E5) | 0.66 | 4/10 | 6/10 | 10/10 | 0/10 | 0.50 | 0.0 / 14.2 / 0.0 / – |
| AE, per-machine train-only threshold [secondary] | 1.24 | 4/10 | 7/10 | 10/10 | 1/10 | 0.55 | 0.0 / 2.9 / 0.0 / 5.0 |

### Seed 9 (Tier 1 matched threshold 16.5)

| detector | FA / machine / day | spike | drift | break | corrbreak | all | delay spike / drift / break / corrbreak (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE (Alibaba, v2), global threshold [main] | 1.29 | 8/10 | 5/10 | 10/10 | 0/10 | 0.57 | 0.0 / 7.0 / 0.0 / – |
| Tier 1, matched false alarms | 1.29 | 7/10 | 3/10 | 10/10 | 0/10 | 0.50 | 0.0 / 20.0 / 0.0 / – |
| Tier 1, threshold 20 (E5) | 0.62 | 7/10 | 2/10 | 10/10 | 0/10 | 0.47 | 0.0 / 17.5 / 0.0 / – |
| AE, per-machine train-only threshold [secondary] | 1.20 | 8/10 | 5/10 | 10/10 | 0/10 | 0.57 | 0.0 / 5.0 / 0.0 / – |

## Score separation (does the AE react without alarming?)

Per fault: peak = max ae_score over the fault's windows; normal = the machine's median ae_score over the clean first 24 h of the slice (no labels used). All seeds pooled; thresholds compared to the peak before the adaptive limit (global threshold 0.1291).

| fault type | faults | median peak / normal (IQR) | peak > 5x normal | median peak | median normal | peak > global thr | peak > own train-only thr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spike | 30 | 402.0 (18.9–31546.2) | 24/30 | 0.4979 | 0.0021 | 19/30 | 19/30 |
| drift | 30 | 872.2 (5.8–2866.8) | 24/30 | 1.9609 | 0.0023 | 18/30 | 18/30 |
| break | 30 | 7797.1 (6945.3–9227.6) | 30/30 | 17.5784 | 0.0022 | 30/30 | 30/30 |
| corrbreak | 30 | 15.9 (6.1–31.4) | 24/30 | 0.0353 | 0.0020 | 3/30 | 4/30 |

AE attribution (all seeds, main row): the injected metric is among `top_metrics` of the first flagged window in spike 19/19, drift 18/18, break 30/30, corrbreak 2/2 detected faults.

## Interpretation

- **Main result: no.** With its production global threshold the AE alarms on only 2 of 30 correlation breaks (seeds 7, 8, 9: 1, 1, 0). Tier 1 alarms on none (0/30) at either threshold. The pre-specified per-machine train-only threshold raises this to 3/30, at a slightly *lower* false-alarm cost (1.26 vs 1.37 per machine per day). That is still far from "catches what Tier 1 cannot". On this data and with these settings, the hypothesis that the multivariate AE catches correlation breaks in alarm terms is not supported.
- **The AE does react.** Its peak score inside a corrbreak is a median 15.9× the machine's normal score (IQR 6.1–31.4×), and 24 of 30 corrbreaks exceed 5× normal. A per-metric score cannot see these faults at all: every value stays inside the clean min–max. But the corrbreak peak (median 0.035) is about 4× below the global threshold (0.129). That threshold is set by the noisiest training windows of bursty machines, which typical windows (median score ~0.002) never come near. The global threshold is therefore what limits corrbreak detection, not the model's sensitivity. A threshold scaled to each machine's own normal level would turn more of this separation into alarms. That needs to be pre-specified and validated on held-out clean hours, not tuned on this slice; it is future work.
- **Most corrbreaks hit the net_in/net_out pair** (19 of 30, |r| ≈ 0.998). Swapping one side breaks a near-deterministic relation, and those faults give the clearest separation. The cpu/mem/disk_io pairs are weaker (|r| 0.5–0.7) and some barely move the score (ratio < 1). A pair with |r| ≈ 0.55 has no strong joint pattern to break.
- **Cost on the E5 faults: none.** At a matched false-alarm budget (Tier 1 threshold 16.5: 1.31 vs AE 1.37 per machine per day) the AE ties Tier 1 on spikes (0.63 vs 0.63), ties on breaks (1.00 vs 1.00) and is better on slow drifts (0.60 vs 0.47). It also catches drifts much earlier: mean delay 4.6 min vs 14.0 min. Overall it catches 69/120 vs 63/120. The detected faults are well explained: the injected metric is in `top_metrics` for all 69 detected faults.
- **Clean false alarms.** On the uninjected hours 0–72 the AE raises 1.53 false alarms per machine per day, within the project's budget of at most 2 per machine per day.
- **Consistency with E5.** The Tier 1 threshold-20 recalls are identical to E5's (spike 0.53, drift 0.40, break 1.00), which confirms the same spike/drift/break faults were injected per seed. Its false-alarm rate (0.65) differs slightly from E5's 0.67 because the 10 machines E5 left fault-free now each carry a labelled corrbreak.
- **Caveats.** There are 10 faults per type per seed, so per-seed recalls move by ±0.2 (see the min–max). The AE was trained once (seed 42), and the global and per-machine rows share that model. The matched Tier 1 threshold came out as 16.5 for all three seeds.

## Commands

```
python src/producer/alibaba_to_parquet.py --input data/external/machine_usage_m1-m40.csv --driver-memory 3g
python experiments/e5b_alibaba_ae.py
```

Outputs: models/ae_alibaba/ (model.pt, scaler.json, threshold.json, history.json), data/e5b/ (injected and replayed layers, logs), this file and e5b_alibaba_ae.json (incl. the Tier 1 sweep per seed and every injected fault).
