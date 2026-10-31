# Autoencoder (Tier 3) ablation on SMD

All runs use `src/detection/autoencoder.py` on all 28 SMD machines (141,669 train and 141,708 test 5-minute
windows): a 38-32-8-32-38 ReLU autoencoder with seed 42 and early stopping (patience 5). Metrics come from
`src/detection/evaluate.py`, computed on time-ordered test windows, with TP/FP/FN summed over machines.

| Run | Command flags | Epochs run (best) |
| --- | --- | --- |
| (a) default | `--threshold global --std-floor 0 --epochs 50` | 50 (49) |
| (b) | `--threshold per-machine --std-floor 0.05 --epochs 100` | 81 (76) |
| (c) | `--threshold global --std-floor 0 --epochs 100` | 68 (63) |

Run (a) is the default and the run whose outputs are in `data/scores_ae/` and `models/ae_smd/`.

## Overall

| | (a) global, no floor, 50 ep | (b) per-machine, floor 0.05, 100 ep | (c) global, no floor, 100 ep |
| --- | --- | --- | --- |
| point precision | 0.142 | 0.119 | 0.141 |
| point recall | 0.492 | 0.454 | 0.495 |
| point F1 | 0.221 | 0.188 | 0.219 |
| event precision | 0.227 | 0.200 | 0.223 |
| event recall | 0.869 | 0.838 | 0.866 |
| event F1 | **0.360** | 0.322 | 0.355 |
| windows flagged | 21,215 | 23,410 | 21,583 |

## The 6 worst machines in run (a)

| machine | anomalous windows | flagged (a) | flagged (b) | flagged (c) | point F1 (a) | point F1 (b) | point F1 (c) | event F1 (a) | event F1 (b) | event F1 (c) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| machine-2-8 | 33 | 2591 | 1542 | 2557 | 0.025 | 0.041 | 0.025 | 0.025 | 0.042 | 0.025 |
| machine-3-11 | 42 | 1673 | 1681 | 1674 | 0.048 | 0.049 | 0.049 | 0.049 | 0.049 | 0.049 |
| machine-3-5 | 93 | 1810 | 1641 | 1842 | 0.073 | 0.077 | 0.071 | 0.092 | 0.100 | 0.090 |
| machine-3-4 | 202 | 4191 | 4190 | 4191 | 0.092 | 0.092 | 0.092 | 0.092 | 0.092 | 0.092 |
| machine-3-7 | 91 | 1521 | 1492 | 1522 | 0.073 | 0.069 | 0.073 | 0.111 | 0.098 | 0.111 |
| machine-1-5 | 26 | 325 | 794 | 363 | 0.114 | 0.054 | 0.103 | 0.146 | 0.063 | 0.132 |

Outside this list, run (b) dropped machine-3-3's event F1 from 0.194 to 0.000 and machine-3-2's from 0.835 to
0.016, and raised machine-2-4's from 0.382 to 0.767.

## What we learned

Machines 3-4, 3-11, 3-5, 3-7, 2-8 and 1-6 flag between a quarter and nearly all of their test windows in every run
(machine-1-6 flags 4,085 of 4,739), so their test data has drifted away from train and no threshold fitted on
train windows can fix it.
Per-machine thresholds hurt the machines whose train data is very regular: in run (b) machine-1-5 more than
doubled its false alarms, and machine-3-3 and machine-3-2 lost almost all of their detections.
Training longer alone (run c) gives slightly lower F1 (0.355 vs 0.360), and run (b) changed the std floor together
with the threshold, so the floor's own effect is not yet measured; the next gain most likely has to come from
handling drift.

## v2: drift handling (flat-metric mask, mean + max inputs, recentering)

Same data, seed, global threshold and 50 epochs as run (a) above. Metrics now come from the newer functions in
`src/detection/evaluate.py`. Composite F1 is the harmonic mean of event recall (share of faults with at least one
flagged window) and point-wise alarm precision, so every false alarm counts. Point-adjusted event F1 is kept only
as a reference: it is generous because one hit marks the whole fault as detected.

| Run | Flags | Composite F1 | Faults caught | Alarm precision | False alarms / machine / day | Median delay (min) | PR-AUC | Event F1 (reference, generous) | Windows flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 (old settings, rerun) | `--flat-std 0 --inputs mean --recenter 0 --adaptive 0` | 0.236 | 222 / 326 | 0.142 | 36.98 | 0.0 | 0.145 | 0.360 | 21,215 |
| v2 first cut (grid row 6) | `--flat-std 0.01 --inputs mean_max --recenter 24 --adaptive 0` | 0.451 | 156 / 326 | 0.427 | 2.98 | 5.0 | 0.227 | 0.732 | 2,558 |

The v1 rerun reproduces the 0.360 run exactly: the same `ae_score`, `ae_flag` and `top_metrics` for all 141,708
test windows, and 50 epochs with best epoch 49. v2 also ran 50 epochs (best 49) and masked 386 machine-metrics
whose train std was below 0.01. The default chosen from the grid below is row 9.

### Ablation grid

Each option is switched off with `--flat-std 0`, `--inputs mean`, `--recenter 0` or `--adaptive 0`; everything
else is the same as above. Every run trained for the full 50 epochs (best epoch 47 or 49).

| # | Options on | Composite F1 | Faults caught | Alarm precision | False alarms / machine / day | Median delay (min) | Point-adjusted F1 | Windows flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | flat-std 0.01 | 0.274 | 174 / 326 | 0.184 | 19.26 | 2.5 | 0.443 | 11,616 |
| 2 | mean_max | 0.264 | 228 / 326 | 0.163 | 31.73 | 0.0 | 0.398 | 18,648 |
| 3 | recenter 24 | 0.535 | 205 / 326 | 0.465 | 5.02 | 0.0 | 0.758 | 4,615 |
| 4 | flat + mean_max | 0.302 | 177 / 326 | 0.209 | 16.09 | 0.0 | 0.483 | 10,012 |
| 5 | flat + recenter | 0.428 | 146 / 326 | 0.410 | 3.01 | 5.0 | 0.720 | 2,514 |
| 6 | flat + mean_max + recenter | 0.451 | 156 / 326 | 0.427 | 2.98 | 5.0 | 0.732 | 2,558 |
| 7 | flat + mean_max + recenter + adaptive 288 | 0.461 | 143 / 326 | 0.487 | 1.06 | 5.0 | 0.751 | 1,015 |
| 8 | recenter + adaptive 288 | 0.513 | 172 / 326 | 0.500 | 1.19 | 5.0 | 0.816 | 1,169 |
| 9 | mean_max + recenter + adaptive 288 | **0.525** | 180 / 326 | 0.500 | 1.18 | 0.0 | 0.828 | 1,161 |

Selection rule: highest composite F1 among rows with at most 2 false alarms per machine per day and a median delay
of at most 15 minutes. Rows 7, 8 and 9 meet both limits; row 9 has the highest composite F1, so it is the default
(`python src/detection/autoencoder.py` with no flags, i.e. `--flat-std 0 --inputs mean_max --recenter 24
--adaptive 288`), and its outputs are in `data/scores_ae/` and `models/ae_smd/`.

What the grid shows. Recentering is the option that matters most: on its own (row 3) it gives the highest composite
F1 (0.535), but it is above the false-alarm limit at 5.02 per machine per day. The adaptive one-day percentile
brings false alarms down to about 1.1 to 1.2 per machine per day wherever it is added, at the cost of faults: 33
with recentering alone (row 3 to row 8) and 13 in the masked setup (row 6 to row 7). A score that stays high for a
while also lifts that machine's own percentile for the next day, which is part of why faults are lost. Masking flat
metrics cuts false alarms in every combination (only slightly once the adaptive check is on: 1.18 to 1.06 from row 9
to row 7), but it always drops faults: 59 of 205 with
recentering (row 3 to row 5) and 37 of 180 in the full setup (row 9 to row 7), so some faults on SMD show up only in
low-variance metrics, and the mask is off by default. The mean + max inputs help a little in every pairing (v1 to
row 2, row 1 to 4, row 5 to 6, row 8 to 9). If the false-alarm limit is relaxed, row 3 is still the best detector.

Separate experiment outside this repo (Claude chat, same v2-style inputs, 30 epochs, early stopping patience 4):
LSTM autoencoder over the last 6 windows scored composite F1 0.569 vs 0.579 for a dense autoencoder in the same
setup, so a sequence model did not help once drift was handled. Not reproduced here.

Why recentering fixes drift: on machine-3-4, metric m06 has a mean of 0.985 over the train windows (std 0.003) but
0.21 over the normal test windows. Standardised with the train mean, every normal test window is about 260 train
standard deviations off, so the autoencoder flags almost the whole test period (4,191 of 4,738 windows in v1).
Recentering standardises each test window against the median of its own previous 24 windows, so a lasting level
shift is absorbed after about 12 windows (one hour) while a sudden spike still stands out against the recent median;
with recentering alone (grid row 3) machine-3-4 drops from 4,191 to 352 flagged windows.
m06 on machine-3-4 is also below the 0.01 flat-std cut, so masking alone (row 1) brings that machine down to 36.
In the default (row 9), which has no mask, machine-3-4 flags 55 windows but catches only 4 of its 8 faults,
against 7 of 8 in the masked row 7.
