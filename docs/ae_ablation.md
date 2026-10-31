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
