# Incident attribution: hit-rate@3 on SMD

Each labelled SMD test anomaly segment (`data/reference/smd_interpretation`) is matched with the incidents (`data/incidents/source=smd`) of the same machine that overlap it in time. A segment is a **hit** if at least one of its labelled metrics is in the `top_metrics` (first 3) of an overlapping incident. **Hit-rate@3 = hits / all segments**, so a segment with no incident counts as a miss. *Hit-rate if covered* = hits / segments that have an overlapping incident.

| Machine | Segments | With incident | Hits@3 | Hit-rate@3 | Hit-rate if covered |
|---|---:|---:|---:|---:|---:|
| **All machines** | 327 | 182 | 172 | 52.6 % | 94.5 % |
| machine-1-1 | 8 | 7 | 6 | 75.0 % | 85.7 % |
| machine-1-2 | 10 | 9 | 9 | 90.0 % | 100.0 % |
| machine-1-3 | 11 | 2 | 2 | 18.2 % | 100.0 % |
| machine-1-4 | 12 | 3 | 3 | 25.0 % | 100.0 % |
| machine-1-5 | 7 | 4 | 4 | 57.1 % | 100.0 % |
| machine-1-6 | 30 | 16 | 16 | 53.3 % | 100.0 % |
| machine-1-7 | 13 | 8 | 8 | 61.5 % | 100.0 % |
| machine-1-8 | 20 | 5 | 4 | 20.0 % | 80.0 % |
| machine-2-1 | 13 | 9 | 7 | 53.8 % | 77.8 % |
| machine-2-2 | 11 | 7 | 7 | 63.6 % | 100.0 % |
| machine-2-3 | 10 | 8 | 8 | 80.0 % | 100.0 % |
| machine-2-4 | 20 | 9 | 9 | 45.0 % | 100.0 % |
| machine-2-5 | 21 | 18 | 18 | 85.7 % | 100.0 % |
| machine-2-6 | 8 | 4 | 3 | 37.5 % | 75.0 % |
| machine-2-7 | 20 | 8 | 7 | 35.0 % | 87.5 % |
| machine-2-8 | 1 | 1 | 1 | 100.0 % | 100.0 % |
| machine-2-9 | 10 | 7 | 7 | 70.0 % | 100.0 % |
| machine-3-1 | 4 | 4 | 3 | 75.0 % | 75.0 % |
| machine-3-2 | 9 | 5 | 5 | 55.6 % | 100.0 % |
| machine-3-3 | 28 | 8 | 8 | 28.6 % | 100.0 % |
| machine-3-4 | 8 | 4 | 4 | 50.0 % | 100.0 % |
| machine-3-5 | 11 | 5 | 5 | 45.5 % | 100.0 % |
| machine-3-6 | 11 | 10 | 9 | 81.8 % | 90.0 % |
| machine-3-7 | 5 | 5 | 4 | 80.0 % | 80.0 % |
| machine-3-8 | 6 | 5 | 5 | 83.3 % | 100.0 % |
| machine-3-9 | 4 | 4 | 4 | 100.0 % | 100.0 % |
| machine-3-10 | 13 | 4 | 4 | 30.8 % | 100.0 % |
| machine-3-11 | 3 | 3 | 2 | 66.7 % | 66.7 % |

## Provenance

- **Scores:** the stream output of the E3 clean run (`data/e3/e3_full2/clean/scores` on spark-master): the whole SMD set (1,416,825 records) through `score_job` with the autoencoder, 141,624 test windows of 28 machines, 1,158 anomalous windows (`is_anomaly = ae_flag`). The offline `data/scores` layer on the VM (4 Nov) has no autoencoder scores (`ae_score` null, no anomalies), so it cannot give incidents.
- **Incidents:** `python src/incidents/group.py --source smd` on those scores: 623 incidents.
- **Segments:** `data/reference/smd_interpretation`, 327 labelled test anomaly segments.
- **Run:** `python src/incidents/attribution_eval.py` on a laptop (pyarrow only).
