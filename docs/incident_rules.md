# Incident rules

An **incident** is a run of anomalous windows (`is_anomaly = true`) for the same `entity_id`. Input is the
`scores` layer; output is the `incidents` layer (`incident_id, entity_id, start_time, end_time, n_windows,
max_score, top_metrics, created_at`). Windows are 5 minutes: `window_end = window_start + 5 min`.

## Rules

1. **Dedup first.** One alert per (`entity_id`, `window_start`).
2. **Per machine.** Incidents never span two `entity_id`s. Sort each machine's alerts by `window_start`.
3. **Merge on a small gap.** Session gap = 2 × window size = 10 minutes. An alert joins the open incident if its
   `window_start` is at most 10 minutes after the previous alert's `window_start`. That means adjacent windows merge and
   **one** normal window between two alerts is bridged; two or more normal windows start a new incident.
4. `start_time` = first alert's `window_start`; `end_time` = last alert's `window_end`.
5. `n_windows` = number of alert windows (a bridged normal window is not counted).
6. `incident_id` = `entity_id` + `_` + `start_time` as `yyyyMMddHHmm` (UTC).
7. `max_score` = highest `ae_score` in the run (nulls ignored; null if every `ae_score` is null).
8. `top_metrics` = the 3 metrics that appear most often in the alert windows' `top_metrics`. Ties are broken by metric
   name, so the result is deterministic. Fewer than 3 distinct metrics gives a shorter list.

## Worked examples

**1. A run of consecutive windows is one incident.**

| entity_id | window_start | ae_score | top_metrics |
|---|---|---|---|
| machine-1-2 | 10:20 | 0.30 | m05, m02, m11 |
| machine-1-2 | 10:25 | 0.22 | m05, m11, m07 |
| machine-1-2 | 10:30 | 0.18 | m05, m02, m11 |

Counts: m05 = 3, m11 = 3, m02 = 2, m07 = 1.

| incident_id | start | end | n_windows | max_score | top_metrics |
|---|---|---|---|---|---|
| machine-1-2_202601011020 | 10:20 | 10:35 | 3 | 0.30 | m05, m11, m02 |

**2. One normal window is bridged; two are not.** Alerts at 10:00, 10:10 and 10:25 (10:05 and 10:15 were normal).
10:00 to 10:10 is 10 minutes, so they merge. 10:10 to 10:25 is 15 minutes, so a new incident starts.

| incident_id | start | end | n_windows |
|---|---|---|---|
| machine-1-1_202601011000 | 10:00 | 10:15 | 2 |
| machine-1-1_202601011025 | 10:25 | 10:30 | 1 |

**3. Machines are separate, and duplicates count once.** `machine-2-1` has the alert at 10:40 twice (a replayed
message); `machine-1-2` also has an alert at 10:40.

| entity_id | window_start | alert rows | after dedup |
|---|---|---|---|
| machine-1-2 | 10:40 | 1 | 1 |
| machine-2-1 | 10:40 | 2 | 1 |

| incident_id | entity_id | start | end | n_windows |
|---|---|---|---|---|
| machine-1-2_202601011040 | machine-1-2 | 10:40 | 10:45 | 1 |
| machine-2-1_202601011040 | machine-2-1 | 10:40 | 10:45 | 1 |
