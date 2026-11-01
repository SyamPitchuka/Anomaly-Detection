# Incident rules

An **incident** is a run of anomalous windows (`is_anomaly = true`) for the same `entity_id`. The output is one row per
incident in the `incidents` layer (`incident_id, entity_id, start_time, end_time, n_windows, max_score, top_metrics,
created_at`; `created_at` is the time the job writes the row). Windows are 5 minutes: `window_end = window_start + 5 min`.
In the examples, all times are on 2026-01-01 UTC.

**Input and mode.** The grouping is one function over a table of alert rows with the columns of the `anomalies` message
(`entity_id, window_start, window_end, ae_score, is_anomaly, top_metrics`). Sprint 1 runs it as a **batch** over the
`scores` layer filtered to `is_anomaly = true`, and rebuilds all incidents on every run (overwrite). The same function
can later be fed from the `anomalies` topic.

## Rules

1. **Dedup first.** One alert per (`entity_id`, `window_start`). If duplicates differ, keep the one with the highest
   `ae_score` (a null `ae_score` loses to any number), so the result does not depend on arrival order.
2. **Per machine.** Incidents never span two `entity_id`s. Sort each machine's alerts by `window_start`.
3. **Merge on a small gap.** Session gap = 2 × window size = 10 minutes. An alert joins the open incident if its
   `window_start` is at most 10 minutes after the previous alert's `window_start`. Adjacent windows merge and **one**
   normal window between two alerts is bridged; two or more normal windows start a new incident.
4. `start_time` = first alert's `window_start`; `end_time` = last alert's `window_end`.
5. `n_windows` = number of alert windows (a bridged normal window is not counted).
6. `incident_id` = `entity_id` + `_` + `start_time` as `yyyyMMddHHmm` (UTC).
7. `max_score` = highest `ae_score` in the run (nulls ignored). It is **null** when every alert in the run came from
   Tier 1 only (no `ae_score`); there is no fallback to `tier1_score`, because the two scores have different scales.
8. `top_metrics` = the 3 metrics that appear most often in the alert windows' `top_metrics` (empty or null lists add
   nothing). Ties are broken by metric name, so the result is deterministic. Fewer than 3 distinct metrics gives a
   shorter list.

## Limits

- **Late or out-of-order alerts:** none are handled incrementally. Every run recomputes all incidents from all alerts, so
  `incident_id` is always consistent with the data. Incidents are therefore not live, only as fresh as the last run.
- **Alert flood:** `is_anomaly = ae_flag OR tier1_flag`. If Tier 1 flags most windows, incidents merge into a few long
  runs with a null `max_score`. Check the share of alert windows before relying on the incident list.

## Worked examples

**1. A run of consecutive windows is one incident** (times and `ae_score` from `tools/toy_pipeline.py`; the per-window
`top_metrics` are illustrative).

| entity_id | window_start | ae_score | top_metrics |
|---|---|---|---|
| machine-1-2 | 10:20 | 0.15 | m00, m02, m01 |
| machine-1-2 | 10:25 | 0.26 | m00, m02, m01 |
| machine-1-2 | 10:30 | 0.26 | m00, m02 |

Counts: m00 = 3, m02 = 3, m01 = 2.

| incident_id | start | end | n_windows | max_score | top_metrics |
|---|---|---|---|---|---|
| machine-1-2_202601011020 | 10:20 | 10:35 | 3 | 0.26 | m00, m02, m01 |

**2. One normal window is bridged; two are not.** Alerts at 10:00, 10:10 and 10:25; the windows at 10:05, 10:15 and 10:20
were normal. 10:00 to 10:10 is 10 minutes (one normal window between), so they merge. 10:10 to 10:25 is 15 minutes (two
normal windows between), so a new incident starts.

| incident_id | start | end | n_windows |
|---|---|---|---|
| machine-1-1_202601011000 | 10:00 | 10:15 | 2 |
| machine-1-1_202601011025 | 10:25 | 10:30 | 1 |

**3. Machines are separate, and duplicates count once.** `machine-2-1` has its 10:40 alert twice (a replayed message);
`machine-1-2` also has an alert at 10:40.

| entity_id | window_start | ae_score | alert rows | after dedup |
|---|---|---|---|---|
| machine-1-2 | 10:40 | 0.26 | 1 | 1 |
| machine-2-1 | 10:40 | 97.47 | 2 | 1 |

| incident_id | entity_id | start | end | n_windows | max_score |
|---|---|---|---|---|---|
| machine-1-2_202601011040 | machine-1-2 | 10:40 | 10:45 | 1 | 0.26 |
| machine-2-1_202601011040 | machine-2-1 | 10:40 | 10:45 | 1 | 97.47 |
