"""E5b: does the multivariate autoencoder catch injected Alibaba faults that the per-metric Tier 1 cannot?

Run from the repo root (after python src/producer/alibaba_to_parquet.py --input ... as in docs/results/e5.md):
    python experiments/e5b_alibaba_ae.py                      # train models/ae_alibaba, seeds 7 8 9, write the report
    python experiments/e5b_alibaba_ae.py --skip-train         # reuse models/ae_alibaba
    python experiments/e5b_alibaba_ae.py --seeds 7 --machines 12 --per-type 2 --out data/e5b/quick   # quick run

Steps:
1. Train AE v2 on Alibaba's 7 metrics with src/detection/autoencoder.py `run` (source alibaba, production settings:
   mean_max inputs, recenter 24, adaptive 288, global threshold = p99.5 of the train score x 1.1, seed 42), with a
   time split: train = windows from hour --train-from-hours (72) to the end of the trace, so the 48-hour test slice
   (hours 0-48) is never seen in training and a 24-hour gap separates them. Model -> models/ae_alibaba/; the scores
   of the unseen hours 0-72 (no faults) -> data/e5b/ae_clean/ (clean false-alarm rate).
2. Per seed (7, 8, 9): src/producer/inject_faults.py `inject` on the first 40 machines, hours 0-48, clean first
   24 hours, 10 faults per type for spike, drift, break (as E5) and corrbreak (E5b). The injected raw layer is
   replayed through src/producer/replay.py into a capture producer (the Kafka messages), and the replayed records
   are written back as a raw layer, so both detectors see exactly the replayed messages.
3. Tier 1 = inject_faults.detect (robust z on 5-minute means, baseline = the clean 24 hours, as E5). AE =
   autoencoder.score_layer with models/ae_alibaba on the whole 48-hour replayed slice (the first 24 hours give the
   recenter and adaptive history). Both evaluated with inject_faults.report (src/detection/evaluate.py) on the
   windows after the clean part: recall per fault type, detection delay, false alarms per machine per day.
4. Tier 1 is reported at threshold 20 (E5) and at the threshold of THRESHOLDS whose all-machine false-alarm rate is
   closest to the AE's (matched false-alarm budget, chosen per seed; ties -> the higher threshold).
5. Report: docs/results/e5b_alibaba_ae.md and .json (per seed, mean (min-max) over seeds); with --out other than
   data/e5b the report goes there instead, so quick runs leave docs/results alone.

Writes only models/ae_alibaba/ (unless --model-dir), data/e5b/ (or --out) and the report. Never models/ae_smd or other
production models, never contracts.
"""
import argparse
import contextlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection import autoencoder as M  # noqa: E402
from src.producer import inject_faults as I  # noqa: E402

SOURCE = "alibaba"
SEEDS = (7, 8, 9)
MACHINES, HOURS, CLEAN_HOURS, PER_TYPE = 40, 48, 24, 10
TRAIN_FROM_HOURS = 72
E5_THRESHOLD = 20.0
THRESHOLDS = tuple(float(t) for t in np.round(np.concatenate([np.arange(3, 20, 0.5), np.arange(20, 60, 1),
                                                             np.arange(60, 201, 5)]), 2))
MODEL_DIR = Path("models/ae_alibaba")
OUT = Path("data/e5b")
DOCS = Path("docs/results")
REPORT_NAME = "e5b_alibaba_ae"
TYPES = I.ALL_TYPES
ROWS = (*TYPES, "all")
WINDOW = I.WINDOW_S


# ---------------------------------------------------------------- helpers

@contextlib.contextmanager
def log_to(path: Path):
    """Send stdout (per-epoch / per-machine output of autoencoder.py) to a log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        yield


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def secs(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def records_to_raw(records: list[dict], folder: Path) -> None:
    """Replayed JSON records -> raw layer, one file per entity_id (rows in replay = event-time order)."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.parquet"):
        old.unlink()
    schema = S.arrow_schema("raw")
    by_entity: dict[str, list[dict]] = {}
    for r in records:
        by_entity.setdefault(r["entity_id"], []).append(r)
    for entity, rows in by_entity.items():
        us = pd.to_datetime([r["event_time"] for r in rows], utc=True).as_unit("us").asi8
        table = pa.table({
            "entity_id": pa.array([entity] * len(rows), pa.string()),
            "event_time": pa.array(us, pa.int64()).cast(schema.field("event_time").type),
            "metrics": pa.array([list(r["metrics"].items()) for r in rows], schema.field("metrics").type),
            "attrs": pa.array([list((r["attrs"] or {}).items()) for r in rows], schema.field("attrs").type),
            "label": pa.array([r["label"] for r in rows], pa.int32()),
        }, schema=schema)
        pq.write_table(table, folder / f"{entity}.parquet")


def ae_flags(meta: pd.DataFrame, ae_flag: np.ndarray) -> pd.DataFrame:
    """score_layer output -> the (entity_id, window [epoch s], flag) frame inject_faults.report takes."""
    return pd.DataFrame({"entity_id": meta["entity_id"].to_numpy(),
                         "window": meta["window_start"].to_numpy(dtype=np.int64) // 1_000_000,
                         "flag": np.asarray(ae_flag, dtype=bool)})


def table_dict(table: pd.DataFrame) -> dict:
    return {name: {"events": int(r["events"]), "detected": int(r["detected"]), "recall": r["recall"],
                   "mean_delay_min": r["mean_delay_min"], "median_delay_min": r["median_delay_min"],
                   "false_alarms_per_day": r["false_alarms/day"]} for name, r in table.iterrows()}


def matched_threshold(tier1_scores: pd.DataFrame, log: dict, windows: pd.DataFrame, target_fa: float,
                      thresholds=THRESHOLDS) -> tuple[float, pd.DataFrame, list[dict]]:
    """Tier 1 threshold whose all-machine false alarms/day is closest to target_fa (ties -> higher threshold)."""
    sweep, best = [], None
    for t in thresholds:
        table = I.report(log, windows, tier1_scores.assign(flag=tier1_scores["score"] > t))
        fa = float(table.loc["all", "false_alarms/day"])
        sweep.append({"threshold": t, "false_alarms_per_day": fa, "all_recall": float(table.loc["all", "recall"])})
        key = (abs(fa - target_fa), -t)
        if best is None or key < best[0]:
            best = (key, t, table)
    return best[1], best[2], sweep


def score_separation(log: dict, meta: pd.DataFrame, ae_score: np.ndarray, info: dict, clean_end: int) -> list[dict]:
    """Per fault: the max ae_score over its windows, the machine's median ae_score over the clean part of the
    slice (no labels needed: faults start after it), their ratio, and whether the max exceeds the global and the
    machine's own train-only threshold (before the adaptive limit)."""
    ent = meta["entity_id"].to_numpy()
    ws = meta["window_start"].to_numpy(dtype=np.int64) // 1_000_000
    out = []
    for f in log["faults"]:
        e, a, b = f["entity_id"], secs(f["start_time"]), secs(f["end_time"])
        inside = (ent == e) & (ws >= a // WINDOW * WINDOW) & (ws < b)
        normal = (ent == e) & (ws < clean_end)
        peak = float(np.nanmax(ae_score[inside])) if np.isfinite(ae_score[inside]).any() else float("nan")
        base = float(np.nanmedian(ae_score[normal])) if np.isfinite(ae_score[normal]).any() else float("nan")
        own = info["per_machine"].get(e, info["global"])
        out.append({"fault_id": f["fault_id"], "type": f["type"], "metric": f["metric"], "peak": peak,
                    "normal_median": base, "ratio": peak / base if base > 0 else float("nan"),
                    "over_global": bool(peak > info["global"]), "over_own_threshold": bool(peak > own)})
    return out


def separation_summary(seeds: list[dict]) -> dict:
    out = {}
    for t in TYPES:
        rows = [r for s in seeds for r in s["separation"] if r["type"] == t]
        ratios = [r["ratio"] for r in rows if r["ratio"] is not None and math.isfinite(r["ratio"])]
        out[t] = {"n": len(rows), "median_ratio": float(np.median(ratios)) if ratios else None,
                  "q1_ratio": float(np.percentile(ratios, 25)) if ratios else None,
                  "q3_ratio": float(np.percentile(ratios, 75)) if ratios else None,
                  "median_peak": float(np.median([r["peak"] for r in rows])),
                  "median_normal": float(np.median([r["normal_median"] for r in rows])),
                  "ratio_over_5": sum(r > 5 for r in ratios),
                  "over_global": sum(r["over_global"] for r in rows),
                  "over_own_threshold": sum(r["over_own_threshold"] for r in rows)}
    return out


# ---------------------------------------------------------------- steps

def train_model(root: Path, model_dir: Path, out: Path, train_from_hours: float, epochs: int) -> dict:
    with log_to(root / out / "logs" / "train_ae_alibaba.log"):
        res = M.run(SOURCE, root=root, epochs=epochs, train_from_hours=train_from_hours, model_dir=root / model_dir,
                    scores_dir=root / out / "ae_clean" / f"source={SOURCE}")
    meta = res["meta"]
    best = min(res["history"], key=lambda h: h["val_loss"])
    return {"train_windows": int((meta["split"] == "train").sum()),
            "clean_windows": int((meta["split"] == "test").sum()), "machines": int(meta["entity_id"].nunique()),
            "epochs_run": len(res["history"]), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
            "threshold_global": res["thresholds"]["global"],
            "clean_false_alarms_per_day": res["overall"]["fa_day"],
            "clean_flagged": int(res["ae_flag"][(meta["split"] == "test").to_numpy()].sum())}


def run_seed(root: Path, seed: int, model_dir: Path, out: Path, machines: int, hours: float, clean_hours: float,
             per_type: int) -> dict:
    seed_dir = root / out / f"seed{seed}"
    log = I.inject(root, seed_dir / "inject", machines=machines, hours=hours, clean_hours=clean_hours,
                   per_type=per_type, seed=seed, types=TYPES)
    records = I.replay_records(seed_dir / "inject")
    replayed = seed_dir / "replayed" / S.PATHS["raw"] / f"source={SOURCE}"
    records_to_raw(records, replayed)
    df = I.to_frame(records)
    windows = I.window_labels(df)
    clean_end = secs(log["clean_end"])

    meta, ae_score, ae_flag, top = M.score_layer(root / model_dir, replayed)
    flags_ae = ae_flags(meta, ae_flag)
    ae_table = I.report(log, windows, flags_ae)
    # secondary, pre-specified variant: the per-machine thresholds saved at training time (p99.5 x 1.1 of each
    # machine's own train-hour scores; no injected labels, nothing tuned on this slice), same adaptive limit
    info = json.loads((root / model_dir / "threshold.json").read_text())
    adaptive = info.get("adaptive")
    limit = M.adaptive_limits(meta, ae_score, adaptive["windows"], adaptive["percentile"]) if adaptive else None
    pm_flag = M.flags(meta, ae_score, {**info, "per_machine": {**{e: info["global"] for e in meta["entity_id"]},
                                                                 **info["per_machine"]}}, "per-machine", limit)
    pm_table = I.report(log, windows, ae_flags(meta, pm_flag))
    separation = score_separation(log, meta, ae_score, info, clean_end)
    tier1 = I.detect(df, clean_end, E5_THRESHOLD)
    t20_table = I.report(log, windows, tier1)
    target = float(ae_table.loc["all", "false_alarms/day"])
    t_match, match_table, sweep = matched_threshold(tier1, log, windows, target)

    hits = {}                                     # AE top_metrics of the first flagged window contain the fault metric
    pos = {(e, w): i for i, (e, w) in enumerate(zip(flags_ae["entity_id"], flags_ae["window"]))}
    for f in log["faults"]:
        start, end = secs(f["start_time"]), secs(f["end_time"])
        first = next((w for w in range(start // 300 * 300, end, 300)
                      if (f["entity_id"], w) in pos and flags_ae["flag"].iat[pos[(f["entity_id"], w)]]), None)
        if first is not None:
            hits.setdefault(f["type"], []).append(f["metric"] in top[pos[(f["entity_id"], first)]])
    return {"seed": seed, "rows": log["rows"], "faults": log["faults"],
            "ae": table_dict(ae_table), "ae_per_machine": table_dict(pm_table), "separation": separation,
            "tier1_matched": table_dict(match_table), "tier1_matched_threshold": t_match,
            "tier1_t20": table_dict(t20_table), "tier1_sweep": sweep,
            "ae_top3_hit": {t: {"hits": int(sum(v)), "of": len(v)} for t, v in hits.items()}}


# ---------------------------------------------------------------- report

DETECTORS = (("ae", "AE (Alibaba, v2), global threshold [main]"), ("tier1_matched", "Tier 1, matched false alarms"),
             ("tier1_t20", "Tier 1, threshold 20 (E5)"),
             ("ae_per_machine", "AE, per-machine train-only threshold [secondary]"))


def summarise(seeds: list[dict]) -> dict:
    out = {}
    for key, _ in DETECTORS:
        out[key] = {}
        for row in ROWS:
            entry = {}
            for metric in ("recall", "mean_delay_min", "false_alarms_per_day"):
                vals = [s[key][row][metric] for s in seeds if row in s[key]]
                vals = [v for v in vals if v is not None and math.isfinite(v)]
                entry[metric] = ({"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))}
                                 if vals else None)
            entry["detected"] = sum(s[key][row]["detected"] for s in seeds if row in s[key])
            entry["events"] = sum(s[key][row]["events"] for s in seeds if row in s[key])
            out[key][row] = entry
    t = [s["tier1_matched_threshold"] for s in seeds]
    out["tier1_matched_threshold"] = {"mean": float(np.mean(t)), "min": min(t), "max": max(t)}
    return out


def _f(x, digits=2):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"


def _mm(e, digits=2):
    if e is None:
        return "–"
    if len({round(e["min"], digits), round(e["max"], digits)}) == 1:
        return _f(e["mean"], digits)
    return f"{_f(e['mean'], digits)} ({_f(e['min'], digits)}–{_f(e['max'], digits)})"


def render(doc: dict) -> str:
    cfg, tr, seeds, summ = doc["config"], doc["training"], doc["seeds"], doc["summary"]
    L = ["# E5b – autoencoder vs Tier 1 on injected Alibaba faults (incl. correlation breaks)", "",
         "Second dataset, no labels: faults are injected into real Alibaba v2018 machine metrics (as in "
         "[E5](e5.md)), with a fourth, multivariate fault type. Script: `experiments/e5b_alibaba_ae.py`; "
         f"run {doc['run_date']}, {doc['runtime_s'] / 60:.1f} min on one laptop (CPU).", "",
         "## Setup", "",
         f"- Data: machine_usage of machines m_1..m_40 (raw layer from `src/producer/alibaba_to_parquet.py`), "
         f"8 days.",
         f"- AE: `src/detection/autoencoder.py --source alibaba --train-from-hours {cfg['train_from_hours']:g}`, "
         f"7 metrics x (mean, max) = 14 inputs, 14-32-8-32-14, production v2 settings (recenter 24, adaptive 288, "
         f"global threshold = p99.5 of the train score x 1.1, seed 42). Trained on hours "
         f"{cfg['train_from_hours']:g}–192 only: {tr['train_windows']} windows of {tr['machines']} machines, "
         f"{tr['epochs_run']} epochs (best {tr['best_epoch']}), threshold {tr['threshold_global']:.4f}. "
         f"The test slice (hours 0–{cfg['hours']:g}) is never seen in training; hours "
         f"{cfg['hours']:g}–{cfg['train_from_hours']:g} are a gap.",
         f"- On the uninjected hours 0–{cfg['train_from_hours']:g} the AE raises "
         f"{tr['clean_false_alarms_per_day']:.2f} false alarms per machine per day ({tr['clean_flagged']} of "
         f"{tr['clean_windows']} windows; includes the first 12 h, where the adaptive limit has no history yet).",
         f"- Faults: `src/producer/inject_faults.py --types spike drift break corrbreak`, {cfg['machines']} machines, "
         f"first {cfg['hours']:g} h, clean first {cfg['clean_hours']:g} h, {cfg['per_type']} per type (one per "
         f"machine), seeds {', '.join(map(str, cfg['seeds']))}. spike/drift/break exactly as E5 (same machines, "
         "metrics and times per seed: corrbreak is planned last, on the 10 machines E5 left fault-free).",
         "- **corrbreak**: for 15 minutes one metric with a correlated partner (|r| >= 0.5 of the clean 5-minute "
         "means; else the most correlated one) is replaced by the same metric's values from a random clean "
         "15-minute stretch of that machine, whose mean lies between the clean p10 and p90 and on the other side "
         "of the clean median than the replaced stretch. Every value stays inside the machine's clean min–max.",
         "- Both detectors see the replayed messages (`replay.py` into a capture producer). Tier 1 = E5's robust "
         "z-score (5-minute means, median/MAD of the clean 24 h). AE = the saved model on the whole 48 h (first "
         "24 h = recenter/adaptive history). Evaluation with `inject_faults.report` (`src/detection/evaluate.py`) "
         "on windows after the clean part: recall = share of faults with a flagged window, delay = fault start to "
         "first flagged window, false alarms per machine per day (all 40 machines for 'all', the type's 10 "
         "machines in the per-type rows).",
         f"- Tier 1 is shown at threshold 20 (E5) and at a matched budget: the threshold (grid 3–200) whose "
         "all-machine false alarms per day are closest to the main AE row's, chosen per seed.",
         "- Main result = AE with the production global threshold. Secondary, pre-specified row = the same scores "
         "with the per-machine thresholds saved at training time (p99.5 x 1.1 of each machine's own train-hour "
         "scores, `--threshold per-machine` of autoencoder.py) and the same adaptive limit. Both thresholds come "
         "from the training hours only; nothing is tuned on the injected slice or its labels.", "",
         "## Mean over seeds (min–max)", "",
         "| detector | false alarms / machine / day | spike | drift | break | corrbreak | all |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, name in DETECTORS:
        s = summ[key]
        label = name + (f" (thr {_mm(summ['tier1_matched_threshold'], 1)})" if key == "tier1_matched" else "")
        L.append(f"| {label} | {_mm(s['all']['false_alarms_per_day'])} | "
                 + " | ".join(f"{_mm(s[t]['recall'])}" for t in TYPES)
                 + f" | {_mm(s['all']['recall'])}, {s['all']['detected']}/{s['all']['events']} |")
    L += ["", "Recall per fault type. Mean detection delay in minutes (detected faults only):", "",
          "| detector | spike | drift | break | corrbreak |", "| --- | --- | --- | --- | --- |"]
    for key, name in DETECTORS:
        L.append(f"| {name} | " + " | ".join(_mm(summ[key][t]["mean_delay_min"], 1) for t in TYPES) + " |")
    L += ["", "## Per seed", ""]
    for s in seeds:
        L += [f"### Seed {s['seed']} (Tier 1 matched threshold {s['tier1_matched_threshold']:g})", "",
              "| detector | FA / machine / day | spike | drift | break | corrbreak | all | delay spike / drift / "
              "break / corrbreak (min) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, name in DETECTORS:
            t = s[key]
            L.append(f"| {name} | {_f(t['all']['false_alarms_per_day'])} | "
                     + " | ".join(f"{t[x]['detected']}/{t[x]['events']}" for x in TYPES)
                     + f" | {_f(t['all']['recall'])} | "
                     + " / ".join(_f(t[x]["mean_delay_min"], 1) for x in TYPES) + " |")
        L.append("")
    hit = {t: [sum(s["ae_top3_hit"].get(t, {}).get(k, 0) for s in seeds) for k in ("hits", "of")] for t in TYPES}
    sep = doc["separation"]
    L += ["## Score separation (does the AE react without alarming?)", "",
          "Per fault: peak = max ae_score over the fault's windows; normal = the machine's median ae_score over the "
          "clean first 24 h of the slice (no labels used). All seeds pooled; thresholds compared to the peak before "
          f"the adaptive limit (global threshold {tr['threshold_global']:.4f}).", "",
          "| fault type | faults | median peak / normal (IQR) | peak > 5x normal | median peak | median normal | "
          "peak > global thr | peak > own train-only thr |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in TYPES:
        e = sep[t]
        L.append(f"| {t} | {e['n']} | {_f(e['median_ratio'], 1)} ({_f(e['q1_ratio'], 1)}–{_f(e['q3_ratio'], 1)}) | "
                 f"{e['ratio_over_5']}/{e['n']} | {_f(e['median_peak'], 4)} | {_f(e['median_normal'], 4)} | "
                 f"{e['over_global']}/{e['n']} | {e['over_own_threshold']}/{e['n']} |")
    L += ["", "AE attribution (all seeds, main row): the injected metric is among `top_metrics` of the first flagged "
          "window in " + ", ".join(f"{t} {h}/{n}" for t, (h, n) in hit.items()) + " detected faults.", "",
          "## Interpretation", "", "INTERPRETATION", "",
          "## Commands", "", "```",
          f"python src/producer/alibaba_to_parquet.py --input data/external/machine_usage_m1-m40.csv --driver-memory 3g",
          "python experiments/e5b_alibaba_ae.py",
          "```", "", "Outputs: models/ae_alibaba/ (model.pt, scaler.json, threshold.json, history.json), data/e5b/ "
          "(injected and replayed layers, logs), this file and e5b_alibaba_ae.json (incl. the Tier 1 sweep per seed "
          "and every injected fault).", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--machines", type=int, default=MACHINES)
    ap.add_argument("--hours", type=float, default=HOURS)
    ap.add_argument("--clean-hours", type=float, default=CLEAN_HOURS)
    ap.add_argument("--per-type", type=int, default=PER_TYPE)
    ap.add_argument("--train-from-hours", type=float, default=TRAIN_FROM_HOURS)
    ap.add_argument("--epochs", type=int, default=M.MAX_EPOCHS)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--out", type=Path, default=OUT, help="data folder; the report goes to docs/results only for "
                                                           f"the default {OUT}")
    ap.add_argument("--skip-train", action="store_true", help="reuse the model in --model-dir")
    a = ap.parse_args(argv)
    if a.train_from_hours < a.hours:
        ap.error("--train-from-hours must not overlap the test slice (>= --hours)")
    if a.model_dir.name.endswith("_smd"):
        ap.error(f"refusing to write a model to {a.model_dir}: production SMD models stay untouched")

    if a.skip_train:
        info = json.loads((a.root / a.out / "training.json").read_text())
    else:
        info = train_model(a.root, a.model_dir, a.out, a.train_from_hours, a.epochs)
        (a.root / a.out).mkdir(parents=True, exist_ok=True)
        (a.root / a.out / "training.json").write_text(json.dumps(_clean(info), indent=2))
    print(f"AE trained: {info}")
    seeds = []
    for seed in a.seeds:
        res = run_seed(a.root, seed, a.model_dir, a.out, a.machines, a.hours, a.clean_hours, a.per_type)
        seeds.append(res)
        print(f"seed {seed}: AE all recall {res['ae']['all']['recall']:.2f} at "
              f"{res['ae']['all']['false_alarms_per_day']:.2f} FA/day; corrbreak AE "
              f"{res['ae']['corrbreak']['detected']}/{res['ae']['corrbreak']['events']}, Tier 1 matched "
              f"(thr {res['tier1_matched_threshold']:g}) {res['tier1_matched']['corrbreak']['detected']}/"
              f"{res['tier1_matched']['corrbreak']['events']}, Tier 1 @20 "
              f"{res['tier1_t20']['corrbreak']['detected']}/{res['tier1_t20']['corrbreak']['events']}")
    doc = {"experiment": "E5b", "run_date": datetime.now().strftime("%d %b %Y"),
           "config": {"machines": a.machines, "hours": a.hours, "clean_hours": a.clean_hours, "per_type": a.per_type,
                      "seeds": a.seeds, "train_from_hours": a.train_from_hours, "e5_threshold": E5_THRESHOLD,
                      "model_dir": a.model_dir.as_posix(), "types": list(TYPES)},
           "training": info, "summary": summarise(seeds), "separation": separation_summary(seeds), "seeds": seeds}
    doc["runtime_s"] = time.perf_counter() - t0
    dest = a.root / DOCS if a.out == OUT else a.root / a.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{REPORT_NAME}.json").write_text(json.dumps(_clean(doc), indent=2) + "\n", encoding="utf-8")
    md = dest / f"{REPORT_NAME}.md"
    old = md.read_text(encoding="utf-8") if md.exists() else ""
    text = render(doc)
    if "## Interpretation" in old:                  # keep a hand-written interpretation across reruns
        kept = old.split("## Interpretation", 1)[1].split("## Commands", 1)[0].strip()
        text = text.replace("INTERPRETATION", kept)
    md.write_text(text, encoding="utf-8")
    print(f"report: {md}")
    print(f"total runtime {doc['runtime_s']:.1f} s ({doc['runtime_s'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
