"""E1 robustness on SMD: AE v2 seed spread, held-out AE threshold, machine bootstrap of the E1 scores.

Run from the repo root:
    python experiments/e1_robustness.py                                   # all three parts, all 28 machines
    python experiments/e1_robustness.py --sample 2 --epochs 3 --seeds 0   # quick run (report under data/robust/)
    python experiments/e1_robustness.py --parts bootstrap                 # only part 3 (no training)
    python experiments/e1_robustness.py --report-only                     # rebuild the report from saved parts

Parts:
1. seeds     AE v2 retrained with seeds 0, 1, 2 by src/detection/autoencoder.py `run` with the production settings
             (mean_max inputs, recenter 24, adaptive 288, global threshold = p99.5 of the train score x 1.1, up to
             50 epochs with early stopping); only the seed and the output folders change. Test windows are scored
             with `evaluate` from autoencoder.py, the function e1_report.py uses.
2. holdout   Same AE v2 and seeds, but per machine the train windows are split in time order: the model, the
             scaler (mean/std) and the flat-metric mask are fitted on the first 80 %, and the p99.5 x 1.1 threshold
             is taken over the scores of the last 20 % (never seen in training; centred on the train mean, as the
             train windows are in production). Main row: one global threshold over the held-out windows of all
             machines (the production mode); secondary row: one threshold per machine from its own held-out windows.
             The steps are the ones in autoencoder.py `run`, called function by function (a test checks that with
             the full train split they give exactly what `run` gives).
3. bootstrap Production scores layer (read-only, default data/scores/source=smd, must pass the e1_report.py sanity
             checks): the machines are resampled with replacement 1000 times (numpy default_rng(0)), and composite F1
             and false alarms per machine per day are recomputed per detector (KMeans tier2_flag, PCA pca_score >
             PCA threshold, AE v2 ae_flag) from per-machine counts, with the same totals as `evaluate` (faults,
             faults caught, true and false alarm windows summed over machines). Paired: every resample uses the same
             machines for every detector. 95 % interval = 2.5th and 97.5th percentiles. No retraining.

Writes only (all git-ignored except the report):
    models/robust/ae_seed<S>/, models/robust/ae_holdout_seed<S>/         model.pt, scaler.json, threshold.json, ...
    data/robust/ae_seed<S>/scores/source=smd/scores.parquet, data/robust/ae_holdout_seed<S>/...   test scores
    data/robust/logs/*.log                                               full training output per run
    data/robust/part_<name>.json                                         results of each part
    docs/results/e1_robustness.md, .json                                 full run
    data/robust/sample<N>/...                                            --sample N run (keeps docs/results clean)
Never models/ae_smd, models/kmeans_smd, models/pca_smd, data/scores or data/scores_ae.
"""
import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection import autoencoder as M  # noqa: E402
from src.detection import e1_report as R  # noqa: E402
from src.detection.evaluate import composite_metrics  # noqa: E402

SOURCE = "smd"
SEEDS = (0, 1, 2)
FIT_FRACTION = 0.8
N_BOOT = 1000
BOOT_SEED = 0
CI = (2.5, 97.5)
PARTS = ("seeds", "holdout", "bootstrap")
MODELS_OUT = Path("models/robust")
DATA_OUT = Path("data/robust")
DOCS_OUT = Path("docs/results")
E1_JSON = Path("docs/results/e1.json")
RAW_BUILD_COMMAND = "python src/detection/smd_to_parquet.py"
# bootstrap detectors: key -> (name, score column, flag column or None for "pca_score > PCA threshold")
BOOT_DETECTORS = {
    "kmeans": ("Tier 2 KMeans (recenter 24)", "tier2_score", "tier2_flag"),
    "pca": ("Tier 2b PCA (recenter 24)", "pca_score", None),
    "ae_v2": ("Tier 3 AE v2", "ae_score", "ae_flag"),
}
SUMMARY_KEYS = ("composite_f1", "faults_caught", "alarm_precision", "false_alarms_per_machine_day")


# ---------------------------------------------------------------- helpers

def raw_dir(root: Path) -> Path:
    return root / S.PATHS["raw"] / f"source={SOURCE}"


def raw_ready(root: Path) -> bool:
    d = raw_dir(root)
    return d.is_dir() and any(p.name not in M.SKIP_FILES for p in d.glob("*.parquet"))


def summary(overall: dict, n_flagged: int) -> dict:
    """The table row of one evaluated run (overall dict from autoencoder.evaluate)."""
    c = overall["composite"]
    return {"composite_f1": c["composite_f1"], "faults_caught": c["events_detected"], "faults": c["events"],
            "event_recall": c["event_recall"], "alarm_precision": c["alarm_precision"],
            "false_alarms_per_machine_day": overall["fa_day"], "windows_flagged": int(n_flagged)}


def mean_std(rows: list[dict], keys=SUMMARY_KEYS) -> dict:
    """Mean and sample std (ddof 1; 0 for a single row) of each key over the rows."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max())}
    return out


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(doc), indent=2), encoding="utf-8")


@contextlib.contextmanager
def log_to(path: Path):
    """Send stdout (the per-epoch and per-machine output of autoencoder.py) to a log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        yield


# ---------------------------------------------------------------- part 1: seeds

def run_seed(root: Path, seed: int, sample: int | None, epochs: int, out_tag: str = "") -> dict:
    """AE v2 with the production settings and this seed, written under models/robust/ and data/robust/."""
    name = f"ae_seed{seed}{out_tag}"
    with log_to(root / DATA_OUT / "logs" / f"{name}.log"):
        res = M.run(SOURCE, sample=sample, epochs=epochs, threshold_mode="global", root=root, verbose=True,
                    seed=seed, model_dir=root / MODELS_OUT / name,
                    scores_dir=root / DATA_OUT / name / "scores" / f"source={SOURCE}")
    test = (res["meta"]["split"] == "test").to_numpy()
    best = min(res["history"], key=lambda h: h["val_loss"])
    return {"seed": seed, **summary(res["overall"], res["ae_flag"][test].sum()),
            "threshold": res["thresholds"]["global"], "epochs": len(res["history"]), "best_epoch": best["epoch"]}


# ---------------------------------------------------------------- part 2: held-out threshold

def holdout_masks(meta: pd.DataFrame, fit_fraction: float = FIT_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Per machine, its train windows in time order: the first floor(fit_fraction * n) -> fit, the rest -> held out."""
    fit = np.zeros(len(meta), dtype=bool)
    held = np.zeros(len(meta), dtype=bool)
    is_train = (meta["split"] == "train").to_numpy()
    starts = meta["window_start"].to_numpy()
    for _, idx in meta.groupby("entity_id").indices.items():
        tr = idx[is_train[idx]]
        tr = tr[np.argsort(starts[tr], kind="stable")]
        n_fit = int(math.floor(fit_fraction * len(tr)))
        fit[tr[:n_fit]] = True
        held[tr[n_fit:]] = True
    return fit, held


def fit_and_score(meta: pd.DataFrame, X: np.ndarray, seed: int, fit_mask: np.ndarray, threshold_mask: np.ndarray,
                  epochs: int = M.MAX_EPOCHS, recenter: int = M.RECENTER, adaptive: int = M.ADAPTIVE,
                  inputs: str = M.DEFAULT_INPUTS) -> dict:
    """The steps of autoencoder.run with the model/scaler fitted on fit_mask windows and the thresholds taken
    over threshold_mask windows. With fit_mask = threshold_mask = all train windows this is autoencoder.run."""
    metrics = M.METRICS[SOURCE]
    aggs = M.INPUTS[inputs]
    feature_metrics = [m for _ in aggs for m in metrics]
    is_test = (meta["split"] == "test").to_numpy()
    meta_fit = meta.assign(split=np.where(is_test, "test", np.where(fit_mask, "train", "held_out")))
    scalers = M.fit_scalers(meta_fit, X, M.STD_FLOOR)
    for entity, masked in M.flat_metrics(meta_fit, X, metrics, M.FLAT_STD).items():
        scalers[entity]["masked"] = masked
    centers = M.recenter_centers(meta_fit, X, scalers, recenter) if recenter else None
    Z, mask = M.standardise(meta_fit, X, scalers, feature_metrics, centers)
    model, history = M.train(Z[fit_mask], mask[fit_mask], epochs=epochs, seed=seed, verbose=True)
    ae_score, top = M.score(model, Z, mask, metrics)
    info = M.thresholds(meta.assign(split=np.where(threshold_mask, "train", "other")), ae_score)
    limit = M.adaptive_limits(meta, ae_score, adaptive) if adaptive else None
    flags = {mode: M.flags(meta, ae_score, info, mode, limit) for mode in M.THRESHOLD_MODES}
    return {"model": model, "scalers": scalers, "history": history, "ae_score": ae_score, "top": top,
            "thresholds": info, "flags": flags, "features": M.feature_names(metrics, aggs), "inputs": inputs}


def save_holdout(root: Path, name: str, meta: pd.DataFrame, res: dict, fit_fraction: float, adaptive: int) -> None:
    model_dir = root / MODELS_OUT / name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": res["model"].state_dict(), "n_in": len(res["features"]), "hidden": list(M.HIDDEN),
                "metrics": M.METRICS[SOURCE], "inputs": res["inputs"], "features": res["features"],
                "window_minutes": M.WINDOW_MINUTES}, model_dir / "model.pt")
    (model_dir / "scaler.json").write_text(json.dumps({"entities": res["scalers"], "fit_fraction": fit_fraction}))
    (model_dir / "threshold.json").write_text(json.dumps(
        {"mode": "global", **res["thresholds"], "fitted_on": f"last {1 - fit_fraction:.0%} of train windows",
         "adaptive": {"windows": adaptive, "percentile": M.ADAPTIVE_PERCENTILE}}, indent=2))
    (model_dir / "history.json").write_text(json.dumps(res["history"], indent=2))
    test = (meta["split"] == "test").to_numpy()
    out = root / DATA_OUT / name / "scores" / f"source={SOURCE}"
    out.mkdir(parents=True, exist_ok=True)
    flag = res["flags"]["global"]
    table = M.scores_table(meta[test].reset_index(drop=True), res["ae_score"][test], flag[test],
                           [res["top"][i] for i in np.flatnonzero(test)])
    pq.write_table(table, out / "scores.parquet")


def run_holdout(root: Path, meta: pd.DataFrame, X: np.ndarray, seed: int, epochs: int,
                fit_fraction: float = FIT_FRACTION, out_tag: str = "") -> dict:
    name = f"ae_holdout_seed{seed}{out_tag}"
    fit, held = holdout_masks(meta, fit_fraction)
    with log_to(root / DATA_OUT / "logs" / f"{name}.log"):
        res = fit_and_score(meta, X, seed, fit, held, epochs=epochs)
    save_holdout(root, name, meta, res, fit_fraction, M.ADAPTIVE)
    test = (meta["split"] == "test").to_numpy()
    rows = {}
    for mode, flag in res["flags"].items():
        _, overall = M.evaluate(meta, flag, res["ae_score"])
        rows[mode] = {"seed": seed, **summary(overall, flag[test].sum())}
    per = np.array(list(res["thresholds"]["per_machine"].values()))
    best = min(res["history"], key=lambda h: h["val_loss"])
    rows["global"].update({"threshold": res["thresholds"]["global"], "epochs": len(res["history"]),
                           "best_epoch": best["epoch"], "fit_windows": int(fit.sum()),
                           "held_out_windows": int(held.sum())})
    rows["per-machine"].update({"threshold_median": float(np.median(per)), "threshold_min": float(per.min()),
                                "threshold_max": float(per.max())})
    return rows


# ---------------------------------------------------------------- part 3: bootstrap

def detector_flags(df: pd.DataFrame, pca_thr: float) -> dict[str, np.ndarray]:
    out = {}
    for key, (_, score_col, flag_col) in BOOT_DETECTORS.items():
        if flag_col is None:
            out[key] = np.nan_to_num(df[score_col].to_numpy(dtype=np.float64), nan=-np.inf) > pca_thr
        else:
            out[key] = R._as_flag(df[flag_col])
    return out


def machine_counts(df: pd.DataFrame, flag: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Per machine (time-ordered windows): faults, faults caught, true alarms, false alarms, windows."""
    d = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(), "window_start": df["window_start"].to_numpy(),
                      "t": df["label"].astype(int).to_numpy() != 0, "p": flag})
    machines, rows = [], []
    for entity, g in d.sort_values(["entity_id", "window_start"]).groupby("entity_id", sort=True):
        t, p = g["t"].to_numpy(), g["p"].to_numpy()
        c = composite_metrics(t, p)
        rows.append([c["events"], c["events_detected"], int(np.sum(t & p)), int(np.sum(~t & p)), len(t)])
        machines.append(entity)
    return machines, np.array(rows, dtype=np.int64)


def from_counts(c: np.ndarray, window_minutes: int = M.WINDOW_MINUTES) -> dict:
    """Composite F1 and false alarms per machine per day from summed counts; c is (..., 5) per-machine counts
    summed over the machine axis already: faults, caught, tp, fp, windows."""
    ev, det, tp, fp, n = (c[..., i].astype(float) for i in range(5))
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(ev > 0, det / ev, 0.0)
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(rec + prec > 0, 2 * rec * prec / (rec + prec), 0.0)
        fa = np.where(n > 0, fp / (n * window_minutes / (24 * 60)), np.nan)
    return {"composite_f1": f1, "event_recall": rec, "alarm_precision": prec, "false_alarms_per_machine_day": fa}


def bootstrap(counts: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Paired machine bootstrap: one (n_boot x n_machines) index draw shared by every detector."""
    n_machines = len(next(iter(counts.values())))
    idx = np.random.default_rng(seed).integers(0, n_machines, size=(n_boot, n_machines))
    return {key: from_counts(c[idx].sum(axis=1)) for key, c in counts.items()}


def _interval(v: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(v, CI)]


def _rel(path: Path, root: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(root.resolve()).as_posix() if path.is_relative_to(root.resolve()) else path.as_posix()


def run_bootstrap(root: Path, scores_dir: Path, pca_thresholds: Path | None, n_boot: int, seed: int,
                  expected_windows: int | None, expected_machines: int | None) -> dict:
    df = R.load_scores(scores_dir)
    if expected_windows is not None:
        with log_to(root / DATA_OUT / "logs" / "bootstrap_sanity.log"):
            R.sanity_checks(df, expected_windows, expected_machines)
    if pca_thresholds is not None and pca_thresholds.exists():
        pca_thr, pca_src = R.pca_threshold(pca_thresholds), _rel(pca_thresholds, root)
    else:
        pca_thr, pca_src = float(json.loads((root / E1_JSON).read_text())["pca_threshold"]), E1_JSON.as_posix()
    flags = detector_flags(df, pca_thr)
    counts, machines = {}, None
    for key, flag in flags.items():
        machines, counts[key] = machine_counts(df, flag)
    point = {key: {k: float(v) for k, v in from_counts(c.sum(axis=0)).items()} for key, c in counts.items()}
    for key, c in counts.items():
        point[key].update({"faults_caught": int(c[:, 1].sum()), "faults": int(c[:, 0].sum()),
                           "windows_flagged": int(c[:, 2].sum() + c[:, 3].sum())})
    boot = bootstrap(counts, n_boot, seed)

    e1 = {r["key"]: r for r in json.loads((root / E1_JSON).read_text())["detectors"]} if (root / E1_JSON).exists() else {}
    detectors = {}
    for key, (name, _, _) in BOOT_DETECTORS.items():
        b = boot[key]
        ref = e1.get(key)
        matches = (ref is not None and ref["faults_caught"] == point[key]["faults_caught"]
                   and ref["windows_flagged"] == point[key]["windows_flagged"]
                   and abs(ref["composite_f1"] - point[key]["composite_f1"]) < 5e-4)
        detectors[key] = {"name": name, "point": point[key], "matches_e1": bool(matches),
                          "composite_f1_ci": _interval(b["composite_f1"]),
                          "composite_f1_boot_mean": float(np.mean(b["composite_f1"])),
                          "false_alarms_ci": _interval(b["false_alarms_per_machine_day"])}
    pairs = {}
    for a, c in (("kmeans", "pca"), ("kmeans", "ae_v2"), ("pca", "ae_v2")):
        d = boot[a]["composite_f1"] - boot[c]["composite_f1"]
        fa = boot[a]["false_alarms_per_machine_day"] - boot[c]["false_alarms_per_machine_day"]
        pairs[f"{a}-{c}"] = {"f1_diff_point": point[a]["composite_f1"] - point[c]["composite_f1"],
                             "f1_diff_ci": _interval(d), "share_first_higher_f1": float(np.mean(d > 0)),
                             "fa_diff_point": (point[a]["false_alarms_per_machine_day"]
                                               - point[c]["false_alarms_per_machine_day"]),
                             "fa_diff_ci": _interval(fa)}
    f = {k: boot[k]["composite_f1"] for k in boot}
    fa = {k: boot[k]["false_alarms_per_machine_day"] for k in boot}
    return {"scores_dir": _rel(scores_dir, root),
            "windows": len(df), "machines": len(machines), "pca_threshold": pca_thr, "pca_threshold_source": pca_src,
            "n_boot": n_boot, "seed": seed, "detectors": detectors, "pairs": pairs,
            "share_ranking_kmeans_pca_ae": float(np.mean((f["kmeans"] > f["pca"]) & (f["pca"] > f["ae_v2"]))),
            "share_ae_fewest_false_alarms": float(np.mean((fa["ae_v2"] < fa["kmeans"]) & (fa["ae_v2"] < fa["pca"])))}


# ---------------------------------------------------------------- report

def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def _ms(stat: dict, fmt="{:.3f}") -> str:
    return f"{fmt.format(stat['mean'])} ± {fmt.format(stat['std'])}"


ROW_HEADER = ("| run | composite F1 | faults caught | alarm precision | false alarms / machine / day | windows flagged "
              "| threshold | epochs (best) |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


def _row(label: str, r: dict, thr_key: str = "threshold") -> str:
    epochs = f"{r['epochs']} ({r['best_epoch']})" if "epochs" in r else "–"
    thr = _f(r.get(thr_key), "{:.4f}") if thr_key in r else "–"
    return (f"| {label} | {r['composite_f1']:.3f} | {r['faults_caught']}/{r['faults']} | {r['alarm_precision']:.3f} "
            f"| {r['false_alarms_per_machine_day']:.2f} | {r['windows_flagged']:,} | {thr} | {epochs} |")


def _mean_row(label: str, stats: dict) -> str:
    return (f"| **{label}** | **{_ms(stats['composite_f1'])}** | {_ms(stats['faults_caught'], '{:.1f}')} "
            f"| {_ms(stats['alarm_precision'])} | {_ms(stats['false_alarms_per_machine_day'], '{:.2f}')} | | | |")


def seed_table(rows: list[dict], e1_ae: dict | None, thr_key: str = "threshold") -> str:
    lines = [ROW_HEADER]
    if e1_ae:
        lines.append(f"| E1 production AE v2 (seed 42, quoted) | {e1_ae['composite_f1']:.3f} "
                     f"| {e1_ae['faults_caught']}/{e1_ae['faults']} | {e1_ae['alarm_precision']:.3f} "
                     f"| {e1_ae['false_alarms_per_machine_day']:.2f} | {e1_ae['windows_flagged']:,} | – | – |")
    lines += [_row(f"seed {r['seed']}", r, thr_key) for r in rows]
    lines.append(_mean_row(f"mean ± std over {len(rows)} seeds", mean_std(rows)))
    return "\n".join(lines)


def bootstrap_table(b: dict) -> str:
    lines = ["| detector | composite F1 (point) | 95 % interval | false alarms / machine / day (point) | 95 % interval "
             "| point estimate equals E1 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for d in b["detectors"].values():
        p = d["point"]
        lo, hi = d["composite_f1_ci"]
        flo, fhi = d["false_alarms_ci"]
        lines.append(f"| {d['name']} | {p['composite_f1']:.3f} | {lo:.3f} – {hi:.3f} "
                     f"| {p['false_alarms_per_machine_day']:.2f} | {flo:.2f} – {fhi:.2f} "
                     f"| {'yes' if d['matches_e1'] else 'no'} |")
    return "\n".join(lines)


def pairs_table(b: dict) -> str:
    names = {k: v[0] for k, v in BOOT_DETECTORS.items()}
    lines = ["| pair (A − B) | composite F1 difference | 95 % interval | resamples with A > B "
             "| false alarms / machine / day difference | 95 % interval |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, p in b["pairs"].items():
        a, c = key.split("-")
        lo, hi = p["f1_diff_ci"]
        flo, fhi = p["fa_diff_ci"]
        lines.append(f"| {names[a]} − {names[c]} | {p['f1_diff_point']:+.3f} | {lo:+.3f} – {hi:+.3f} "
                     f"| {p['share_first_higher_f1']:.1%} | {p['fa_diff_point']:+.2f} | {flo:+.2f} – {fhi:+.2f} |")
    return "\n".join(lines)


def interpretation(doc: dict, e1: dict) -> list[str]:
    """Plain statements generated from the numbers (each part only if it was run)."""
    out = []
    km, pca, ae = e1.get("kmeans"), e1.get("pca"), e1.get("ae_v2")
    for part, label in (("seeds", "Across seeds"), ("holdout", "With the held-out threshold")):
        p = doc.get(part)
        if not p or not km or not pca or not ae:
            continue
        rows = p["rows"] if part == "seeds" else p["global"]
        st = mean_std(rows)
        f1, fa = st["composite_f1"], st["false_alarms_per_machine_day"]
        below = f1["max"] < pca["composite_f1"]
        fewer = fa["max"] < min(km["false_alarms_per_machine_day"], pca["false_alarms_per_machine_day"])
        out.append(
            f"**{label}** ({len(rows)} seeds) AE v2 composite F1 is {_ms(f1)} (range {f1['min']:.3f}–{f1['max']:.3f}) "
            f"against E1's {ae['composite_f1']:.3f}, and false alarms are {_ms(fa, '{:.2f}')} per machine per day (max {fa['max']:.2f}). "
            + ("Every run stays below E1's PCA (" if below else "At least one run reaches E1's PCA (")
            + f"{pca['composite_f1']:.3f}) and KMeans ({km['composite_f1']:.3f}), "
            + (("so the AE's place at the bottom of the E1 ranking does not depend on the seed" if part == "seeds"
                else "so moving the threshold to held-out data does not lift the AE above PCA or KMeans") if below
               else "so the AE-vs-PCA order is not stable here")
            + "; " + ("every run also keeps fewer false alarms than KMeans "
                      f"({km['false_alarms_per_machine_day']:.2f}) and PCA ({pca['false_alarms_per_machine_day']:.2f})."
                      if fewer else "not every run keeps fewer false alarms than KMeans and PCA.")
            + (" KMeans and PCA were not retrained here (their seed-42 E1 numbers are the comparison)."
               if part == "seeds" else
               " Only the AE threshold moved to held-out data; KMeans and PCA keep their E1 thresholds."))
    h = doc.get("holdout")
    if h:
        thr = np.array([r["threshold"] for r in h["global"]])
        ref = doc.get("seeds")
        ref_thr = np.array([r["threshold"] for r in ref["rows"]]) if ref else None
        out.append(f"The global held-out threshold is {thr.min():.2f}–{thr.max():.2f}"
                   + (f", {thr.mean() / ref_thr.mean():.1f}x the threshold from all train windows "
                      f"({ref_thr.min():.2f}–{ref_thr.max():.2f} in part 1)" if ref_thr is not None else "")
                   + ": the last 20 % of each machine's train series scores higher than the first 80 % the model "
                     "was fitted on (the held-out windows are centred on the train mean, like train windows in "
                     "production, so slow drift inside the train period raises their error). The held-out alarm "
                     "is therefore stricter: it catches fewer faults and raises fewer false alarms than part 1.")
        pm = mean_std(h["per-machine"])
        out.append(f"With **per-machine** held-out thresholds AE v2 gets composite F1 {_ms(pm['composite_f1'])} at "
                   f"{_ms(pm['false_alarms_per_machine_day'], '{:.2f}')} false alarms per machine per day "
                   "(secondary; production uses one global threshold).")
    b = doc.get("bootstrap")
    if b:
        d, pr = b["detectors"], b["pairs"]
        same = all(x["matches_e1"] for x in d.values())
        out.append(
            "**Bootstrap over machines** ("
            + ("the point estimates equal the E1 table" if same
               else "point estimates equal E1 for " + (", ".join(x["name"] for x in d.values() if x["matches_e1"])
                                                       or "no detector")
               + "; not for " + ", ".join(x["name"] for x in d.values() if not x["matches_e1"])
               + ", see the note in part 3")
            + f"): KMeans {d['kmeans']['composite_f1_ci'][0]:.3f}–{d['kmeans']['composite_f1_ci'][1]:.3f}, "
              f"PCA {d['pca']['composite_f1_ci'][0]:.3f}–{d['pca']['composite_f1_ci'][1]:.3f}, "
              f"AE v2 {d['ae_v2']['composite_f1_ci'][0]:.3f}–{d['ae_v2']['composite_f1_ci'][1]:.3f} (95 %). "
            + f"The full order KMeans > PCA > AE v2 holds in {b['share_ranking_kmeans_pca_ae']:.1%} of resamples. "
            + "; ".join(_pair_sentence(k, p) for k, p in pr.items()) + ". "
            + f"AE v2 has the fewest false alarms in {b['share_ae_fewest_false_alarms']:.1%} of resamples "
              f"(KMeans minus AE: {pr['kmeans-ae_v2']['fa_diff_ci'][0]:+.2f} to "
              f"{pr['kmeans-ae_v2']['fa_diff_ci'][1]:+.2f}; PCA minus AE: "
              f"{pr['pca-ae_v2']['fa_diff_ci'][0]:+.2f} to {pr['pca-ae_v2']['fa_diff_ci'][1]:+.2f} "
              "false alarms per machine per day, 95 %).")
    return out


def _pair_sentence(key: str, p: dict) -> str:
    names = {"kmeans": "KMeans", "pca": "PCA", "ae_v2": "AE v2"}
    a, c = (names[x] for x in key.split("-"))
    lo, hi = p["f1_diff_ci"]
    sep = "separated" if lo > 0 or hi < 0 else "not separated (interval contains 0)"
    return f"{a} − {c} = {p['f1_diff_point']:+.3f} [{lo:+.3f}, {hi:+.3f}], {sep}"


def _retrained_rows(doc: dict) -> list[dict]:
    rows = list(doc["seeds"]["rows"]) if doc.get("seeds") else []
    return rows + (list(doc["holdout"]["global"]) if doc.get("holdout") else [])


def verdict(doc: dict, e1: dict) -> str:
    """The two questions (does the E1 ranking survive, does the AE's false-alarm advantage survive), answered from
    the retrained runs (parts 1 and 2, against the E1 KMeans/PCA rows) and the bootstrap (part 3)."""
    b, runs = doc.get("bootstrap"), _retrained_rows(doc)
    km, pca = e1.get("kmeans"), e1.get("pca")
    if not b or not runs or not km or not pca:
        return "_Needs all three parts and docs/results/e1.json._"
    pr = b["pairs"]
    names = {"kmeans": "KMeans", "pca": "PCA", "ae_v2": "AE v2"}
    f1_max = max(r["composite_f1"] for r in runs)
    fa_max = max(r["false_alarms_per_machine_day"] for r in runs)
    runs_below = f1_max < pca["composite_f1"]
    separated = {k: p["f1_diff_ci"][0] > 0 for k, p in pr.items()}
    rank_answer = ("Yes" if runs_below and all(separated.values()) else "Partly" if runs_below else "No")
    shares = ", ".join(f"{names[k.split('-')[0]]} > {names[k.split('-')[1]]} in {p['share_first_higher_f1']:.1%}"
                       for k, p in pr.items())
    not_sep = [f"{names[k.split('-')[0]]} − {names[k.split('-')[1]]}" for k, ok in separated.items() if not ok]
    rank = (f"**Does the E1 ranking survive? {rank_answer}.** "
            + (f"All {len(runs)} retrained AE v2 runs (parts 1 and 2) stay below E1's PCA "
               f"({pca['composite_f1']:.3f}) and KMeans ({km['composite_f1']:.3f}), best {f1_max:.3f}. "
               if runs_below else f"At least one retrained AE v2 run reaches E1's PCA ({f1_max:.3f}). ")
            + f"Over machines the point order holds in most resamples ({shares}; full order "
              f"{b['share_ranking_kmeans_pca_ae']:.1%}), "
            + ("and every difference has a 95 % interval above 0." if not not_sep else
               "but the 95 % interval" + ("s" if len(not_sep) > 1 else "") + " of " + ", ".join(not_sep)
               + (" contain" if len(not_sep) > 1 else " contains") + " 0: KMeans > PCA > AE v2 is the "
               "likely order on SMD, not a settled one, and the gaps are of the same size as the spread between "
               "machines."))
    fa_ok = (fa_max < min(km["false_alarms_per_machine_day"], pca["false_alarms_per_machine_day"])
             and pr["kmeans-ae_v2"]["fa_diff_ci"][0] > 0 and pr["pca-ae_v2"]["fa_diff_ci"][0] > 0)
    fa = (f"**Does the AE's false-alarm advantage survive? {'Yes' if fa_ok else 'No'}.** "
          f"Every retrained run stays at or below {fa_max:.2f} false alarms per machine per day against E1's "
          f"{km['false_alarms_per_machine_day']:.2f} (KMeans) and {pca['false_alarms_per_machine_day']:.2f} (PCA); "
          f"over machines AE v2 has the fewest false alarms in {b['share_ae_fewest_false_alarms']:.1%} of resamples "
          f"and the 95 % intervals of KMeans − AE ({pr['kmeans-ae_v2']['fa_diff_ci'][0]:+.2f} to "
          f"{pr['kmeans-ae_v2']['fa_diff_ci'][1]:+.2f}) and PCA − AE ({pr['pca-ae_v2']['fa_diff_ci'][0]:+.2f} to "
          f"{pr['pca-ae_v2']['fa_diff_ci'][1]:+.2f}) "
          + ("exclude 0." if fa_ok else "do not both exclude 0."))
    return f"{rank}{chr(10) * 2}{fa}"


def _runtimes(doc: dict) -> str:
    got = {p: doc[p]["runtime_min"] for p in PARTS if p in doc and doc[p].get("runtime_min") is not None}
    if not got:
        return "not recorded"
    return ", ".join(f"{p} {m:.1f} min" for p, m in got.items()) + f"; total {sum(got.values()):.1f} min"


def write_report(out_dir: Path, doc: dict, e1_path: Path) -> None:
    e1 = {r["key"]: r for r in json.loads(e1_path.read_text())["detectors"]} if e1_path.exists() else {}
    e1_ae = e1.get("ae_v2")
    s, h, b = doc.get("seeds"), doc.get("holdout"), doc.get("bootstrap")
    parts = [f"""# E1 robustness: seeds, held-out threshold, machine bootstrap (SMD)

Generated by `python experiments/e1_robustness.py`. E1 itself (`docs/results/e1.md`), the production models and
`data/scores` are unchanged; everything retrained here lives under `models/robust/` and `data/robust/`.
Runtime: {_runtimes(doc)}.

## Interpretation

{verdict(doc, e1)}

""" + "\n\n".join(f"- {x}" for x in interpretation(doc, e1)) + """

Caveats: three seeds give a rough spread only (std over 3 values); the bootstrap treats the 28 machines as the
sampling unit and keeps each machine's test series whole, so it measures how much the result depends on which
machines are in SMD, not on which faults; it uses the fixed E1 models and thresholds, so it does not include
training noise (that is what parts 1 and 2 add). All on one dataset (SMD).
"""]
    parts.append("""## 1. Seeds (AE v2, production settings)

`src/detection/autoencoder.py` `run` with mean_max inputs, recenter 24, adaptive 288, global threshold p99.5 of the
train score x 1.1, up to 50 epochs with early stopping (patience 5); only `seed` changes (production = 42). Scored on
the SMD test windows with the `evaluate` function e1_report.py uses.

""" + (seed_table(s["rows"], e1_ae) if s else "_Not run._"))
    if h:
        g0 = h["global"][0]
        hold_text = (f"Per machine the train windows are split in time order: the first {h['fit_fraction']:.0%} "
                     f"({g0['fit_windows']:,} windows over all machines) fit the scaler, the flat-metric mask and the "
                     f"model; the threshold (p99.5 x 1.1) is taken over the AE scores of the last "
                     f"{1 - h['fit_fraction']:.0%} ({g0['held_out_windows']:,} windows), which training never saw. "
                     "Recenter 24 and adaptive 288 as in production.\n\n**Global held-out threshold (production mode)**"
                     "\n\n" + seed_table(h["global"], e1_ae) + "\n\n**Per-machine held-out threshold (secondary)**\n\n"
                     + seed_table(h["per-machine"], None, thr_key="threshold_median")
                     + "\n\n(threshold column = median of the 28 per-machine thresholds.)")
    else:
        hold_text = "_Not run._"
    parts.append("## 2. Held-out threshold (AE v2)\n\n" + hold_text)
    if b:
        boot_text = (f"Scores layer `{b['scores_dir']}` ({b['windows']:,} test windows, {b['machines']} machines; "
                     f"PCA threshold {b['pca_threshold']:.5f} from `{b['pca_threshold_source']}`). The machines are "
                     f"resampled with replacement {b['n_boot']} times (numpy `default_rng({b['seed']})`), the same "
                     "draw for every detector (paired), and composite F1 and false alarms per machine per day are "
                     "recomputed from the summed per-machine counts, which gives exactly the `evaluate` totals for "
                     "the original 28 machines. 95 % interval = 2.5th–97.5th percentile.\n\n"
                     + ("" if b["scores_dir"] == f"{S.PATHS['scores']}/source={SOURCE}" else
                        f"**Note.** This is not the production `{S.PATHS['scores']}/source={SOURCE}` (it was not on the "
                        "machine that ran this). It is a rebuild: `python src/detection/baselines.py --recenter 24 "
                        "--root data/robust/e1_rebuild` (KMeans and PCA re-fitted, deterministic) with the AE columns "
                        "from that machine's `models/ae_smd` / `data/scores_ae`; the last column says which rows "
                        "equal E1 exactly.\n\n")
                     + bootstrap_table(b) + "\n\n**Paired differences**\n\n" + pairs_table(b)
                     + f"\n\nFull order KMeans > PCA > AE v2 in {b['share_ranking_kmeans_pca_ae']:.1%} of resamples; "
                       f"AE v2 lowest false alarms in {b['share_ae_fewest_false_alarms']:.1%}.")
    else:
        boot_text = "_Not run._"
    parts.append("## 3. Bootstrap over machines (E1 scores, no retraining)\n\n" + boot_text)
    parts.append("""## Reproduce

```
python experiments/e1_robustness.py                    # parts 1-3, writes docs/results/e1_robustness.md/.json
python experiments/e1_robustness.py --parts bootstrap  # part 3 only (reads data/scores/source=smd)
python experiments/e1_robustness.py --report-only      # rebuild the report from data/robust/part_*.json
```
""")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_robustness.md").write_text("\n\n".join(parts), encoding="utf-8")
    write_json(out_dir / "e1_robustness.json", {"experiment": "E1 robustness", "source": SOURCE, **doc})


# ---------------------------------------------------------------- driver

def part_path(root: Path, sample: int | None, part: str) -> Path:
    base = root / DATA_OUT / (f"sample{sample}" if sample else "")
    return base / f"part_{part}.json"


def load_parts(root: Path, sample: int | None) -> dict:
    doc = {}
    for part in PARTS:
        p = part_path(root, sample, part)
        if p.exists():
            doc[part] = json.loads(p.read_text())
    return doc


def run(root: Path = ROOT, parts=PARTS, seeds=SEEDS, sample: int | None = None, epochs: int = M.MAX_EPOCHS,
        scores_dir: Path | None = None, pca_thresholds: Path | None = None, n_boot: int = N_BOOT,
        boot_seed: int = BOOT_SEED, fit_fraction: float = FIT_FRACTION, report_only: bool = False,
        out_dir: Path | None = None) -> dict | None:
    t0 = time.perf_counter()
    if scores_dir is not None and not Path(scores_dir).is_absolute():
        scores_dir = root / scores_dir
    if pca_thresholds is not None and not Path(pca_thresholds).is_absolute():
        pca_thresholds = root / pca_thresholds
    tag = f"_sample{sample}" if sample else ""
    if not report_only and {"seeds", "holdout"} & set(parts) and not raw_ready(root):
        print(f"{raw_dir(root).relative_to(root).as_posix()} has no machine Parquet files on this laptop. "
              f"Build it first with:\n    {RAW_BUILD_COMMAND}")
        return None
    if not report_only:
        if "seeds" in parts:
            t_part = time.perf_counter()
            rows = []
            for seed in seeds:
                t = time.perf_counter()
                r = run_seed(root, seed, sample, epochs, tag)
                rows.append(r)
                print(f"[seeds]   seed {seed}: composite F1 {r['composite_f1']:.3f}, caught "
                      f"{r['faults_caught']}/{r['faults']}, alarm precision {r['alarm_precision']:.3f}, "
                      f"{r['false_alarms_per_machine_day']:.2f} false alarms/machine/day, {r['epochs']} epochs "
                      f"({(time.perf_counter() - t) / 60:.1f} min)", flush=True)
            write_json(part_path(root, sample, "seeds"), {"seeds": list(seeds), "rows": rows,
                                                          "runtime_min": (time.perf_counter() - t_part) / 60})
        if "holdout" in parts:
            t_part = time.perf_counter()
            meta, X = M.load_windows(raw_dir(root), M.METRICS[SOURCE], sample, M.INPUTS[M.DEFAULT_INPUTS])
            g_rows, pm_rows = [], []
            for seed in seeds:
                t = time.perf_counter()
                r = run_holdout(root, meta, X, seed, epochs, fit_fraction, tag)
                g_rows.append(r["global"])
                pm_rows.append(r["per-machine"])
                g = r["global"]
                print(f"[holdout] seed {seed}: composite F1 {g['composite_f1']:.3f}, caught "
                      f"{g['faults_caught']}/{g['faults']}, alarm precision {g['alarm_precision']:.3f}, "
                      f"{g['false_alarms_per_machine_day']:.2f} false alarms/machine/day, held-out threshold "
                      f"{g['threshold']:.4f} ({(time.perf_counter() - t) / 60:.1f} min)", flush=True)
            write_json(part_path(root, sample, "holdout"),
                       {"seeds": list(seeds), "fit_fraction": fit_fraction, "global": g_rows, "per-machine": pm_rows,
                        "runtime_min": (time.perf_counter() - t_part) / 60})
        if "bootstrap" in parts:
            t = time.perf_counter()
            sd = scores_dir or root / S.PATHS["scores"] / f"source={SOURCE}"
            exp_w, exp_m = (None, None) if sample else (R.EXPECTED_WINDOWS, R.EXPECTED_MACHINES)
            b = run_bootstrap(root, sd, pca_thresholds or root / R.PCA_THRESHOLDS, n_boot, boot_seed, exp_w, exp_m)
            b["runtime_min"] = (time.perf_counter() - t) / 60
            write_json(part_path(root, sample, "bootstrap"), b)
            d = b["detectors"]
            print(f"[bootstrap] {n_boot} resamples of {b['machines']} machines: "
                  + ", ".join(f"{k} {v['point']['composite_f1']:.3f} [{v['composite_f1_ci'][0]:.3f}, "
                              f"{v['composite_f1_ci'][1]:.3f}]" for k, v in d.items())
                  + f" ({(time.perf_counter() - t) / 60:.1f} min)", flush=True)
    doc = load_parts(root, sample)
    runtime = (time.perf_counter() - t0) / 60
    out = out_dir or (root / DATA_OUT / f"sample{sample}" if sample else root / DOCS_OUT)
    write_report(out, doc, root / E1_JSON)
    print(f"wrote {(out / 'e1_robustness.md').as_posix()} and e1_robustness.json")
    print(f"total runtime {runtime:.1f} min")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--sample", type=int, default=None, help="use only the first N machines (quick run)")
    parser.add_argument("--epochs", type=int, default=M.MAX_EPOCHS, help="maximum epochs per AE")
    parser.add_argument("--scores-dir", type=Path, default=None,
                        help="scores layer for the bootstrap (default data/scores/source=smd, read-only)")
    parser.add_argument("--pca-thresholds", type=Path, default=None,
                        help="PCA thresholds.json (default models/pca_smd/thresholds.json; if missing, the "
                             "pca_threshold recorded in docs/results/e1.json)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from data/robust/part_*.json")
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/ and models/")
    args = parser.parse_args()
    try:
        doc = run(args.root, tuple(args.parts), tuple(args.seeds), args.sample, args.epochs, args.scores_dir,
                  args.pca_thresholds, args.n_boot, args.boot_seed, report_only=args.report_only)
    except R.SanityError as e:
        print(f"E1 robustness FAILED: {e}", file=sys.stderr)
        return 1
    return 0 if doc is not None else 1


if __name__ == "__main__":
    sys.exit(main())
