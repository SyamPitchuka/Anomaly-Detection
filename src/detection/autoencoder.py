"""Tier 3 detector: dense autoencoder on 5-minute windows (offline training, PyTorch CPU, no Spark).

Run from the repo root:
    python src/detection/autoencoder.py                 # all SMD machines, v2 defaults (row 9 of docs/ae_ablation.md)
    python src/detection/autoencoder.py --sample 2      # first 2 machines, quick test
    python src/detection/autoencoder.py --epochs 10
    python src/detection/autoencoder.py --flat-std 0 --inputs mean --recenter 0 --adaptive 0   # v1, the 0.360 run
    python src/detection/autoencoder.py --threshold per-machine --std-floor 0.05       # variant (b), docs/ae_ablation.md

Steps:
1. Read data/raw/source=smd/*.parquet (fake.parquet is skipped) and build 5-minute tumbling windows per
   entity_id: window_start = event_time floored to 5 minutes, window_end = window_start + 5 minutes,
   value = mean of each metric m00..m37 over the rows that have it (missing values are skipped); with
   --inputs mean_max (default) also the max, so there are 76 inputs (all means, then all maxes).
   A window is `test` if any of its rows is a test row (only the window at the train/test boundary can
   mix both), else `train`. Test windows get label = 1 if any row has label 1, else 0; train windows null.
2. Standardise per entity_id with the mean/std of its train windows (std 0 -> 1, then std raised to at
   least --std-floor, default 0 = no floor); saved to scaler.json.
   A metric missing from a window is fed to the model as 0 (the train mean) and left out of its error.
   --flat-std (default 0 = off, e.g. 0.01): per machine, metrics whose train-window std of the mean is below this are
   masked: fed as 0 everywhere, left out of the error and of top_metrics; listed per machine in scaler.json.
   --recenter N (default 24): test windows are centred on the median of the machine's previous N windows
   (train or test; current window excluded; at least N // 2 values, else the train mean) instead of the
   train mean; std stays from train. No labels are used. Saved in scaler.json for the streaming job.
3. One 38-32-8-32-38 ReLU autoencoder (76-32-8-32-76 with mean_max) on the train windows of all machines:
   MSE, Adam lr 1e-3, batch 256, up to 50 epochs (--epochs), early stopping (patience 5) on a random 10%
   of train windows, seed 42.
4. ae_score = mean squared reconstruction error of a window; threshold = 99.5th percentile of the train
   ae_score x 1.1, taken over all machines (--threshold global, default) or per machine (per-machine);
   both kinds are saved in threshold.json. ae_flag = ae_score > threshold of the window's machine.
   --adaptive N (default 288 = one day; 0 = off): also require ae_score > the 99th percentile of the
   machine's previous N scores (at least N // 2 of them, else this test is skipped).
   top_metrics = 3 metrics with the largest squared error (mean + max error with --inputs mean_max).
5. Test windows are written to data/scores_ae/source=smd/scores.parquet with the scores-layer schema
   (tier1/tier2/pca columns null, is_anomaly = ae_flag).
6. From src/detection/evaluate.py, per machine on time-ordered test windows and overall: composite F1
   (event recall with alarm precision), faults caught, alarm precision, false alarms per machine per day,
   median detection delay in minutes, PR-AUC of ae_score, and point-adjusted event F1 (reference, generous).

Outputs: models/ae_smd/model.pt, scaler.json, threshold.json, history.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection.evaluate import (  # noqa: E402
    composite_metrics,
    detection_delays,
    event_metrics,
    false_alarms_per_day,
    point_metrics,
    pr_auc,
)

METRICS = {"smd": S.SMD_METRICS}
WINDOW_US = 5 * 60 * 1_000_000
WINDOW_MINUTES = WINDOW_US // 60_000_000
SCORES_AE_DIR = "data/scores_ae"
SKIP_FILES = {"fake.parquet"}

SEED = 42
HIDDEN = (32, 8)
LR = 1e-3
BATCH = 256
MAX_EPOCHS = 50
STD_FLOOR = 0.0
THRESHOLD_MODES = ("global", "per-machine")
PATIENCE = 5
VAL_FRACTION = 0.10
THRESHOLD_PERCENTILE = 99.5
THRESHOLD_FACTOR = 1.1
TOP_K = 3
FLAT_STD = 0.0
INPUTS = {"mean": ("mean",), "mean_max": ("mean", "max")}
DEFAULT_INPUTS = "mean_max"
RECENTER = 24
ADAPTIVE = 288
ADAPTIVE_PERCENTILE = 99


# ---------------------------------------------------------------- windows

def build_windows(table: pa.Table, metrics: list[str],
                  aggs: tuple[str, ...] = ("mean",)) -> tuple[pd.DataFrame, np.ndarray]:
    """5-minute window aggregates (mean, and max if asked) of one or more machines' raw rows.

    Returns (meta, X): meta has entity_id, window_start (int64 microseconds UTC), split, label
    (float, NaN = null); X is n_windows x (len(aggs) * len(metrics)), one block of metrics per aggregate,
    with NaN where a metric has no value in the window. Rows are sorted by entity_id, window_start.
    """
    n = table.num_rows
    event_us = table["event_time"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy()

    # metrics map -> dense n x k matrix, missing keys stay NaN
    values = np.full((n, len(metrics)), np.nan)
    m = table["metrics"].combine_chunks()
    offsets = m.offsets.to_numpy()  # honours the array offset; keys/items are the unsliced children
    first, count = int(offsets[0]), int(offsets[-1] - offsets[0])
    rows = np.repeat(np.arange(n), np.diff(offsets))
    keys = m.keys.slice(first, count).to_numpy(zero_copy_only=False)
    vals = m.items.slice(first, count).to_numpy(zero_copy_only=False).astype(np.float64)
    cols = pd.Categorical(keys, categories=metrics).codes
    known = (cols >= 0) & ~np.isnan(vals)
    values[rows[known], cols[known]] = vals[known]

    split = pc.map_lookup(table["attrs"].combine_chunks(), "split", "first").to_numpy(zero_copy_only=False)
    df = pd.DataFrame(values, columns=metrics)
    df["entity_id"] = table["entity_id"].to_numpy(zero_copy_only=False)
    df["window_start"] = event_us - np.mod(event_us, WINDOW_US)
    df["is_test"] = split == "test"
    df["label"] = table["label"].to_pandas().astype("float64")

    g = df.groupby(["entity_id", "window_start"], sort=True)
    X = np.hstack([getattr(g[metrics], agg)().to_numpy() for agg in aggs])
    meta = g.agg(is_test=("is_test", "max"), label=("label", "max")).reset_index()
    meta["split"] = np.where(meta.pop("is_test"), "test", "train")
    meta.loc[meta["split"] == "train", "label"] = np.nan
    meta.loc[(meta["split"] == "test") & meta["label"].isna(), "label"] = 0.0
    return meta, X


def feature_names(metrics: list[str], aggs: tuple[str, ...]) -> list[str]:
    """Model input names: the metric names for means only, else <metric>_<agg>."""
    if tuple(aggs) == ("mean",):
        return list(metrics)
    return [f"{m}_{agg}" for agg in aggs for m in metrics]


def load_windows(raw_dir: Path, metrics: list[str], sample: int | None,
                 aggs: tuple[str, ...] = ("mean",)) -> tuple[pd.DataFrame, np.ndarray]:
    files = sorted(p for p in raw_dir.glob("*.parquet") if p.name not in SKIP_FILES)
    if sample:
        files = files[:sample]
    if not files:
        raise FileNotFoundError(f"no machine Parquet files under {raw_dir}")
    metas, xs = [], []
    for path in files:
        meta, X = build_windows(pq.read_table(path, schema=S.arrow_schema("raw")), metrics, aggs)
        metas.append(meta)
        xs.append(X)
    return pd.concat(metas, ignore_index=True), np.vstack(xs)


# ---------------------------------------------------------------- scaling

def fit_scalers(meta: pd.DataFrame, X: np.ndarray, std_floor: float = STD_FLOOR) -> dict[str, dict[str, list[float]]]:
    """Per-entity mean/std over train windows; std 0 (or undefined) -> 1, then std >= std_floor;
    undefined mean -> 0."""
    scalers = {}
    is_train = (meta["split"] == "train").to_numpy()
    for entity, idx in meta.groupby("entity_id").indices.items():
        rows = X[idx][is_train[idx]]
        with np.errstate(all="ignore"):
            mean = np.nanmean(rows, axis=0) if len(rows) else np.zeros(X.shape[1])
            std = np.nanstd(rows, axis=0) if len(rows) else np.ones(X.shape[1])
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.maximum(np.where(np.isfinite(std) & (std > 0), std, 1.0), std_floor)
        scalers[entity] = {"mean": mean.tolist(), "std": std.tolist()}
    return scalers


def flat_metrics(meta: pd.DataFrame, X: np.ndarray, metrics: list[str], flat_std: float) -> dict[str, list[str]]:
    """Per entity, the metrics whose train-window std of the mean (the first len(metrics) columns of X) is
    below flat_std; with flat_std > 0 a metric without any train value counts as flat too."""
    flat = {}
    is_train = (meta["split"] == "train").to_numpy()
    for entity, idx in meta.groupby("entity_id").indices.items():
        rows = X[idx][is_train[idx]][:, :len(metrics)]
        with np.errstate(all="ignore"):
            std = np.nanstd(rows, axis=0) if len(rows) else np.full(len(metrics), np.nan)
        is_flat = (std < flat_std) | (~np.isfinite(std) & (flat_std > 0))
        flat[entity] = [m for m, f in zip(metrics, is_flat) if f]
    return flat


def recenter_centers(meta: pd.DataFrame, X: np.ndarray, scalers: dict, windows: int = RECENTER) -> np.ndarray:
    """Per-window centres: the train mean for train windows; for test windows the median of the machine's
    previous `windows` windows (current one excluded, at least windows // 2 values, else the train mean).
    Rows of one entity must be in time order."""
    centers = np.empty_like(X)
    is_test = (meta["split"] == "test").to_numpy()
    for entity, idx in meta.groupby("entity_id").indices.items():
        mean = np.asarray(scalers[entity]["mean"])
        rolled = (pd.DataFrame(X[idx]).rolling(windows, min_periods=max(1, windows // 2)).median()
                  .shift(1).to_numpy())
        rolled = np.where(np.isnan(rolled), mean, rolled)
        centers[idx] = np.where(is_test[idx, None], rolled, mean)
    return centers


def standardise(meta: pd.DataFrame, X: np.ndarray, scalers: dict, feature_metrics: list[str] | None = None,
                centers: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Z, mask): Z standardised with NaN replaced by 0, mask = value present.

    centers (same shape as X) replaces the train mean when given. With feature_metrics (the metric of each
    column), the columns of each entity's "masked" metrics are set to 0 and left out of the mask.
    """
    Z = np.empty_like(X)
    for entity, idx in meta.groupby("entity_id").indices.items():
        s = scalers[entity]
        center = np.asarray(s["mean"]) if centers is None else centers[idx]
        Z[idx] = (X[idx] - center) / np.asarray(s["std"])
        if feature_metrics is not None and s.get("masked"):
            cols = np.flatnonzero(np.isin(feature_metrics, s["masked"]))
            Z[np.ix_(idx, cols)] = np.nan
    mask = ~np.isnan(Z)
    return np.nan_to_num(Z, nan=0.0), mask


# ---------------------------------------------------------------- model

class DenseAE(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, int] = HIDDEN):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _masked_mse(out, z, mask):
    return ((out - z) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def train(Z: np.ndarray, mask: np.ndarray, epochs: int = MAX_EPOCHS, seed: int = SEED,
          verbose: bool = True) -> tuple[DenseAE, list[dict]]:
    """Train on Z (train windows only); early stopping on a random VAL_FRACTION hold-out."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z))
    n_val = max(1, int(round(VAL_FRACTION * len(Z))))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    zt = torch.tensor(Z, dtype=torch.float32)
    mt = torch.tensor(mask, dtype=torch.float32)
    z_tr, m_tr, z_val, m_val = zt[tr_idx], mt[tr_idx], zt[val_idx], mt[val_idx]

    model = DenseAE(Z.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    history, best_val, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(z_tr), generator=gen)
        total, count = 0.0, 0
        for start in range(0, len(order), BATCH):
            b = order[start:start + BATCH]
            loss = _masked_mse(model(z_tr[b]), z_tr[b], m_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            count += len(b)
        model.eval()
        with torch.no_grad():
            val_loss = _masked_mse(model(z_val), z_val, m_val).item()
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        if verbose:
            print(f"  epoch {epoch:3d}  train {total / count:.5f}  val {val_loss:.5f}")
        if val_loss < best_val:
            best_val, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def score(model: DenseAE, Z: np.ndarray, mask: np.ndarray, metrics: list[str]) -> tuple[np.ndarray, list]:
    """Returns (ae_score, top_metrics) per window; score is NaN if a window has no metric values.

    Z has one block of len(metrics) columns per aggregate; a metric's error for top_metrics is the sum of
    its errors over the blocks.
    """
    with torch.no_grad():
        out = model(torch.tensor(Z, dtype=torch.float32)).numpy().astype(np.float64)
    sq = (out - Z) ** 2 * mask
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ae_score = np.where(n > 0, sq.sum(axis=1) / n, np.nan)
    blocks = Z.shape[1] // len(metrics)
    per_metric = sq.reshape(len(Z), blocks, len(metrics)).sum(axis=1)
    present = mask.reshape(len(Z), blocks, len(metrics)).any(axis=1)
    ranked = np.argsort(-np.where(present, per_metric, -np.inf), axis=1, kind="stable")[:, :TOP_K]
    names = np.asarray(metrics)
    top = [names[r[present[i, r]]].tolist() for i, r in enumerate(ranked)]
    return ae_score, top


# ---------------------------------------------------------------- output and evaluation

def thresholds(meta: pd.DataFrame, ae_score: np.ndarray) -> dict:
    """Global and per-machine thresholds: THRESHOLD_PERCENTILE of the train ae_score x THRESHOLD_FACTOR."""
    train = (meta["split"] == "train").to_numpy() & np.isfinite(ae_score)
    p_global = float(np.percentile(ae_score[train], THRESHOLD_PERCENTILE))
    per_machine = {}
    for entity, idx in meta.groupby("entity_id").indices.items():
        own = ae_score[idx][train[idx]]
        # a machine without train scores falls back to the global threshold
        p = float(np.percentile(own, THRESHOLD_PERCENTILE)) if len(own) else p_global
        per_machine[entity] = p * THRESHOLD_FACTOR
    return {"percentile": THRESHOLD_PERCENTILE, "factor": THRESHOLD_FACTOR,
            "global": p_global * THRESHOLD_FACTOR, "per_machine": per_machine}


def adaptive_limits(meta: pd.DataFrame, ae_score: np.ndarray, n: int,
                    percentile: float = ADAPTIVE_PERCENTILE) -> np.ndarray:
    """Per window, the percentile of the machine's previous n scores (current one excluded); NaN while
    fewer than n // 2 previous scores exist. Rows of one entity must be in time order."""
    limit = np.full(len(meta), np.nan)
    for entity, idx in meta.groupby("entity_id").indices.items():
        s = pd.Series(ae_score[idx])
        limit[idx] = (s.rolling(n, min_periods=max(1, n // 2)).quantile(percentile / 100)
                      .shift(1).to_numpy())
    return limit


def flags(meta: pd.DataFrame, ae_score: np.ndarray, info: dict, mode: str,
          adaptive_limit: np.ndarray | None = None) -> np.ndarray:
    """ae_flag = ae_score > threshold of the window's machine (mode per-machine) or the global one,
    and, when adaptive_limit is given, ae_score > adaptive_limit wherever that limit is defined."""
    if mode == "global":
        limit = np.full(len(meta), info["global"])
    else:
        limit = meta["entity_id"].map(info["per_machine"]).to_numpy(dtype=np.float64)
    flag = np.nan_to_num(ae_score, nan=-np.inf) > limit
    if adaptive_limit is not None:
        flag &= np.isnan(adaptive_limit) | (np.nan_to_num(ae_score, nan=-np.inf) > adaptive_limit)
    return flag


def scores_table(meta: pd.DataFrame, ae_score: np.ndarray, ae_flag: np.ndarray, top: list) -> pa.Table:
    schema = S.arrow_schema("scores")
    n = len(meta)
    start = pa.array(meta["window_start"].to_numpy(), type=pa.int64())
    cols = {
        "entity_id": pa.array(meta["entity_id"].tolist(), type=pa.string()),
        "window_start": start.cast(schema.field("window_start").type),
        "window_end": pc.add(start, WINDOW_US).cast(schema.field("window_end").type),
        "ae_score": pa.array(ae_score, type=pa.float64(), from_pandas=True),
        "ae_flag": pa.array(ae_flag, type=pa.bool_()),
        "is_anomaly": pa.array(ae_flag, type=pa.bool_()),
        "top_metrics": pa.array(top, type=pa.list_(pa.string())),
        "label": pa.array(meta["label"].to_numpy(), type=pa.float64(), from_pandas=True).cast(pa.int32()),
    }
    arrays = [cols[f.name] if f.name in cols else pa.nulls(n, type=f.type) for f in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _median_minutes(delays: list[int]) -> float:
    return float(np.median(delays)) * WINDOW_MINUTES if delays else float("nan")


def evaluate(meta: pd.DataFrame, ae_flag: np.ndarray,
             ae_score: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-machine metrics on time-ordered test windows, plus overall metrics.

    Overall = TP/FP/FN summed over machines: the machines' sequences are concatenated with one
    (label 0, prediction 0) separator point between them, so no anomaly segment spans two machines
    and the separators add no TP, FP or FN. False alarms per day are per machine-day (no separators),
    the median delay is over all detected faults, and PR-AUC pools the scores of all machines
    (a NaN ae_score counts as 0; without ae_score PR-AUC is NaN).
    """
    score_col = np.full(len(meta), np.nan) if ae_score is None else np.nan_to_num(ae_score, nan=0.0)
    df = meta.assign(pred=ae_flag.astype(int), score=score_col)
    df = df[df["split"] == "test"].sort_values(["entity_id", "window_start"])
    rows, all_t, all_p, all_delays = [], [], [], []
    for entity, g in df.groupby("entity_id", sort=True):
        t, p, s = g["label"].astype(int).to_numpy(), g["pred"].to_numpy(), g["score"].to_numpy()
        pt, ev, comp = point_metrics(t, p), event_metrics(t, p), composite_metrics(t, p)
        delays = detection_delays(t, p)
        rows.append({"entity_id": entity, "windows": len(t), "anom": int(t.sum()), "events": comp["events"],
                     "flagged": int(p.sum()),
                     "pt_p": pt["precision"], "pt_r": pt["recall"], "pt_f1": pt["f1"],
                     "ev_p": ev["precision"], "ev_r": ev["recall"], "ev_f1": ev["f1"],
                     "caught": comp["events_detected"], "alarm_p": comp["alarm_precision"],
                     "comp_f1": comp["composite_f1"],
                     "fa_day": false_alarms_per_day(t, p, WINDOW_MINUTES),
                     "delay_min": _median_minutes(delays),
                     "pr_auc": pr_auc(t, s) if ae_score is not None else float("nan")})
        all_t += [0] + t.tolist()
        all_p += [0] + p.tolist()
        all_delays += delays
    test_t, test_p = df["label"].astype(int).to_numpy(), df["pred"].to_numpy()
    overall = {"point": point_metrics(all_t, all_p), "event": event_metrics(all_t, all_p),
               "composite": composite_metrics(all_t, all_p),
               "fa_day": false_alarms_per_day(test_t, test_p, WINDOW_MINUTES),
               "delay_min": _median_minutes(all_delays),
               "pr_auc": pr_auc(test_t, df["score"].to_numpy()) if ae_score is not None else float("nan")}
    return pd.DataFrame(rows), overall


REPORT_COLUMNS = {"entity_id": "machine", "windows": "windows", "events": "faults", "caught": "caught",
                  "flagged": "flagged", "comp_f1": "composite_f1", "alarm_p": "alarm_prec",
                  "fa_day": "false_alarms/day", "delay_min": "med_delay_min", "pr_auc": "pr_auc",
                  "ev_f1": "event_f1(ref,generous)"}


def _print_report(per_machine: pd.DataFrame, overall: dict) -> None:
    with pd.option_context("display.max_rows", None, "display.width", 250,
                           "display.float_format", "{:.3f}".format):
        print(per_machine[list(REPORT_COLUMNS)].rename(columns=REPORT_COLUMNS).to_string(index=False))
    c = overall["composite"]
    print(f"OVERALL composite F1 {c['composite_f1']:.3f}  faults caught {c['events_detected']}/{c['events']} "
          f"(event recall {c['event_recall']:.3f})  alarm precision {c['alarm_precision']:.3f}")
    print(f"OVERALL false alarms per machine per day {overall['fa_day']:.2f}  "
          f"median detection delay {overall['delay_min']:.1f} min  PR-AUC {overall['pr_auc']:.3f}")
    for level in ("point", "event"):
        m = overall[level]
        note = "  (reference, generous)" if level == "event" else ""
        print(f"OVERALL {level:5s}  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"f1 {m['f1']:.3f}{note}")


# ---------------------------------------------------------------- driver

def run(source: str = "smd", sample: int | None = None, epochs: int = MAX_EPOCHS,
        threshold_mode: str = "global", std_floor: float = STD_FLOOR,
        root: Path = ROOT, verbose: bool = True, flat_std: float = FLAT_STD,
        inputs: str = DEFAULT_INPUTS, recenter: int = RECENTER, adaptive: int = ADAPTIVE) -> dict:
    metrics = METRICS[source]
    aggs = INPUTS[inputs]
    features = feature_names(metrics, aggs)
    feature_metrics = [m for _ in aggs for m in metrics]
    meta, X = load_windows(root / S.PATHS["raw"] / f"source={source}", metrics, sample, aggs)
    is_train = (meta["split"] == "train").to_numpy()
    print(f"{meta['entity_id'].nunique()} machines, {is_train.sum()} train + {(~is_train).sum()} test windows")

    scalers = fit_scalers(meta, X, std_floor)
    for entity, masked in flat_metrics(meta, X, metrics, flat_std).items():
        scalers[entity]["masked"] = masked
    centers = recenter_centers(meta, X, scalers, recenter) if recenter else None
    Z, mask = standardise(meta, X, scalers, feature_metrics, centers)
    model, history = train(Z[is_train], mask[is_train], epochs=epochs, verbose=verbose)

    ae_score, top = score(model, Z, mask, metrics)
    adaptive_info = ({"windows": adaptive, "percentile": ADAPTIVE_PERCENTILE, "min_periods": max(1, adaptive // 2)}
                     if adaptive else None)
    threshold_info = {"mode": threshold_mode, **thresholds(meta, ae_score), "adaptive": adaptive_info}
    limit = adaptive_limits(meta, ae_score, adaptive) if adaptive else None
    ae_flag = flags(meta, ae_score, threshold_info, threshold_mode, limit)

    model_dir = root / S.MODELS["ae"].replace("<source>", source)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_in": len(features), "hidden": list(HIDDEN),
                "metrics": metrics, "inputs": inputs, "features": features,
                "window_minutes": WINDOW_MINUTES}, model_dir / "model.pt")
    recenter_info = ({"windows": recenter, "min_periods": max(1, recenter // 2), "statistic": "median",
                      "applies_to": "test", "fallback": "train mean"} if recenter else None)
    (model_dir / "scaler.json").write_text(
        json.dumps({"metrics": metrics, "inputs": inputs, "features": features, "std_floor": std_floor,
                    "flat_std": flat_std, "recenter": recenter_info, "entities": scalers}))
    (model_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2))
    (model_dir / "history.json").write_text(json.dumps(history, indent=2))

    test = ~is_train
    out_dir = root / SCORES_AE_DIR / f"source={source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_meta = meta[test].reset_index(drop=True)
    table = scores_table(test_meta, ae_score[test], ae_flag[test], [top[i] for i in np.flatnonzero(test)])
    pq.write_table(table, out_dir / "scores.parquet")

    best = min(history, key=lambda h: h["val_loss"])
    print(f"training: {len(history)} epochs run, best epoch {best['epoch']} "
          f"(train {best['train_loss']:.5f}, val {best['val_loss']:.5f}); "
          f"epoch 1 train {history[0]['train_loss']:.5f}, val {history[0]['val_loss']:.5f}")
    n_masked = sum(len(s["masked"]) for s in scalers.values())
    print(f"inputs {inputs} ({len(features)}), flat std {flat_std} ({n_masked} machine-metrics masked), "
          f"recenter {recenter}, adaptive {adaptive}")
    per = np.array(list(threshold_info["per_machine"].values()))
    print(f"threshold mode {threshold_mode}, std floor {std_floor}: p{THRESHOLD_PERCENTILE} of train ae_score "
          f"x {THRESHOLD_FACTOR}; global {threshold_info['global']:.5f}, per machine min "
          f"{per.min():.5f} / median {np.median(per):.5f} / max {per.max():.5f}")
    print(f"wrote {table.num_rows} test windows to {out_dir / 'scores.parquet'} ({int(ae_flag[test].sum())} flagged)")

    per_machine, overall = evaluate(meta, ae_flag, ae_score)
    _print_report(per_machine, overall)
    return {"meta": meta, "ae_score": ae_score, "ae_flag": ae_flag, "thresholds": threshold_info,
            "scalers": scalers, "history": history, "per_machine": per_machine, "overall": overall}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=sorted(METRICS), default="smd")
    parser.add_argument("--sample", type=int, default=None, help="use only the first N machines")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="maximum epochs")
    parser.add_argument("--threshold", choices=THRESHOLD_MODES, default="global",
                        help="one threshold for all machines, or one per machine")
    parser.add_argument("--std-floor", type=float, default=STD_FLOOR,
                        help="minimum per-machine std when standardising, e.g. 0.05 (0 = no floor)")
    parser.add_argument("--flat-std", type=float, default=FLAT_STD,
                        help="mask metrics whose per-machine train-window std is below this (0 = off)")
    parser.add_argument("--inputs", choices=sorted(INPUTS), default=DEFAULT_INPUTS,
                        help="window aggregates fed to the model: mean (38 inputs) or mean_max (76)")
    parser.add_argument("--recenter", type=int, default=RECENTER,
                        help="centre test windows on the median of the previous N windows (0 = train mean)")
    parser.add_argument("--adaptive", type=int, default=ADAPTIVE,
                        help="also require ae_score > p99 of the machine's previous N scores, "
                             "default 288 = one day (0 = off)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/ and models/")
    args = parser.parse_args()
    run(args.source, args.sample, args.epochs, args.threshold, args.std_floor, args.root,
        flat_std=args.flat_std, inputs=args.inputs, recenter=args.recenter, adaptive=args.adaptive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
