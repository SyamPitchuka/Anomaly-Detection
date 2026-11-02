"""Tier 2 (KMeans) and Tier 2b (PCA) scoring for the streaming score job, with the offline baseline models.

src/streaming/score_job.py calls, inside its Python workers (no SparkSession there):
    models["baselines"] = load_models()                       # once per worker process
    score_baselines(pdf, models, history)                     # per micro-batch of one machine

load_models() reads models/kmeans_smd/ and models/pca_smd/ exactly as src/detection/baselines.py saved them
(PipelineModel folders + scaler.json + thresholds.json) into plain numpy: the KMeans cluster centres and the
PCA matrix come from the stage's data Parquet (Spark ML vector / matrix structs, read with pyarrow).

score_baselines() gives the batch answers of baselines.py for the new windows of one machine:
- standardise each window mean with the machine's train mean / std (scaler.json); with recenter N on,
  test windows are centred on the median of the machine's previous N windows (train or test, at least
  N // 2 values, else the train mean), via autoencoder.recenter_centers as in baselines.recenter_stats.
  The previous windows come from `history` (that machine's earlier raw window means, oldest first) plus
  the earlier windows of pdf, so the result does not depend on how windows are split over micro-batches
  as long as history holds the last N windows (score_job keeps the last ae recenter windows; both are 24).
- tier2_score = distance to the nearest centre, tier2_flag = tier2_score > thresholds.json global,
  pca_score = mean squared reconstruction error (baselines.kmeans_distance / pca_error).
A machine unknown to the models (not in scaler.json) gets NaN / None.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from contracts import schemas as S
from src.detection import autoencoder as AE
from src.detection.baselines import kmeans_distance, pca_error, pca_projector

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "smd"
KMEANS_DIR = ROOT / S.MODELS["kmeans"].replace("<source>", SOURCE)
PCA_DIR = ROOT / S.MODELS["pca"].replace("<source>", SOURCE)
OUTPUT_COLUMNS = ["tier2_score", "tier2_flag", "pca_score"]


# ---------------------------------------------------------------- reading saved PipelineModels

def _stage_data(model_dir: Path, cls: str) -> list[dict]:
    """Rows of the data Parquet of the PipelineModel stage whose metadata class is `cls`."""
    for stage in sorted((Path(model_dir) / "stages").iterdir()):
        meta = json.loads((stage / "metadata" / "part-00000").read_text().splitlines()[0])
        if meta["class"] == cls:
            files = sorted((stage / "data").glob("*.parquet"))
            return [r for f in files for r in pq.read_table(f).to_pylist()]
    raise FileNotFoundError(f"no {cls} stage under {model_dir}")


def _vector(v: dict) -> np.ndarray:
    """Spark ML vector struct (type 0 sparse, 1 dense) -> numpy."""
    if v["type"] == 1:
        return np.asarray(v["values"], dtype=np.float64)
    out = np.zeros(v["size"])
    out[np.asarray(v["indices"], dtype=int)] = v["values"]
    return out


def _matrix(m: dict) -> np.ndarray:
    """Spark ML dense matrix struct -> numpy (values column-major unless isTransposed)."""
    if m["type"] != 1:
        raise ValueError("sparse PCA matrix not supported")
    values = np.asarray(m["values"], dtype=np.float64)
    if m["isTransposed"]:
        return values.reshape(m["numRows"], m["numCols"])
    return values.reshape(m["numCols"], m["numRows"]).T


def load_models(kmeans_dir: Path = KMEANS_DIR, pca_dir: Path = PCA_DIR,
                history_windows: int | None = None) -> dict | None:
    """The two baseline models as numpy, or None if either folder is missing (scores then stay NaN).

    history_windows: how many previous windows the caller keeps per machine for `history` (score_job:
    the AE recenter count kept in hist_x). If the baselines' recenter needs more, raise ValueError at
    startup, since the stream would otherwise silently drift from the batch scores. None = not checked."""
    kmeans_dir, pca_dir = Path(kmeans_dir), Path(pca_dir)
    if not (kmeans_dir / "stages").exists() or not (pca_dir / "stages").exists():
        return None
    rows = sorted(_stage_data(kmeans_dir, "org.apache.spark.ml.clustering.KMeansModel"),
                  key=lambda r: r["clusterIdx"])
    centers = np.vstack([_vector(r["clusterCenter"]) for r in rows])
    pc = _matrix(_stage_data(pca_dir, "org.apache.spark.ml.feature.PCAModel")[0]["pc"])
    scaler = json.loads((kmeans_dir / "scaler.json").read_text())
    if json.loads((pca_dir / "scaler.json").read_text()) != scaler:
        raise ValueError(f"{kmeans_dir} and {pca_dir} have different scaler.json files")
    recenter = recenter["windows"] if (recenter := scaler.get("recenter")) else 0
    if history_windows is not None and recenter > history_windows:
        raise ValueError(
            f"baseline models in {kmeans_dir.as_posix()} recenter on the previous {recenter} windows, but the "
            f"stream keeps only {history_windows} previous windows per machine (autoencoder recenter in "
            f"scaler.json); tier2/pca scores would drift from the batch. Re-run src/detection/baselines.py "
            f"with --recenter {history_windows} or less, or the autoencoder with --recenter {recenter} or more.")
    return {"metrics": scaler["metrics"], "scalers": scaler["entities"], "recenter": recenter,
            "centers": centers, "pc": pc, "projector": pca_projector(pc),
            "threshold": {"tier2_score": json.loads((kmeans_dir / "thresholds.json").read_text())["global"],
                          "pca_score": json.loads((pca_dir / "thresholds.json").read_text())["global"]}}


# ---------------------------------------------------------------- scoring

def _bundle(models: dict | None) -> dict | None:
    """Accepts score_job's model dict (models["baselines"]) or load_models()'s result itself."""
    if models is not None and "centers" not in models:
        return models.get("baselines")
    return models


def score_baselines(pdf: pd.DataFrame, models: dict | None, history: np.ndarray | None = None) -> pd.DataFrame:
    """pdf: one machine's new windows in time order (entity_id, split, <metric>_mean). history: that
    machine's previous raw window means (n x metrics, oldest first, NaN = no value), or None.
    Returns tier2_score, tier2_flag, pca_score aligned with pdf."""
    n = len(pdf)
    out = pd.DataFrame({"tier2_score": np.full(n, np.nan), "tier2_flag": pd.Series([None] * n, dtype=object),
                        "pca_score": np.full(n, np.nan)})
    out.index = pdf.index
    b = _bundle(models)
    if b is None or n == 0:
        return out
    entities = pdf["entity_id"].unique()
    if len(entities) != 1:
        raise ValueError(f"score_baselines expects one machine per call, got {len(entities)}")
    entity = entities[0]
    if entity not in b["scalers"]:
        return out

    X = pdf[[f"{m}_mean" for m in b["metrics"]]].to_numpy(dtype=np.float64)
    split = pdf["split"].tolist()
    meta = pd.DataFrame({"entity_id": [entity] * n, "split": split})
    centers = None
    if b["recenter"]:
        H = np.zeros((0, X.shape[1])) if history is None else np.asarray(history, dtype=np.float64)
        H = H[len(H) - min(len(H), b["recenter"]):]    # only the last N windows can reach a median
        meta_all = pd.DataFrame({"entity_id": [entity] * (len(H) + n), "split": ["train"] * len(H) + split})
        centers = AE.recenter_centers(meta_all, np.vstack([H, X]), b["scalers"], b["recenter"])[len(H):]
    Z, _ = AE.standardise(meta, X, b["scalers"], None, centers)

    tier2 = kmeans_distance(Z, b["centers"])
    out["tier2_score"] = tier2
    out["tier2_flag"] = pd.Series([bool(s > b["threshold"]["tier2_score"]) for s in tier2],
                                  index=pdf.index, dtype=object)
    out["pca_score"] = pca_error(Z, b["projector"])
    return out
