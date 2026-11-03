"""E1 accuracy report: every detector in the scores layer on all 28 SMD machines (test windows).

Run from the repo root:
    python src/detection/e1_report.py

Reads data/scores/source=smd (the scores layer written by the streaming job; test windows only) and scores
each detector with src/detection/evaluate.py exactly as src/detection/autoencoder.py and baselines.py report:
per machine on time-ordered windows, machines joined by one (label 0, prediction 0) separator, totals over
machines (the `evaluate` function of autoencoder.py is reused, so the numbers cannot drift apart).

Detectors (rows):
    Tier 1 robust z-score       tier1_flag, score tier1_score
    Tier 2 KMeans (recenter 24) tier2_flag, score tier2_score
    Tier 2b PCA (recenter 24)   pca_score > the threshold in models/pca_smd/thresholds.json
                                (the scores layer has no pca flag)
    Tier 3 AE v2                ae_flag, score ae_score
    Final rule                  is_anomaly (contract rev C: is_anomaly = ae_flag), score ae_score

Sanity checks run first and stop the script with a clear message if not met: 141,708 test windows and
28 machines; tier1, tier2, pca and ae scores not all null; is_anomaly == ae_flag on every window.

Outputs: docs/results/e1.md, docs/results/e1.json, docs/results/e1.png.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection.autoencoder import evaluate as evaluate_flags  # noqa: E402

SOURCE = "smd"
EXPECTED_WINDOWS = 141_708
EXPECTED_MACHINES = 28
PCA_THRESHOLDS = Path("models/pca_smd/thresholds.json")
OUT_DIR = Path("docs/results")

# name -> (score column, flag column or None for "score > PCA threshold")
DETECTORS = {
    "Tier 1 robust z-score": ("tier1_score", "tier1_flag"),
    "Tier 2 KMeans (recenter 24)": ("tier2_score", "tier2_flag"),
    "Tier 2b PCA (recenter 24)": ("pca_score", None),
    "Tier 3 AE v2": ("ae_score", "ae_flag"),
    "Final rule: is_anomaly": ("ae_score", "is_anomaly"),
}
SHORT = {"Tier 1 robust z-score": "tier1", "Tier 2 KMeans (recenter 24)": "kmeans",
         "Tier 2b PCA (recenter 24)": "pca", "Tier 3 AE v2": "ae_v2", "Final rule: is_anomaly": "is_anomaly"}

# Earlier numbers quoted in the report, with their source (not recomputed here).
REFERENCE_ROWS = [
    {"detector": "Tier 2 KMeans, recenter 0", "metric": "composite F1", "value": 0.266,
     "faults_caught": "225/326", "false_alarms_per_machine_day": 31.47, "source": "PR #9"},
    {"detector": "Tier 2b PCA, recenter 0", "metric": "composite F1", "value": 0.246,
     "faults_caught": "195/326", "false_alarms_per_machine_day": 30.15, "source": "PR #9"},
    {"detector": "Tier 3 AE v1", "metric": "point-adjusted event F1 (generous)", "value": 0.360,
     "faults_caught": "222/326", "false_alarms_per_machine_day": 36.98,
     "source": "PR #6 (composite F1 0.236 in the v1 rerun, docs/ae_ablation.md)"},
    {"detector": "Tier 1 v2, best setting (k = 3, t = 6)", "metric": "composite F1", "value": 0.338,
     "faults_caught": "150/326", "false_alarms_per_machine_day": 9.4,
     "source": "PR #15, docs/results/tier1_stream.md (not selected: above 2 false alarms per machine per day)"},
]


class SanityError(RuntimeError):
    """The scores layer is not what E1 expects."""


# ---------------------------------------------------------------- input

def load_scores(scores_dir: Path) -> pd.DataFrame:
    """The scores layer of one source as a pandas frame, sorted by machine and window."""
    if not scores_dir.exists():
        raise SanityError(f"{scores_dir} does not exist: run the scoring job first")
    table = ds.dataset(str(scores_dir), format="parquet").to_table(columns=list(S.LAYERS["scores"]))
    return table.to_pandas().sort_values(["entity_id", "window_start"]).reset_index(drop=True)


def pca_threshold(path: Path) -> float:
    info = json.loads(path.read_text())
    if info.get("mode", "global") != "global":
        raise SanityError(f"{path}: E1 expects a global PCA threshold, found mode {info.get('mode')!r}")
    return float(info["global"])


def sanity_checks(df: pd.DataFrame, expected_windows: int = EXPECTED_WINDOWS,
                  expected_machines: int = EXPECTED_MACHINES, verbose: bool = True) -> None:
    """Raises SanityError with a clear message on the first check that fails."""
    def ok(msg):
        if verbose:
            print(f"  OK   {msg}")

    def fail(msg):
        raise SanityError(f"sanity check failed: {msg}")

    if verbose:
        print("Sanity checks")
    n, m = len(df), df["entity_id"].nunique()
    if n != expected_windows:
        fail(f"{n:,} test windows, expected {expected_windows:,}")
    ok(f"{n:,} test windows")
    if m != expected_machines:
        fail(f"{m} machines, expected {expected_machines}")
    ok(f"{m} machines")
    dup = int(df.duplicated(["entity_id", "window_start"]).sum())
    if dup:
        fail(f"{dup} duplicate (entity_id, window_start) rows")
    ok("no duplicate (entity_id, window_start)")
    if df["label"].isna().any():
        fail(f"{int(df['label'].isna().sum())} test windows without a label")
    ok("every window has a label")
    for col in ("tier1_score", "tier2_score", "pca_score", "ae_score"):
        if df[col].isna().all():
            fail(f"{col} is null on every window")
        ok(f"{col} has values ({int(df[col].isna().sum())} null)")
    ae, final = _as_flag(df["ae_flag"]), _as_flag(df["is_anomaly"])
    diff = int((ae != final).sum())
    if diff:
        fail(f"is_anomaly differs from ae_flag on {diff} windows (contract rev C: is_anomaly = ae_flag)")
    ok("is_anomaly == ae_flag on every window (rev C)")


def _as_flag(s: pd.Series) -> np.ndarray:
    return s.map(lambda v: v is True or v is np.True_).to_numpy(dtype=bool)


# ---------------------------------------------------------------- evaluation

def detector_flags(df: pd.DataFrame, pca_thr: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name, (score_col, flag_col) in DETECTORS.items():
        score = df[score_col].to_numpy(dtype=np.float64)
        if flag_col is None:
            flag = np.nan_to_num(score, nan=-np.inf) > pca_thr
        else:
            flag = _as_flag(df[flag_col])
        out[name] = (flag, score)
    return out


def evaluate_all(df: pd.DataFrame, pca_thr: float) -> tuple[list[dict], pd.DataFrame]:
    """Overall rows (one per detector) and a per-machine composite F1 table."""
    meta = pd.DataFrame({"entity_id": df["entity_id"], "window_start": df["window_start"],
                         "split": "test", "label": df["label"].astype("float64")})
    rows, per_machine = [], {}
    for name, (flag, score) in detector_flags(df, pca_thr).items():
        pm, o = evaluate_flags(meta, flag, score)
        c = o["composite"]
        rows.append({
            "detector": name, "key": SHORT[name],
            "composite_f1": c["composite_f1"], "faults_caught": c["events_detected"], "faults": c["events"],
            "event_recall": c["event_recall"], "alarm_precision": c["alarm_precision"],
            "false_alarms_per_machine_day": o["fa_day"], "median_delay_min": o["delay_min"],
            "pr_auc": o["pr_auc"],
            "point_precision": o["point"]["precision"], "point_recall": o["point"]["recall"],
            "point_f1": o["point"]["f1"],
            "event_precision": o["event"]["precision"], "event_recall_adjusted": o["event"]["recall"],
            "event_f1": o["event"]["f1"],
            "windows_flagged": int(flag.sum()), "windows": int(len(flag)),
        })
        per_machine[SHORT[name]] = pm.set_index("entity_id")["comp_f1"]
    return rows, pd.DataFrame(per_machine).rename_axis("machine").reset_index()


# ---------------------------------------------------------------- output

def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


MAIN_HEADER = ("| detector | composite F1 | faults caught | alarm precision | false alarms / machine / day "
               "| median delay (min) | PR-AUC | point P | point R | point F1 | event P (adj.) | event R (adj.) "
               "| event F1 (adj.) | windows flagged |\n"
               "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


def main_table(rows: list[dict]) -> str:
    lines = [MAIN_HEADER]
    for r in rows:
        lines.append(
            f"| {r['detector']} | {_f(r['composite_f1'])} | {r['faults_caught']}/{r['faults']} "
            f"| {_f(r['alarm_precision'])} | {_f(r['false_alarms_per_machine_day'], '{:.2f}')} "
            f"| {_f(r['median_delay_min'], '{:.0f}')} | {_f(r['pr_auc'])} "
            f"| {_f(r['point_precision'])} | {_f(r['point_recall'])} | {_f(r['point_f1'])} "
            f"| {_f(r['event_precision'])} | {_f(r['event_recall_adjusted'])} | {_f(r['event_f1'])} "
            f"| {r['windows_flagged']:,} |")
    return "\n".join(lines)


def per_machine_table(pm: pd.DataFrame) -> str:
    cols = list(SHORT.values())
    names = {v: k for k, v in SHORT.items()}
    head = "| machine | " + " | ".join(names[c] for c in cols) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in cols) + " |"
    body = ["| " + r["machine"] + " | " + " | ".join(_f(r[c]) for c in cols) + " |" for _, r in pm.iterrows()]
    return "\n".join([head, sep, *body])


def reference_table() -> str:
    lines = ["| detector | metric | value | faults caught | false alarms / machine / day | source |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for r in REFERENCE_ROWS:
        lines.append(f"| {r['detector']} | {r['metric']} | {r['value']:.3f} | {r['faults_caught']} "
                     f"| {r['false_alarms_per_machine_day']:.2f} | {r['source']} |")
    return "\n".join(lines)


def findings(rows: list[dict], pm: pd.DataFrame) -> str:
    by = {r["key"]: r for r in rows}
    best = max(rows, key=lambda r: r["composite_f1"])
    alarm_keys = ["tier1", "kmeans", "pca", "ae_v2"]
    ranked = sorted(alarm_keys, key=lambda k: -by[k]["composite_f1"])
    wins = pm[alarm_keys].idxmax(axis=1).value_counts()
    t1, ae = by["tier1"], by["ae_v2"]
    km, pca = by["kmeans"], by["pca"]
    names = {v: k for k, v in SHORT.items()}
    higher_adj = all(r["event_f1"] >= r["composite_f1"] for r in rows)
    s = [
        f"On the {ae['windows']:,} test windows, {best['detector']} has the highest composite F1 "
        f"({best['composite_f1']:.3f}), followed by "
        + ", ".join(f"{names[key]} ({by[key]['composite_f1']:.3f})" for key in ranked if key != best["key"])
        + ".",
        f"The AE v2 alarm catches {ae['faults_caught']}/{ae['faults']} faults at "
        f"{ae['false_alarms_per_machine_day']:.2f} false alarms per machine per day, while KMeans and PCA "
        f"(recenter 24) catch {km['faults_caught']} and {pca['faults_caught']} faults at "
        f"{km['false_alarms_per_machine_day']:.2f} and {pca['false_alarms_per_machine_day']:.2f}, so the higher "
        f"composite F1 comes with more false alarms and E1 alone does not settle which alarm to use; AE v2 also "
        f"runs with `--adaptive 288` (score must exceed the 99th percentile of the machine's previous 288 scores), "
        f"which KMeans and PCA do not have.",
        f"Tier 1 v1 catches {t1['faults_caught']}/{t1['faults']} faults but flags {t1['windows_flagged']:,} of "
        f"{t1['windows']:,} windows ({t1['false_alarms_per_machine_day']:.1f} false alarms per machine per day), "
        f"which is why it is kept as a baseline and explainer, not part of the alarm.",
        f"Per machine, the best single detector by composite F1 is "
        + ", ".join(f"{names[key]} on {int(wins.get(key, 0))}"
                    for key in alarm_keys) + " of the 28 machines (ties go to the first listed), "
        "so no detector is best everywhere.",
        ("Point-adjusted event F1 is at least as high as composite F1 for every detector because one hit credits "
         "the whole fault; it is shown only as a reference. " if higher_adj else
         "Point-adjusted event F1 is shown only as a reference (one hit credits the whole fault). ")
        + "All numbers come from one test split of one dataset (SMD), with thresholds fixed from train data "
        "before the test windows were scored.",
    ]
    return " ".join(s)


def write_markdown(path: Path, rows: list[dict], pm: pd.DataFrame, pca_thr: float, n_windows: int) -> None:
    ae = next(r for r in rows if r["key"] == "ae_v2")
    text = f"""# E1: detection accuracy on SMD (all 28 machines)

Generated by `python src/detection/e1_report.py` from `data/scores/source=smd`.

## Method

**Data.** Server Machine Dataset (SMD), 28 machines, 38 metrics (`m00`..`m37`), one row per minute.
Each machine's train rows come first, then its test rows; only train rows are used to fit models and thresholds,
only test rows are scored here. **Windows.** 5-minute tumbling windows per machine (epoch-aligned); a window is
`test` if any of its rows is a test row, and its label is 1 if any row is labelled anomalous. This gives
{n_windows:,} test windows and {ae['faults']} labelled faults (runs of consecutive anomalous windows).
**Thresholds.** KMeans, PCA and AE flag a window when its score exceeds the 99.5th percentile of their train
scores x 1.1, taken over all machines (PCA: {pca_thr:.5f}, read from `models/pca_smd/thresholds.json` because the scores layer
has no PCA flag). AE v2 also requires the score to exceed the 99th percentile of the machine's previous 288 scores.
**Recentring.** KMeans, PCA and AE v2 centre each test window on the median of the machine's previous 24 windows
(current window excluded) instead of the train mean; std stays from train and no labels are used. Tier 1 is v1
(robust z-score against the train median, flag if z > 3). **Final rule.** `is_anomaly = ae_flag` (contract rev C).
**Metrics** (`src/detection/evaluate.py`, scored per machine on time-ordered windows; machines joined by one
label-0/prediction-0 separator so no fault spans two machines; totals over machines):
composite F1 = harmonic mean of event recall (faults with at least one flagged window) and alarm precision
(point-wise precision of flagged windows, so every false alarm counts); false alarms per machine per day = flagged
normal windows / (windows x 5 min / 1 day); median detection delay = minutes from a detected fault's first window
to its first flagged window, over all detected faults; PR-AUC = average precision of the raw score pooled over
machines; point P/R/F1 = window-wise; event P/R/F1 (adj.) = point-adjusted (a hit anywhere in a fault credits the
whole fault), a generous reference only.

## Main table (test windows)

{main_table(rows)}

![composite F1 and false alarms per detector](e1.png)

## Findings

{findings(rows, pm)}

## Notes on small differences

- AE v2 here is 0.523 composite F1 / 179 faults caught / 1,157 windows flagged, against 0.525 / 180 / 1,158 in
  `docs/results/tier1_stream.md`, because the AE was retrained on this laptop (seed 42; PyTorch CPU results differ
  slightly across machines). Every row in this table is scored from the same scores layer, so all of them use this
  same model.
- Tier 1's 111,702 flagged windows are counted over all 141,708 batch test windows, while `tier1_stream.md` counts
  111,674 for `ae_flag OR tier1_flag` over the 141,624 stream-matched windows (the 84 newest windows are still open
  in the stream's state).

## Composite F1 per machine

{per_machine_table(pm)}

## Reference rows (earlier runs, quoted, not recomputed)

{reference_table()}

## Reproduce

```
python src/detection/e1_report.py      # reads data/scores/source=smd, writes docs/results/e1.md/.json/.png
```
"""
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, rows: list[dict], pm: pd.DataFrame, pca_thr: float, n_windows: int) -> None:
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if math.isnan(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v

    doc = {
        "experiment": "E1",
        "source": SOURCE,
        "input": f"{S.PATHS['scores']}/source={SOURCE}",
        "test_windows": n_windows,
        "machines": int(len(pm)),
        "pca_threshold": pca_thr,
        "is_anomaly_rule": "ae_flag (contract rev C)",
        "detectors": [{k: clean(v) for k, v in r.items()} for r in rows],
        "per_machine_composite_f1": [{k: clean(v) for k, v in r.items()} for r in pm.to_dict("records")],
        "reference_rows": REFERENCE_ROWS,
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def write_plot(path: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, bar = "#1f2328", "#59636e", "#e5e7eb", "#2a6fdb"
    names = [r["detector"] for r in rows][::-1]
    comp = [r["composite_f1"] for r in rows][::-1]
    fa = [r["false_alarms_per_machine_day"] for r in rows][::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, vals, title, fmt in ((a1, comp, "Composite F1 (higher is better)", "{:.3f}"),
                                 (a2, fa, "False alarms per machine per day (log scale, lower is better)", "{:.2f}")):
        ax.barh(names, vals, height=0.55, color=bar)
        for y, v in enumerate(vals):
            ax.text(v * (1.08 if ax is a2 else 1) + (0 if ax is a2 else 0.01), y, fmt.format(v),
                    va="center", fontsize=9, color=ink)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9, length=0)
    a1.set_xlim(0, max(comp) * 1.25)
    a2.set_xscale("log")
    a2.set_xlim(min(fa) * 0.5, max(fa) * 4)
    fig.suptitle(f"E1 on SMD: {rows[0]['windows']:,} test windows, 28 machines, {rows[0]['faults']} faults",
                 fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def print_table(rows: list[dict]) -> None:
    view = pd.DataFrame([{
        "detector": r["detector"], "comp_F1": r["composite_f1"], "caught": f"{r['faults_caught']}/{r['faults']}",
        "alarm_P": r["alarm_precision"], "FA/mach/day": r["false_alarms_per_machine_day"],
        "med_delay": r["median_delay_min"], "PR-AUC": r["pr_auc"], "pt_P": r["point_precision"],
        "pt_R": r["point_recall"], "pt_F1": r["point_f1"], "ev_P(adj)": r["event_precision"],
        "ev_R(adj)": r["event_recall_adjusted"], "ev_F1(adj)": r["event_f1"], "flagged": r["windows_flagged"],
    } for r in rows])
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(view.to_string(index=False))


# ---------------------------------------------------------------- driver

def run(root: Path = ROOT, out_dir: Path | None = None, expected_windows: int = EXPECTED_WINDOWS,
        expected_machines: int = EXPECTED_MACHINES, verbose: bool = True) -> dict:
    scores_dir = root / S.PATHS["scores"] / f"source={SOURCE}"
    df = load_scores(scores_dir)
    sanity_checks(df, expected_windows, expected_machines, verbose)
    thr = pca_threshold(root / PCA_THRESHOLDS)
    rows, pm = evaluate_all(df, thr)
    out = root / OUT_DIR if out_dir is None else out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_markdown(out / "e1.md", rows, pm, thr, len(df))
    write_json(out / "e1.json", rows, pm, thr, len(df))
    write_plot(out / "e1.png", rows)
    if verbose:
        print("\nE1 main table (test windows)")
        print_table(rows)
        print(f"\nwrote {out / 'e1.md'}, e1.json, e1.png")
    return {"rows": rows, "per_machine": pm, "pca_threshold": thr}


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    try:
        run()
    except SanityError as e:
        print(f"E1 FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
