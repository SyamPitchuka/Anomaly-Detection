"""Streamlit dashboard (WP-4): detector scores and incidents per machine.

Start from the repo root:  streamlit run src/dashboard/app.py
Reads data only through src/dashboard/data.py. The main view refreshes every 10 seconds.
"""
import sys
from pathlib import Path

# streamlit run puts src/dashboard on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.dashboard.data import load_incidents, load_scores  # noqa: E402

REFRESH_SECONDS = 10
SCORE_COLUMNS = ["tier1_score", "tier2_score", "ae_score"]

st.set_page_config(page_title="SPARK anomalies", layout="wide")
st.title("SPARK anomaly detection")

# Sidebar widgets cannot live inside a fragment, so the machine list refreshes on any interaction.
with st.sidebar:
    source = st.selectbox("Source", S.SOURCES)
    machines = sorted(load_scores(source)["entity_id"].unique())
    machine = st.selectbox("Machine", machines, index=0 if machines else None,
                           placeholder="no scored machines yet")
    st.caption(f"Refreshes every {REFRESH_SECONDS} s.")


def score_chart(scores: pd.DataFrame) -> alt.Chart:
    long = scores.melt(id_vars=["window_start"], value_vars=SCORE_COLUMNS,
                       var_name="detector", value_name="score")
    x = alt.X("window_start:T", title="window_start (UTC)", scale=alt.Scale(type="utc"))
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=x, y=alt.Y("score:Q", title="score"),
        color=alt.Color("detector:N", sort=SCORE_COLUMNS),
        tooltip=[alt.Tooltip("window_start:T", format="%Y-%m-%d %H:%M"), "detector:N",
                 alt.Tooltip("score:Q", format=".3f")],
    )
    anomalous = scores.loc[scores["is_anomaly"], ["window_start", "window_end"]]
    bands = alt.Chart(anomalous).mark_rect(color="crimson", opacity=0.15).encode(
        x=x, x2="window_end:T",
        tooltip=[alt.Tooltip("window_start:T", title="anomalous from", format="%H:%M"),
                 alt.Tooltip("window_end:T", title="to", format="%H:%M")],
    )
    return (bands + lines).properties(height=380)


def incidents_table(incidents: pd.DataFrame) -> pd.DataFrame:
    table = incidents.sort_values("start_time", ascending=False).reset_index(drop=True)
    table["top_metrics"] = table["top_metrics"].map(lambda m: ", ".join(m) if m is not None else "")
    return table


@st.experimental_fragment(run_every=REFRESH_SECONDS)
def main_view(source: str, machine: str | None):
    scores, incidents = load_scores(source), load_incidents(source)

    c1, c2, c3 = st.columns(3)
    c1.metric("Windows scored", f"{len(scores):,}")
    c2.metric("Anomalous windows", f"{int(scores['is_anomaly'].sum()):,}")
    c3.metric("Incidents", f"{len(incidents):,}")

    st.subheader(f"Scores for {machine}" if machine else "Scores")
    mine = scores[scores["entity_id"] == machine].sort_values("window_start")
    if mine.empty:
        st.info(f"No scores for {machine or 'any machine'} in {S.PATHS['scores']}/source={source} yet.")
    else:
        st.altair_chart(score_chart(mine), use_container_width=True)
        st.caption("Shaded bands are windows with is_anomaly = true.")

    st.subheader("Incidents, newest first")
    if incidents.empty:
        st.info(f"No incidents in {S.PATHS['incidents']}/source={source} yet.")
    else:
        st.dataframe(incidents_table(incidents), use_container_width=True, hide_index=True)


main_view(source, machine)
