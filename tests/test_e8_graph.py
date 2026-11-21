"""experiments/e8_graph.py: co-failure edges, null model, comparisons, report; Pregel PageRank on a tiny graph."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from experiments import e8_graph as E

T0 = datetime(2026, 1, 17, 10, 0, tzinfo=timezone.utc)
M0 = int(T0.timestamp() // 60)


def inc_rows():
    """(machine, start minute, end minute) relative to T0. 1-1 and 1-2 overlap twice; 2-1 starts 10 min after an
    incident of 1-1 ends (gap10 only); 3-1 touches 1-2 (end = start: gap10 only, not gap0) and starts 5 min after
    1-1's first incident ends; 3-2 is alone."""
    return [("machine-1-1", 0, 10), ("machine-1-1", 100, 105), ("machine-1-2", 5, 15), ("machine-1-2", 95, 110),
            ("machine-2-1", 115, 120), ("machine-3-1", 15, 20), ("machine-3-2", 500, 505)]


def inc_df():
    return pd.DataFrame([{"entity_id": m, "start": M0 + s, "end": M0 + e} for m, s, e in inc_rows()])


def write_incidents(path):
    schema = S.arrow_schema("incidents")
    rows = [{"incident_id": f"{m}_x", "entity_id": m, "start_time": T0 + timedelta(minutes=s),
             "end_time": T0 + timedelta(minutes=e), "n_windows": 1, "max_score": 1.0, "top_metrics": ["m00"],
             "created_at": T0} for m, s, e in inc_rows()]
    path.mkdir(parents=True)
    pq.write_table(pa.table({n: [r[n] for r in rows] for n in schema.names}, schema=schema), path / "p.parquet")


def as_tuples(e):
    return [(r.src, r.dst, r.weight) for r in e.itertuples()]


def test_load_incidents_reads_microsecond_timestamps_as_minutes(tmp_path):
    # Regression: the contract writes timestamp[us]; astype(int64) would give microseconds, not nanoseconds.
    write_incidents(tmp_path / "inc")
    got = E.load_incidents(tmp_path / "inc")
    pd.testing.assert_frame_equal(got, inc_df().sort_values(["entity_id", "start"]).reset_index(drop=True))


def test_edges_with_and_without_gap():
    inc = inc_df()
    assert as_tuples(E.edge_weights(inc, 10)) == [("machine-1-1", "machine-1-2", 2), ("machine-1-1", "machine-2-1", 1),
                                                  ("machine-1-1", "machine-3-1", 1), ("machine-1-2", "machine-2-1", 1),
                                                  ("machine-1-2", "machine-3-1", 1)]
    assert as_tuples(E.edge_weights(inc, 9)) == [("machine-1-1", "machine-1-2", 2), ("machine-1-1", "machine-3-1", 1),
                                                 ("machine-1-2", "machine-2-1", 1), ("machine-1-2", "machine-3-1", 1)]
    assert as_tuples(E.edge_weights(inc, 0)) == [("machine-1-1", "machine-1-2", 2)]
    assert E.edge_weights(inc.iloc[[0]], 10).empty


def test_smd_group_and_vertices():
    assert [E.smd_group(m) for m in ("machine-1-1", "machine-3-11", "m_12", "a-b-c")] == ["1", "3", "-", "-"]
    v = E.vertices(inc_df()["entity_id"])
    assert list(v["id"]) == sorted(set(inc_df()["entity_id"])) and list(v["group"]) == ["1", "1", "2", "3", "3"]


def test_output_names_per_source(tmp_path):
    assert E.out_file(tmp_path, "pregel", "smd", "json").name == "pregel.json"
    assert E.out_file(tmp_path, "neo4j", "alibaba", "json").name == "neo4j_alibaba.json"
    assert E.edges_file(tmp_path, "alibaba", "sig").name == "edges_alibaba_sig.csv"
    assert E.edges_file(tmp_path, "smd", "gap0").name == "edges_gap0.csv"
    for src in E.SOURCES:
        cmd = E.cluster_command(src)
        assert f"--source {src}" in cmd and "--total-executor-cores 4" in cmd
        assert f"pregel{E.SOURCES[src]['suffix']}_cluster.json" in cmd
    assert "e8_alibaba_input.py" in E.cluster_command("alibaba")


def test_spans_fall_back_to_incidents_and_widen_from_scores(tmp_path):
    inc = inc_df()
    spans = E.load_spans(None, inc)
    assert spans["machine-1-1"] == (M0, M0 + 105)
    sc = pd.DataFrame({"entity_id": ["machine-1-1", "machine-1-1"],
                       "window_start": [T0 - timedelta(minutes=60), T0 + timedelta(minutes=900)]})
    sc["window_end"] = sc["window_start"] + timedelta(minutes=5)
    sc.to_parquet(tmp_path / "scores.parquet")
    spans = E.load_spans(tmp_path / "scores.parquet", inc)
    assert spans["machine-1-1"] == (M0 - 60, M0 + 905) and spans["machine-3-2"] == (M0 + 500, M0 + 505)
    # a machine with scores but no incident gets a span, hence a vertex
    sc2 = pd.concat([sc, sc.assign(entity_id="machine-9-9")])
    sc2.to_parquet(tmp_path / "scores2.parquet")
    spans = E.load_spans(tmp_path / "scores2.parquet", inc)
    assert spans["machine-9-9"] == (M0 - 60, M0 + 905)
    vert, graphs, null = E.build_graphs(inc, spans, n_shuffles=5, seed=1)
    assert "machine-9-9" in set(vert["id"]) and null["n_pairs"] == 15
    assert "machine-9-9" not in set(graphs["gap10"]["src"]) | set(graphs["gap10"]["dst"])


def test_shuffle_keeps_count_durations_and_span():
    inc = inc_df()
    spans = {m: (M0 - 50, M0 + 1000) for m in inc["entity_id"]}
    sh = E.shuffle_incidents(inc, spans, np.random.default_rng(1))
    for m, g in inc.groupby("entity_id"):
        h = sh[sh["entity_id"] == m].sort_values("start")
        assert len(h) == len(g)
        assert sorted(h["end"] - h["start"]) == sorted(g["end"] - g["start"])
        assert h["start"].min() >= M0 - 50 and h["end"].max() <= M0 + 1000
        assert (h["start"].to_numpy()[1:] >= h["end"].to_numpy()[:-1]).all()  # no self-overlap


def test_null_model_is_seeded_and_significance():
    inc = inc_df()
    spans = {m: (M0, M0 + 10_000) for m in inc["entity_id"]}
    a = E.null_model(inc, spans, (10, 0), n=20, seed=3)
    b = E.null_model(inc, spans, (10, 0), n=20, seed=3)
    assert all((a[g] == b[g]).all() for g in (10, 0)) and a[10].shape == (20, 10)
    pairs = E.all_pairs(inc["entity_id"])
    obs = E.edge_weights(inc, 10)
    sig = E.significant_edges(obs, a[10], pairs)
    assert list(sig.columns) == ["src", "dst", "weight", "null_mean", "p", "sig"]
    assert ((sig["p"] > 0) & (sig["p"] <= 1)).all()
    assert sig["sig"].tolist() == (sig["p"] <= 1 / 21 + 1e-12).tolist()
    summ = E.null_summary(obs, a[10])
    assert summ["observed_edges"] == 5 and summ["observed_weight"] == 6 and summ["chance_edges_mean"] >= 0


def test_significance_and_chance_count_on_handmade_null():
    pairs = [("a", "b"), ("a", "c")]
    null_w = np.array([[0, 1], [1, 1], [0, 2]])
    obs = pd.DataFrame({"src": ["a", "a"], "dst": ["b", "c"], "weight": [2, 2]})
    sig = E.significant_edges(obs, null_w, pairs)
    assert sig["p"].tolist() == [0.25, 0.5] and sig["sig"].tolist() == [True, False]
    # leave-one-out: shuffle 1 beats the others on (a,b); shuffle 2 on (a,c)
    assert E.chance_sig_edges(null_w) == pytest.approx(2 / 3)


def test_alarm_waves_chain_incidents_across_machines():
    w = E.alarm_waves(inc_df(), 10)
    assert w["machines"].tolist() == [3, 3, 1] and w["incidents"].tolist() == [3, 3, 1]
    assert w["members"].tolist()[0] == ["machine-1-1", "machine-1-2", "machine-3-1"]
    assert w["members"].tolist()[1] == ["machine-1-1", "machine-1-2", "machine-2-1"]
    assert (w["start"] - M0).tolist() == [0, 95, 500] and (w["end"] - M0).tolist() == [20, 120, 505]
    w0 = E.alarm_waves(inc_df(), 0)  # touching (3-1 at 15) and a 5-minute gap (2-1 at 115) split without a gap
    assert w0["machines"].tolist() == [2, 1, 2, 1, 1]
    assert E.alarm_waves(inc_df().iloc[:0], 10).empty


def test_wave_summary_against_handmade_null():
    null = np.array([[1, 0, 0, 2], [2, 0, 0, 3], [0, 0, 0, 1]])  # columns: >=3, >=5, >=8 machines, largest
    w = E.wave_summary(inc_df(), null, M0, top=5)
    r3 = w["by_size"][0]
    assert [r["min_machines"] for r in w["by_size"]] == list(E.WAVE_SIZES)
    assert r3["observed"] == 2 and r3["incidents"] == 6 and r3["p"] == pytest.approx(2 / 4)
    assert w["by_size"][1]["observed"] == 0 and w["by_size"][1]["p"] == pytest.approx(1.0)
    assert w["largest"] == 3 and w["p_largest"] == pytest.approx(2 / 4) and w["chance_largest_max"] == 3
    assert [t["start_h"] for t in w["top"]] == [0.0, 95 / 60] and w["top"][0]["minutes"] == 20


def test_null_model_records_waves():
    inc = inc_df()
    spans = {m: (M0, M0 + 3_000) for m in inc["entity_id"]}
    a = E.null_model(inc, spans, (10, 0), n=6, seed=2)
    assert a["waves"].shape == (6, len(E.WAVE_SIZES) + 1)
    assert (a["waves"][:, -1] >= 1).all() and (a["waves"][:, 0] >= a["waves"][:, 1]).all()


def test_build_graphs_shapes():
    inc = inc_df()
    spans = {m: (M0, M0 + 2_000) for m in inc["entity_id"]}
    vert, graphs, null = E.build_graphs(inc, spans, n_shuffles=10, seed=1)
    assert len(vert) == 5 and set(graphs) == set(E.GRAPHS)
    assert graphs["sig"]["sig"].all() and len(graphs["sig"]) == null["gap10"]["sig_edges"]
    assert null["p_threshold"] == pytest.approx(1 / 11)


def test_partitions_ari_topk_and_mixing():
    a = {"x": 1, "y": 1, "z": 2}
    assert E.groups_of(a) == [["x", "y"], ["z"]]
    assert E.same_partition(a, {"x": 7, "y": 7, "z": 0}) and not E.same_partition(a, {"x": 1, "y": 2, "z": 2})
    assert E.ari(a, {"x": "p", "y": "p", "z": "q"}) == pytest.approx(1.0)
    assert E.top_k({"a": 1.0, "b": 3.0, "c": 3.0}, 2) == ["b", "c"]
    rc = E.rank_compare({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.1, "c": 3.0}, k=2)
    assert rc["top_same_order"] and rc["spearman"] == pytest.approx(1.0) and rc["max_abs_diff"] == pytest.approx(0.1)
    edges = E.edge_weights(inc_df(), 10)
    mix = E.group_mixing(edges, inc_df()["entity_id"])
    assert mix["within_edges"] == 1 and mix["within_weight_share"] == pytest.approx(2 / 6)
    assert mix["within_pair_share"] == pytest.approx(2 / 10)


def test_pagerank_reference_matches_networkx_scaled():
    nx = pytest.importorskip("networkx")
    edges = pd.DataFrame({"src": ["a", "a", "b"], "dst": ["b", "c", "c"], "weight": [3, 1, 1]})
    vert = E.vertices(["a", "b", "c"])
    ours = E.pagerank_reference(vert, edges, iters=200)
    g = nx.Graph()
    g.add_weighted_edges_from(edges.itertuples(index=False))
    ref = nx.pagerank(g, alpha=0.85, weight="weight", tol=1e-12)
    # GDS formula = N x the normalized formula when there are no dangling nodes
    for m in ours:
        assert ours[m] == pytest.approx(3 * ref[m], rel=1e-6)


def fake_results(machines):
    lab = {m: E.smd_group(m) for m in machines}
    pr = {m: 1.0 + k / 10 for k, m in enumerate(machines)}
    return {"components": {m: 0 for m in machines}, "lpa": lab, "pagerank": pr}, \
        {"wcc": {m: 5 for m in machines}, "louvain": lab, "lpa": lab, "pagerank": dict(pr),
         "louvain_modularity": 0.1}


def write_fake_source(tmp_path, source, inc):
    spans = {m: (int(inc["start"].min()), int(inc["start"].min()) + 2_000) for m in inc["entity_id"]}
    vert, graphs, null = E.build_graphs(inc, spans, n_shuffles=10, seed=1)
    E.write_graph_files(tmp_path, vert, graphs, source)
    p_res, n_res = {}, {}
    for g in E.GRAPHS:
        p_res[g], n_res[g] = fake_results(list(vert["id"]))
    pregel = {"source": source, "incidents": "x", "scores": "y", "n_incidents": len(inc), "n_machines": len(vert),
              "machines_with_incidents": int(inc["entity_id"].nunique()),
              "span_minutes": {"min": 2_000, "max": 2_000}, "incident_minutes": {"median": 5.0, "mean": 6.0},
              "null": null, "results": p_res,
              "env": {"master": "local[*]", "default_parallelism": 2, "spark": "3.5.1", "graphframes": "gf"},
              "spark_edges_equal_pandas": {"gap10": True, "gap0": True},
              "timings": {"total_s": 1.0, "graphs_and_null_s": 0.1, "spark_start_s": 0.2,
                          **{f"{g}_s": 0.3 for g in E.GRAPHS}}}
    first = sorted(vert["id"])[0]
    neo = {"env": {"neo4j": "5.26.0 community", "gds": "2.13.2"}, "results": n_res,
           "queries": [{"title": "Q1", "cypher": "MATCH (n) RETURN n.id AS id",
                        "rows": [{"id": first, "w": 0.5, "l": ["a", "b"]}]},
                       {"title": "Q2", "cypher": "RETURN 1", "rows": []}],
           "timings": {"total_s": 2.0, "load_s": 0.5, "algorithms_s": 1.0, "queries_s": 0.1}}
    E.out_file(tmp_path, "pregel", source, "json").write_text(json.dumps(pregel))
    E.out_file(tmp_path, "neo4j", source, "json").write_text(json.dumps(neo))


def alibaba_inc():
    return inc_df().assign(entity_id=lambda d: d["entity_id"].map(
        {"machine-1-1": "m_1", "machine-1-2": "m_2", "machine-2-1": "m_3", "machine-3-1": "m_4",
         "machine-3-2": "m_5"}))


def test_report_end_to_end_both_sources(tmp_path):
    pytest.importorskip("networkx")  # the plot (requirements-graph.txt)
    write_fake_source(tmp_path, "smd", inc_df())
    write_fake_source(tmp_path, "alibaba", alibaba_inc())
    ai = tmp_path / "input.json"
    ai.write_text(json.dumps({"total_s": 9.0, "raw_s": 1.0, "score_s": 5.0, "incidents_s": 3.0,
                              "scores": {"windows": 100, "flagged": 7}}))
    md, js, png = tmp_path / "r.md", tmp_path / "r.json", tmp_path / "r.png"
    assert E.main(["report", "--out-dir", str(tmp_path), "--md", str(md), "--json", str(js), "--png", str(png),
                   "--alibaba-input", str(ai)]) == 0
    text = md.read_text(encoding="utf-8")
    assert text.index("## Part 1 -- SMD") < text.index("## Part 2 -- Alibaba")
    assert "SMD timestamps are synthetic" in text and "Alibaba has no labels" in text
    assert "## SMD vs Alibaba" in text and "share of edges beyond chance" in text
    assert "| gap10 | yes |" in text and "machine-1-1 | 0.500 | a, b" in text and "m_1 | 0.500 | a, b" in text
    assert text.count("spark-submit --master spark://spark-master:7077") == 2 and "(no rows)" in text
    assert "Do the groups follow the SMD machine groups" in text
    assert text.count("Do the groups follow") == 1  # SMD only
    assert "100 windows, 7 flagged" in text
    doc = json.loads(js.read_text())
    assert doc["sources"] == ["smd", "alibaba"]
    assert doc["smd"]["compare"]["gap10"]["ari_vs_smd_group"]["louvain"] == 1
    assert "ari_vs_smd_group" not in doc["alibaba"]["compare"]["gap10"]
    assert doc["comparison"]["alibaba"]["machines"] == 5 and doc["comparison"]["smd"]["components_same_all"]
    assert png.stat().st_size > 10_000 and (tmp_path / "r_alibaba.png").stat().st_size > 10_000


def test_plot_alibaba_with_incident_timeline(tmp_path):
    pytest.importorskip("networkx")
    inc = alibaba_inc()
    spans = {m: (M0, M0 + 2_000) for m in inc["entity_id"]}
    vert, graphs, null = E.build_graphs(inc, spans, n_shuffles=5, seed=1)
    waves = {**null["waves"], "top": [{"start_h": 0.0, "minutes": 20, "machines": 8, "incidents": 3,
                                       "members": ["m_1", "m_2", "m_4"]}]}
    neo = {"results": {g: {"pagerank": {m: 1.0 for m in vert["id"]}} for g in E.GRAPHS}}
    out = tmp_path / "t.png"
    E.plot(out, vert, graphs["gap10"], neo, "alibaba", inc, waves)
    assert out.stat().st_size > 10_000


def test_report_with_one_source_and_none(tmp_path):
    pytest.importorskip("networkx")
    md, js, png = tmp_path / "r.md", tmp_path / "r.json", tmp_path / "r.png"
    args = ["report", "--out-dir", str(tmp_path), "--md", str(md), "--json", str(js), "--png", str(png),
            "--alibaba-input", str(tmp_path / "missing.json")]
    assert E.main(args) == 2
    write_fake_source(tmp_path, "alibaba", alibaba_inc())
    assert E.main(args) == 0
    text = md.read_text(encoding="utf-8")
    assert "## Part 2 -- Alibaba" in text and "## Part 1 -- SMD" not in text


# ---- Spark / GraphFrames (needs the package from the spark-packages repo or the local Ivy cache) -----------------
@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    try:
        s = E.build_spark(master="local[2]", ckpt=tmp_path_factory.mktemp("ckpt"), app="test_e8")
        import graphframes  # noqa: F401
    except Exception as exc:  # pragma: no cover - offline machine without the package in the Ivy cache
        pytest.skip(f"GraphFrames not available: {exc}")
    yield s
    s.stop()


def test_spark_edges_cc_and_pregel_pagerank(spark):
    inc = inc_df()
    for gap in (10, 0):
        assert E.spark_edges(spark, inc, gap).equals(E.edge_weights(inc, gap))
    vert = E.vertices(inc["entity_id"])
    edges = E.edge_weights(inc, 10)
    pr = E.pregel_pagerank(spark, vert, edges, iters=6)
    ref = E.pagerank_reference(vert, edges, iters=6)
    assert pr.keys() == ref.keys() and all(pr[m] == pytest.approx(ref[m], abs=1e-12) for m in ref)
    assert pr["machine-3-2"] == pytest.approx(0.15)  # isolated: (1 - d), as in Neo4j GDS
    cc = E._graphframe(spark, vert, edges).connectedComponents()
    comps = {r["id"]: r["component"] for r in cc.collect()}
    assert E.groups_of(comps) == [["machine-1-1", "machine-1-2", "machine-2-1", "machine-3-1"], ["machine-3-2"]]
