import importlib.util
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import data

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "src/dashboard/app.py")


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Write the tools/fake_data.py scores and incidents into tmp_path and point data.py at it."""
    spec = importlib.util.spec_from_file_location("fake_data", ROOT / "tools/fake_data.py")
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    monkeypatch.setattr(fake, "ROOT", tmp_path)
    scores = fake.scores(fake.smd_rows(fake.random.Random(42)))
    fake.write_layer("scores", scores)
    fake.write_layer("incidents", fake.incidents(scores))
    # AppTest runs the script in this process, so it uses these patched defaults.
    monkeypatch.setattr(data.load_scores, "__defaults__", ("smd", tmp_path))
    monkeypatch.setattr(data.load_incidents, "__defaults__", ("smd", tmp_path))
    return tmp_path


def test_app_on_fake_data(fake_root):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [(m.label, m.value) for m in at.metric] == [
        ("Windows scored", "36"), ("Anomalous windows", "3"), ("Incidents", "1")]
    assert "machine-1-2" in at.sidebar.selectbox[1].options
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 1


def test_app_alibaba_without_data(fake_root):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.selectbox[0].select("alibaba").run()
    assert not at.exception
    assert any(i.value.startswith("No scores") for i in at.info)
    assert any(i.value.startswith("No incidents") for i in at.info)
