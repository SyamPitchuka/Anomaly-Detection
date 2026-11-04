"""Host description of experiments/e2e_latency_runs.py: detected per run, never hard-coded."""
import json
import platform

from experiments import e2e_latency_runs as M


def test_hardware_text_detects_this_host():
    text = M.hardware_text()
    assert platform.platform() in text
    assert platform.python_version() in text
    assert "logical CPUs" in text and "GB RAM" in text


def test_hardware_text_keeps_note_and_detection():
    text = M.hardware_text("Azure Standard_D2as_v4 (2 vCPU, 8 GB)")
    assert text.startswith("Azure Standard_D2as_v4 (2 vCPU, 8 GB)")
    assert platform.platform() in text


def test_hardware_argument_reaches_meta_json(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "RUNS_DIR", tmp_path)
    real_run_once = M.run_once

    def skip_run(*args, **kwargs):
        # an impossible free-RAM floor makes run_once write meta.json and stop before touching Kafka
        return real_run_once(args[0], args[1], args[2], args[3], args[4], float("inf"), *args[6:], **kwargs)

    monkeypatch.setattr(M, "run_once", skip_run)
    assert M.main(["--runs", "100:10", "--hardware", "test VM"]) == 0
    meta = json.loads((tmp_path / "speedup_100" / "meta.json").read_text())
    assert meta["status"].startswith("skipped")
    assert meta["hardware"].startswith("test VM")
    assert platform.platform() in meta["hardware"]
