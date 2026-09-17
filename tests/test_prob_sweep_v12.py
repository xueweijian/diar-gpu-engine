"""v12 additions: env pins reach the child + parity candidate export.

Run:  python3 -m pytest tests/test_prob_sweep_v12.py -q
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_prob_sweep  # noqa: E402
from test_prob_sweep import _snippet, _stub_h  # noqa: E402

V12_ENV_PINS = _snippet.V12_ENV_PINS
V11_SESSION_STABLE = _snippet.V11_SESSION_STABLE
_export_parity_candidate = _snippet._export_parity_candidate


def test_v12_pins_reach_every_child_env(tmp_path, monkeypatch):
    stub, calls = _stub_h(monkeypatch, tmp_path, ["0.0 1.0 spk"])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = _snippet.run_prob_sweep(Path("/nope/bin"), [("case1", audio, None)])
    assert sweep["cases"][0]["verdict"] == "identical"
    assert len(calls) == 3
    for call in calls:
        assert call["env"] == {"CUDNN_DETERMINISTIC": "1",
                               "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def test_v12_pins_constants():
    assert V12_ENV_PINS == {"CUDNN_DETERMINISTIC": "1",
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert set(V11_SESSION_STABLE) == {"short_streaming", "mid_offline_full"}


def test_v12_export_parity_candidate_rep0_only(tmp_path, monkeypatch):
    stub, _calls = _stub_h(monkeypatch, tmp_path, ["0.0 1.0 spk"])
    stub.MODEL_PATH = Path("/fake/nemo.gguf")
    # rep0 artifacts on disk, as run_prob_sweep would have left them
    case_dir = stub.WORK_ROOT / "det_sweep" / "short_streaming"
    case_dir.mkdir(parents=True)
    (case_dir / "probs.0.f32").write_bytes(
        struct.pack("<q", 2) + struct.pack("<i", 4) + struct.pack("<8f", *([0.5] * 8)))
    (case_dir / "rep0.rttm").write_text("SPEAKER rec 1 0.0 1.0 spk <NA> <NA> <NA> <NA>\n")
    sweep = {"cases": [{"label": "short_streaming", "reps": [
        {"rep": 0, "probs_available": True, "probs_shape": [2, 4],
         "probs_sha256": "aa", "body_sha256": "bb"},
        {"rep": 1, "probs_available": True, "probs_shape": [2, 4],
         "probs_sha256": "aa", "body_sha256": "bb"},
    ]}]}
    out_dir = tmp_path / "out"
    monkeypatch.setattr(_snippet, "OUT", out_dir, raising=False)
    monkeypatch.setattr(_snippet, "REPORT", {}, raising=False)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    result = _export_parity_candidate("short_streaming", audio, None, sweep)
    assert result is not None and result["stable"] is True
    dst = out_dir / "parity_candidates" / "short_streaming"
    assert (dst / "probs.f32").exists()
    assert (dst / "rep0.rttm").exists()
    prov = json.loads((dst / "candidate.json").read_text())
    assert prov["cross_session_stable"] is True
    assert prov["env_pins"] == V12_ENV_PINS
    assert len(prov["reps"]) == 2  # provenance keeps the whole sweep
    assert prov["upstream_commit"] == "a5b6953"
    assert prov["audio_sha256"] is not None
    assert prov["preset"] is None  # sentinel-free label maps to None preset


def test_v12_export_mid_case_not_marked_stable(tmp_path, monkeypatch):
    stub, _calls = _stub_h(monkeypatch, tmp_path, ["0.0 1.0 spk"])
    stub.MODEL_PATH = Path("/fake/nemo.gguf")
    case_dir = stub.WORK_ROOT / "det_sweep" / "mid_streaming"
    case_dir.mkdir(parents=True)
    (case_dir / "probs.0.f32").write_bytes(
        struct.pack("<q", 2) + struct.pack("<i", 4) + struct.pack("<8f", *([0.5] * 8)))
    sweep = {"cases": [{"label": "mid_streaming", "reps": [
        {"rep": 0, "probs_available": True, "probs_shape": [2, 4]}]}]}
    out_dir = tmp_path / "out"
    monkeypatch.setattr(_snippet, "OUT", out_dir, raising=False)
    monkeypatch.setattr(_snippet, "REPORT", {}, raising=False)
    result = _export_parity_candidate("mid_streaming", tmp_path / "none.wav", None, sweep)
    assert result is not None and result["stable"] is False
    prov = json.loads((out_dir / "parity_candidates" / "mid_streaming" / "candidate.json").read_text())
    assert prov["cross_session_stable"] is False
    assert prov["audio_sha256"] is None  # missing audio is recorded, not fatal


def test_v12_export_skips_when_no_dump(tmp_path, monkeypatch):
    stub, _calls = _stub_h(monkeypatch, tmp_path, ["0.0 1.0 spk"])
    stub.MODEL_PATH = Path("/fake/nemo.gguf")
    monkeypatch.setattr(_snippet, "OUT", tmp_path / "out", raising=False)
    monkeypatch.setattr(_snippet, "REPORT", {}, raising=False)
    sweep = {"cases": [{"label": "x", "reps": [
        {"rep": 0, "probs_available": False}]}]}
    assert _export_parity_candidate("x", tmp_path / "a.wav", None, sweep) is None
