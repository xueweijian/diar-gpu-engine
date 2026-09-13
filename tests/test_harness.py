"""Local tests for the Kaggle harness helpers that need no GPU, CUDA or git.

Run:  python3 -m pytest tests/test_harness.py -q
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import diar_harness as harness  # noqa: E402


@pytest.fixture()
def tmp_work(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(harness, "LOG_PATH", tmp_path / "work" / "run.log")
    monkeypatch.setattr(harness, "OUT_DIR", tmp_path / "out")
    harness.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    harness.OUT_DIR.mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_write_tone_wav_is_deterministic(tmp_work):
    first = harness.write_tone_wav(tmp_work / "a.wav", seconds=1)
    second = harness.write_tone_wav(tmp_work / "b.wav", seconds=1)
    assert harness.sha256(first) == harness.sha256(second)
    assert harness.wav_info(first)["seconds"] == 1.0


def test_wav_info_reports_shape(tmp_work):
    path = harness.write_tone_wav(tmp_work / "tone.wav", seconds=2, rate=8000)
    info = harness.wav_info(path)
    assert info == {
        "channels": 1, "sample_rate": 8000, "sample_width": 2,
        "frames": 16000, "seconds": 2.0,
    }


def test_concat_wav_scales_length(tmp_work):
    unit = harness.write_tone_wav(tmp_work / "unit.wav", seconds=1)
    merged = harness.concat_wav([unit, unit, unit], tmp_work / "merged.wav")
    assert harness.wav_info(merged)["seconds"] == 3.0
    with wave.open(str(merged), "rb") as handle:
        assert handle.getnframes() == 3 * 16000


def test_concat_wav_rejects_format_mismatch(tmp_work):
    unit = harness.write_tone_wav(tmp_work / "unit.wav", seconds=1)
    other = harness.write_tone_wav(tmp_work / "other.wav", seconds=1, rate=8000)
    with pytest.raises(ValueError, match="format mismatch"):
        harness.concat_wav([unit, other], tmp_work / "bad.wav")


def test_concat_wav_requires_input(tmp_work):
    with pytest.raises(ValueError):
        harness.concat_wav([], tmp_work / "none.wav")


def test_parse_rttm_aggregates_speakers(tmp_work):
    path = tmp_work / "x.rttm"
    path.write_text(
        "SPEAKER rec 1 0.000 1.500 <NA> <NA> spk1 <NA> <NA>\n"
        "SPEAKER rec 1 2.000 0.500 <NA> <NA> spk2 <NA> <NA>\n"
        "SPEAKER rec 1 3.000 1.000 <NA> <NA> spk1 <NA> <NA>\n",
        encoding="utf-8",
    )
    parsed = harness.parse_rttm(path)
    assert parsed["segments"] == 3
    assert parsed["speaker_count"] == 2
    assert parsed["speakers"] == ["spk1", "spk2"]
    assert parsed["speech_seconds"] == {"spk1": 2.5, "spk2": 0.5}
    assert parsed["last_end_seconds"] == 4.0


def test_parse_rttm_handles_empty_and_missing(tmp_work):
    assert harness.parse_rttm(tmp_work / "missing.rttm")["segments"] == 0
    empty = tmp_work / "empty.rttm"
    empty.write_text("", encoding="utf-8")
    assert harness.parse_rttm(empty)["speaker_count"] == 0


def test_slugify_normalises_unsafe_characters():
    assert harness.slugify("../../etc/passwd") == ".._.._etc_passwd"
    assert harness.slugify("中文 audio 1") == "___audio_1"
    assert len(harness.slugify("x" * 100)) == 40


def test_gpu_sampler_summary_from_samples(tmp_work):
    sampler = harness.GpuSampler()
    sampler.samples = [
        {"vram_used_mib": 100.0, "power_w": 60.0, "temperature_c": 40.0,
         "sm_clock_mhz": 1100.0, "mem_clock_mhz": 715.0, "util_percent": 10.0},
        {"vram_used_mib": 300.0, "power_w": 120.0, "temperature_c": 55.0,
         "sm_clock_mhz": 1300.0, "mem_clock_mhz": 715.0, "util_percent": 90.0},
    ]
    summary = sampler.summary()
    assert summary["peak_vram_mib"] == 300.0
    assert summary["peak_power_w"] == 120.0
    assert summary["median_sm_clock_mhz"] == 1200.0
    assert summary["first_sm_clock_mhz"] == 1100.0
    assert summary["last_sm_clock_mhz"] == 1300.0


def test_benchmark_record_matches_schema_requirements():
    record = harness.benchmark_record(
        commit="abc1234", stage="unit",
        environment={"os": "linux", "compiler": "gcc", "gpu": "none"},
        workload={"model": "m", "fixture": "f", "precision": "FP32", "batch": 1},
        timing={"audio_seconds": 10.0, "wall_seconds": 1.0, "rtf": 0.1, "realtime_x": 10.0},
    )
    for key in ("schema_version", "commit", "scope", "environment", "workload", "timing", "accuracy"):
        assert key in record
    assert record["schema_version"] == 1
    assert record["scope"] == "pure_speaker_diarization"
    assert record["accuracy"] == {"status": "not_scored"}


def test_append_jsonl_roundtrip(tmp_work):
    path = tmp_work / "r.jsonl"
    harness.append_jsonl(path, [{"a": 1}, {"b": 2}])
    harness.append_jsonl(path, [{"c": 3}])
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_as_float_tolerates_gpu_placeholders():
    assert harness._as_float(" 123 ") == 123.0
    assert harness._as_float("[N/A]") is None
    assert harness._as_float("") is None


def test_run_reports_missing_executable(tmp_work):
    code, text, _ = harness.run(["definitely-not-a-real-binary-xyz"], timeout=10)
    assert code == 127
    assert "missing executable" in text
