"""Tests for scripts/fill_parity_fixture.py (synthetic candidates, no kernel).

Run:  python3 -m pytest tests/test_fill_parity_fixture.py -q

The fixture pipeline is the truth intake for parity L1, so these tests pin
the discipline: exact-observation sha check, stable-label gate, wire-format
validation, and a successful end-to-end fill that the parity loader accepts.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fill_parity_fixture as fpf  # noqa: E402

GOOD_SHA = "ab" * 32


def _make_candidate(tmp_path: Path, label: str = "short_streaming",
                    stable: bool = True, rep0_sha: str | None = None,
                    bodies: list[str] | None = None) -> Path:
    """Synthetic v12 candidate dir as the kernel export would create it."""
    cdir = tmp_path / "parity_candidates" / label
    cdir.mkdir(parents=True)
    payload = struct.pack(f"<{8}f", *([0.5] * 8))
    raw = struct.pack("<qi", 2, 4) + payload
    (cdir / "probs.f32").write_bytes(raw)
    sha = rep0_sha if rep0_sha is not None else hashlib.sha256(raw).hexdigest()
    prov = {
        "label": label,
        "audio_path": "/kaggle/input/datasets/diar-smoke-audio/x.wav",
        "audio_sha256": "cd" * 32,
        "preset": None,
        "offline_flag": False,
        "env_pins": {"CUDNN_DETERMINISTIC": "1",
                     "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "cross_session_stable": stable,
        "reps": [
            {"rep": i, "probs_available": True, "probs_shape": [2, 4],
             "probs_sha256": sha, "body_sha256": (bodies or ["bb" * 32])[i % len(bodies or ["bb" * 32])]}
            for i in range(3)
        ],
        "model_path": "/fake/nemo.gguf",
        "runtime_version": "nemo-speech 1.2.3",
        "gpu_before": {"name": "Tesla P100-PCIE-16GB"},
        "upstream_commit": "a5b6953",
        "harness_commit": "d00a769",
        "captured_at": "2026-09-17T00:00:00Z",
    }
    (cdir / "candidate.json").write_text(json.dumps(prov), encoding="utf-8")
    return cdir


def _fill(tmp_path: Path, cdir: Path, **kwargs) -> Path:
    return fpf.fill(
        cdir, name=kwargs.pop("name", "v12-short-streaming-r0"),
        kernel_version=kwargs.pop("kernel_version", 12),
        src_commit=kwargs.pop("src_commit", "deadbee"),
        fixtures_dir=kwargs.pop("fixtures_dir", tmp_path / "fixtures"),
        **kwargs)


def test_fill_end_to_end_and_loader_verified(tmp_path):
    cdir = _make_candidate(tmp_path)
    fixture_dir = _fill(tmp_path, cdir)
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["source"]["kernel_version"] == 12
    assert manifest["source"]["upstream_commit"] == "a5b6953"
    assert manifest["observation"]["n_frames"] == 2
    assert manifest["observation"]["n_spk"] == 4
    assert manifest["observation"]["deterministic"] is True
    assert manifest["observation"]["frame_grid_ms"] == 80
    tensor = manifest["tensors"][0]
    assert tensor["nbytes"] == 32  # logical payload only, header excluded
    assert tensor["sha256"] == hashlib.sha256(
        (fixture_dir / "probs.f32").read_bytes()).hexdigest()
    assert (fixture_dir / "probs.f32").read_bytes()[:12] == struct.pack("<qi", 2, 4)


def test_fill_rejects_sha_mismatch(tmp_path):
    cdir = _make_candidate(tmp_path, rep0_sha=GOOD_SHA)
    with pytest.raises(SystemExit, match="not the pinned observation"):
        _fill(tmp_path, cdir)


def test_fill_rejects_unstable_without_opt_in(tmp_path):
    cdir = _make_candidate(tmp_path, label="mid_streaming", stable=False)
    with pytest.raises(SystemExit, match="NOT legitimate truth"):
        _fill(tmp_path, cdir)


def test_fill_allows_unstable_with_explicit_opt_in(tmp_path):
    cdir = _make_candidate(tmp_path, label="mid_streaming", stable=False)
    fixture_dir = _fill(tmp_path, cdir, name="x", allow_unstable=True)
    assert (fixture_dir / "manifest.json").exists()


def test_fill_flags_drifted_bodies_as_not_deterministic(tmp_path):
    cdir = _make_candidate(tmp_path, bodies=["bb" * 32, "cc" * 32, "bb" * 32])
    fixture_dir = _fill(tmp_path, cdir, name="drifty")
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    assert manifest["observation"]["deterministic"] is False
    assert len(manifest["observation"]["hash_groups"]) == 2


def test_fill_rejects_truncated_dump(tmp_path):
    cdir = _make_candidate(tmp_path)
    raw = (cdir / "probs.f32").read_bytes()
    truncated = raw[:20]  # header + partial payload
    (cdir / "probs.f32").write_bytes(truncated)
    # rep0 pin follows the truncated bytes so the sha gate passes and the
    # wire-format validator is what catches the truncation.
    prov = json.loads((cdir / "candidate.json").read_text())
    prov["reps"][0]["probs_sha256"] = hashlib.sha256(truncated).hexdigest()
    (cdir / "candidate.json").write_text(json.dumps(prov))
    with pytest.raises(SystemExit, match="truncated"):
        _fill(tmp_path, cdir)


def test_fill_rejects_missing_rep0_dump(tmp_path):
    cdir = _make_candidate(tmp_path)
    prov = json.loads((cdir / "candidate.json").read_text())
    prov["reps"][0]["probs_available"] = False
    (cdir / "candidate.json").write_text(json.dumps(prov))
    with pytest.raises(SystemExit, match="refusing to fabricate"):
        _fill(tmp_path, cdir)
