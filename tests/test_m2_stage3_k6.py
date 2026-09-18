"""K6 mechanics tests (no torch, no Kaggle, no real weights).

Pins the case mapping against the FOUR REAL M1 manifests in
parity/fixtures/ (a mis-mapped preset/compare-file would silently invalidate
the whole K6 gate), the wav decoder convention, and runs the full gate CLI
end-to-end against the tiny DFW1 container (verdict is valid data either
color — mechanics only).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
STAGE3 = REPO / "kag" / "m2_stage3" if False else REPO / "kaggle" / "m2_stage3"
sys.path.insert(0, str(STAGE3))
import m2_stage3_k6 as k6  # noqa: E402

# reuse the tiny-DFW1 builder from the k5a mechanics module
sys.path.insert(0, str(REPO / "tests"))
import test_m2_stage3_k5a as k5a  # noqa: E402


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "m2_stage3_run", STAGE3 / "m2_stage3_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPECTED_MAPPING = {
    # fixture dir          -> (runner argv, compared output file)
    "v12-short-streaming-r0": (["--run"], "postgate"),
    "v13-mid-streaming-r0": (["--run"], "postgate"),
    "v13-mid-offline-preset-r0": (["--run", "--offline"], "postgate"),
    "v12-mid-offline-full-r0": (["--full-offline"], "pregate"),
}


def test_case_mapping_matches_real_manifests():
    """The four REAL M1 manifests must map to the pinned runner invocations."""
    for name, (argv, which) in EXPECTED_MAPPING.items():
        manifest = json.loads(
            (REPO / "parity" / "fixtures" / name / "manifest.json").read_text())
        got_argv, got_which = k6.classify(manifest)
        assert (got_argv, got_which) == (argv, which), \
            f"{name}: got {(got_argv, got_which)}"


def test_case_mapping_rejects_unknown_preset_shape():
    # offline flag wins over preset (production --offline is full-attention)
    m = {"case": {"offline": True, "preset": "offline"}}
    assert k6.classify(m) == k6.CASE_PLAN["full_offline"]


def test_decode_wav_int16_convention(tmp_path):
    """int16 /32768 (dr_wav convention), mono + 16k enforced."""
    p = tmp_path / "tone.wav"
    with wave.open(str(p), "wb") as w_:
        w_.setnchannels(1)
        w_.setsampwidth(2)
        w_.setframerate(16000)
        w_.writeframes(np.array([0, 16384, -16384, -32768], dtype="<i2").tobytes())
    pcm = k6.decode_wav(p)
    assert pcm.dtype == np.float32
    np.testing.assert_allclose(pcm, [0.0, 0.5, -0.5, -1.0], atol=1e-7)


def test_decode_wav_rejects_wrong_rate(tmp_path):
    p = tmp_path / "bad.wav"
    with wave.open(str(p), "wb") as w_:
        w_.setnchannels(1)
        w_.setsampwidth(2)
        w_.setframerate(44100)
        w_.writeframes(b"\0\0" * 100)
    with pytest.raises(RuntimeError, match="16 kHz"):
        k6.decode_wav(p)


def _write_fixture(tmp_path: Path, name: str, n_spk: int, n_rows: int) -> Path:
    d = tmp_path / "fixtures" / name
    d.mkdir(parents=True)
    manifest = {
        "case": {"audio": "fake.wav", "geometry": None,
                 "offline": name.endswith("full-r0"),
                 "preset": "offline" if "preset" in name else None},
        "observation": {"n_spk": n_spk, "n_frames": n_rows},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "probs.f32").write_bytes(
        np.zeros((n_rows, n_spk), dtype="<f4").tobytes())
    return d


def test_gate_cli_end_to_end_tiny(tmp_path):
    """Full gate CLI against tiny DFW1 + synthetic wav; verdict is data."""
    runner = k5a._runner()
    weights = k5a._write_tiny_dfw1(tmp_path)

    # audio under a fake mount root (gate rglobs for the manifest basename)
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    t = np.arange(32000, dtype=np.float32) / 16000.0
    pcm = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    with wave.open(str(audio_root / "fake.wav"), "wb") as w_:
        w_.setnchannels(1)
        w_.setsampwidth(2)
        w_.setframerate(16000)
        w_.writeframes(np.ascontiguousarray(pcm * 32767, dtype="<i2").tobytes())

    # Fixture row counts come from the tiny runner itself — a hardcoded
    # guess silently rots whenever the engine/frame geometry moves.
    af32 = tmp_path / "fake.f32"
    af32.write_bytes(pcm.tobytes())
    for name in ("v12-short-streaming-r0", "v12-mid-offline-full-r0"):
        argv_extra, which = EXPECTED_MAPPING[name]
        prefix = str(tmp_path / f"probe-{name}")
        subprocess.run([str(runner), *argv_extra, "--weights", str(weights),
                        "--audio", str(af32), "--out", prefix],
                       check=True, capture_output=True, text=True, timeout=300)
        n_rows = int(np.fromfile(prefix + f".{which}.f32", dtype="<f4").size // 4)
        _write_fixture(tmp_path, name, 4, n_rows)

    out = tmp_path / "k6_verdict.json"
    p = subprocess.run(
        [sys.executable, str(STAGE3 / "m2_stage3_k6.py"),
         "--runner", str(runner), "--gguf", str(weights),
         "--fixtures-dir", str(tmp_path / "fixtures"),
         "--audio-roots", str(audio_root),
         "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-800:]
    v = json.loads(out.read_text())
    assert v["verdict"] in ("k6-green", "k6-red")
    assert set(v["per_case"]) == {"v12-short-streaming-r0",
                                  "v12-mid-offline-full-r0"}
    for r in v["per_case"].values():
        assert "error" not in r, r
        assert r["rows_delta"] == 0  # synthetic fixture rows == tiny model rows
        assert {"max_abs", "mean_abs", "frame_agreement"} <= set(r)
    # offline-full case compared the PREGATE file (no BirthGate upstream)
    assert v["per_case"]["v12-mid-offline-full-r0"]["compare_file"] == "pregate"
    assert v["per_case"]["v12-short-streaming-r0"]["compare_file"] == "postgate"
    # determinism double-run attached for the streaming case
    assert v["per_case"]["v12-short-streaming-r0"]["determinism_bit_identical"] is True


# ---------------------------------------------------------------------------
# kernel-side fixture discovery (dataset may mount extracted OR as a zip)
# ---------------------------------------------------------------------------
def _case_tree(root: Path) -> Path:
    for name in ("v12-short-streaming-r0", "v12-mid-offline-full-r0"):
        d = root / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"case": {}, "observation": {}}))
        (d / "probs.f32").write_bytes(b"\0" * 16)
    return root


def test_find_fixtures_dir_extracted(tmp_path):
    mod = _runner_module()
    root = _case_tree(tmp_path / "input" / "diar-m2-fixtures")
    got = mod.find_fixtures_dir(tmp_path / "work", roots=[tmp_path / "input"])
    assert got == root


def test_find_fixtures_dir_zip(tmp_path):
    mod = _runner_module()
    zdir = tmp_path / "input" / "diar-m2-fixtures"
    zdir.mkdir(parents=True)
    src = _case_tree(tmp_path / "src")
    zpath = zdir / "fixtures.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src))
    work = tmp_path / "work"
    got = mod.find_fixtures_dir(work, roots=[tmp_path / "input"])
    assert (got / "v12-short-streaming-r0" / "probs.f32").exists()
    assert got.is_relative_to(work)


def test_find_fixtures_dir_zip_with_wrapper(tmp_path):
    """A wrapper folder inside the zip must not defeat the search."""
    mod = _runner_module()
    zdir = tmp_path / "input" / "diar-m2-fixtures"
    zdir.mkdir(parents=True)
    src = _case_tree(tmp_path / "src")
    with zipfile.ZipFile(zdir / "f.zip", "w") as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, Path("wrapper") / f.relative_to(src))
    got = mod.find_fixtures_dir(tmp_path / "work", roots=[tmp_path / "input"])
    assert (got / "v12-short-streaming-r0" / "probs.f32").exists()


def test_find_fixtures_dir_missing_raises(tmp_path):
    mod = _runner_module()
    (tmp_path / "input").mkdir()
    with pytest.raises(RuntimeError, match="diar-m2-fixtures"):
        mod.find_fixtures_dir(tmp_path / "work", roots=[tmp_path / "input"])
