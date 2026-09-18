"""K6 mechanics tests (no torch, no Kaggle, no real weights).

Pins the case mapping against the FOUR REAL M1 manifests in
parity/fixtures/ (a mis-mapped preset/compare-file would silently invalidate
the whole K6 gate), the wav decoder convention, and runs the full gate CLI
end-to-end against the tiny DFW1 container (verdict is valid data either
color — mechanics only).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
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


# ---------------------------------------------------------------------------
# fixture frame_probs wire format (12B <qi> header + f32 payload, sha-pinned)
# ---------------------------------------------------------------------------
def test_load_probs_f32_equals_canonical_loader_on_real_fixture():
    """Pin the K6 parser to parity/loader.py on the REAL short fixture."""
    fix = REPO / "parity" / "fixtures" / "v12-short-streaming-r0"
    manifest = json.loads((fix / "manifest.json").read_text())
    got = k6.load_probs_f32(fix, manifest)
    assert got.shape == (711, 4)
    sys.path.insert(0, str(REPO))
    from parity.loader import load_case  # noqa: E402
    _, tensors = load_case(fix)
    canon = tensors["probs"]
    assert (canon.n_frames, canon.n_spk) == (711, 4)
    np.testing.assert_array_equal(
        got.ravel(), np.asarray(canon.values, dtype="<f4").reshape(-1))


def test_load_probs_f32_real_mid_shape():
    fix = REPO / "parity" / "fixtures" / "v12-mid-offline-full-r0"
    manifest = json.loads((fix / "manifest.json").read_text())
    got = k6.load_probs_f32(fix, manifest)
    assert got.shape == (4467, 4)
    assert got.dtype == np.float32


def test_load_probs_f32_rejects_header_manifest_mismatch(tmp_path):
    d = tmp_path / "fx"
    d.mkdir()
    payload = np.zeros((19, 4), dtype="<f4").tobytes()
    raw = struct.pack("<qi", 20, 4) + payload  # header disagrees with manifest
    (d / "probs.f32").write_bytes(raw)
    manifest = {
        "observation": {"n_frames": 19, "n_spk": 4},
        "tensors": [{"name": "probs", "layer": "frame_probs", "dtype": "f32",
                     "shape": [19, 4], "nbytes": len(payload),
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "path": "probs.f32"}],
    }
    with pytest.raises(RuntimeError, match="header"):
        k6.load_probs_f32(d, manifest)


def test_load_probs_f32_rejects_sha_mismatch(tmp_path):
    d = tmp_path / "fx"
    d.mkdir()
    payload = np.zeros((19, 4), dtype="<f4").tobytes()
    raw = struct.pack("<qi", 19, 4) + payload
    (d / "probs.f32").write_bytes(raw)
    manifest = {
        "observation": {"n_frames": 19, "n_spk": 4},
        "tensors": [{"name": "probs", "layer": "frame_probs", "dtype": "f32",
                     "shape": [19, 4], "nbytes": len(payload),
                     "sha256": "0" * 64, "path": "probs.f32"}],
    }
    with pytest.raises(RuntimeError, match="sha256"):
        k6.load_probs_f32(d, manifest)


def _write_fixture(tmp_path: Path, name: str, n_spk: int, n_rows: int) -> Path:
    d = tmp_path / "fixtures" / name
    d.mkdir(parents=True)
    payload = np.zeros((n_rows, n_spk), dtype="<f4").tobytes()
    raw = struct.pack("<qi", n_rows, n_spk) + payload  # probdump wire format
    (d / "probs.f32").write_bytes(raw)
    manifest = {
        "case": {"audio": "fake.wav", "geometry": None,
                 "offline": name.endswith("full-r0"),
                 "preset": "offline" if "preset" in name else None},
        "observation": {"n_spk": n_spk, "n_frames": n_rows},
        "tensors": [{
            "name": "probs", "layer": "frame_probs", "dtype": "f32",
            "shape": [n_rows, n_spk], "nbytes": len(payload),
            "sha256": hashlib.sha256(raw).hexdigest(), "path": "probs.f32",
        }],
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
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
# K6 kernel entry file (generated: runner with DEFAULT_GATE flipped to "k6")
# ---------------------------------------------------------------------------
def _k6_entry_path() -> Path:
    return REPO / "kaggle" / "m2_stage3_k6" / "m2_stage3_run_k6.py"


def test_k6_entry_is_runner_with_gate_flipped():
    """The K6 kernel ships a generated copy — exactly one line may differ."""
    src = (STAGE3 / "m2_stage3_run.py").read_text(encoding="utf-8")
    assert src.count('DEFAULT_GATE = "k5a"') == 1
    got = _k6_entry_path().read_text(encoding="utf-8")
    assert got == src.replace('DEFAULT_GATE = "k5a"', 'DEFAULT_GATE = "k6"'), \
        "K6 entry stale — run scripts/build_k6_entry.py"


def test_k6_entry_embedded_payload_is_fresh():
    """The generated entry must carry the same fresh embedded anchors."""
    sys.path.insert(0, str(STAGE3))
    import embed_stage3 as e
    text = _k6_entry_path().read_text(encoding="utf-8")
    texts, blobs = e.build_embedded(text)
    for anchor, value in texts.items():
        assert e._read_anchor(text, anchor) == value, \
            f"{anchor} stale in K6 entry (embed_stage3.py + build_k6_entry.py)"
    assert e._read_anchor(text, "EMBEDDED_CPP_BLOBS_B64") == blobs, \
        "EMBEDDED_CPP_BLOBS_B64 stale in K6 entry"


def test_gate_pins():
    """Both files pin their gate; a flip is a conscious, test-visible change."""
    src = (STAGE3 / "m2_stage3_run.py").read_text(encoding="utf-8")
    k6 = _k6_entry_path().read_text(encoding="utf-8")
    assert 'DEFAULT_GATE = "k5a"' in src
    assert 'DEFAULT_GATE = "k6"' in k6
    assert 'DEFAULT_GATE = "k5a"' not in k6


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


def test_pack_fixtures_zip_layout(tmp_path):
    """The publish zip must carry <case>/<file> entries at the TOP level."""
    spec = importlib.util.spec_from_file_location(
        "pack_fixtures", REPO / "scripts" / "pack_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.pack(REPO / "parity" / "fixtures", tmp_path / "fixtures.zip")
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    for case in mod.CASES:
        assert f"{case}/manifest.json" in names
        assert f"{case}/probs.f32" in names
    # no wrapper dir: find_fixtures_dir() searches for "<case>/" under the mount
    assert all(any(n.startswith(f"{c}/") for c in mod.CASES) for n in names)
