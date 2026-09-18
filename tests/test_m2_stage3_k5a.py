"""Mechanics tests for the m2_stage3 kernel payload (no torch, no weights).

Covers:
  * embed round-trip: embedded anchors == sibling sources, cpp b64 blobs ==
    repo files (run embed_stage3.py --check logic directly);
  * f32bin_writer DFW1 v2 format: tiny container written by the PYTHON
    writer loads in the C++ SortformerWeights::load via k5_runner
    (--probe-weights) and the python name mirror equals the C++
    --expected-names list exactly;
  * k5a gate helpers (metrics) on hand-computed cases.

The k5_runner binary is built on demand into .local-build (same recipe as
scripts/run_local_tests.sh).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
STAGE3 = REPO / "kaggle" / "m2_stage3"
BUILD = REPO / ".local-build"
RUNNER_SRC = REPO / "tools" / "k5_runner.cpp"

sys.path.insert(0, str(STAGE3))
import f32bin_writer as w  # noqa: E402


def _runner() -> Path:
    exe = BUILD / "k5_runner"
    if not exe.exists():
        BUILD.mkdir(parents=True, exist_ok=True)
        srcs = [str(RUNNER_SRC)]
        srcs += [str(p) for p in sorted((REPO / "src").glob("*.cpp"))]
        cmd = ["g++", "-std=c++17", "-O2", "-o", str(exe), *srcs, "-I", str(REPO / "include")]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    return exe


# ---------------------------------------------------------------------------
# tiny config mirror (must equal k5_runner.cpp TinyCfg; the name-list test
# below fails loudly if they drift)
# ---------------------------------------------------------------------------
TINY_CFG = {
    "d_model": 16,
    "encoder_layers": 2,
    "encoder_d_ff": 32,
    "transformer_layers": 2,
    "transformer_hidden": 16,
    "transformer_inner": 32,
    "num_speakers": 4,
}


def _tiny_shapes() -> dict:
    """name -> shape for the tiny config (formula mirror of collect_slots)."""
    c = TINY_CFG
    D, C, F, K = c["d_model"], 8, 128, 9
    FF, X, I, SPK = c["encoder_d_ff"], c["transformer_hidden"], c["transformer_inner"], c["num_speakers"]
    fbins = w.subsampling_freq_bins(F, 8)
    s: dict[str, tuple] = {
        "encoder.pre_encode.conv.0.weight": (C, 1, 3, 3),
        "encoder.pre_encode.conv.0.bias": (C,),
        "encoder.pre_encode.conv.2.weight": (C, 1, 3, 3),
        "encoder.pre_encode.conv.2.bias": (C,),
        "encoder.pre_encode.conv.3.weight": (C, C, 1, 1),
        "encoder.pre_encode.conv.3.bias": (C,),
        "encoder.pre_encode.conv.5.weight": (C, 1, 3, 3),
        "encoder.pre_encode.conv.5.bias": (C,),
        "encoder.pre_encode.conv.6.weight": (C, C, 1, 1),
        "encoder.pre_encode.conv.6.bias": (C,),
        "encoder.pre_encode.out.weight": (D, C * fbins),
        "encoder.pre_encode.out.bias": (D,),
    }
    for li in range(c["encoder_layers"]):
        p = f"encoder.layers.{li}."
        s.update({
            p + "norm_feed_forward1.weight": (D,), p + "norm_feed_forward1.bias": (D,),
            p + "feed_forward1.linear1.weight": (FF, D), p + "feed_forward1.linear1.bias": (FF,),
            p + "feed_forward1.linear2.weight": (D, FF), p + "feed_forward1.linear2.bias": (D,),
            p + "norm_self_att.weight": (D,), p + "norm_self_att.bias": (D,),
            p + "self_attn.linear_q.weight": (D, D), p + "self_attn.linear_q.bias": (D,),
            p + "self_attn.linear_k.weight": (D, D), p + "self_attn.linear_k.bias": (D,),
            p + "self_attn.linear_v.weight": (D, D), p + "self_attn.linear_v.bias": (D,),
            p + "self_attn.linear_pos.weight": (D, D),
            p + "self_attn.pos_bias_u": (4, D // 4), p + "self_attn.pos_bias_v": (4, D // 4),
            p + "self_attn.linear_out.weight": (D, D), p + "self_attn.linear_out.bias": (D,),
            p + "norm_conv.weight": (D,), p + "norm_conv.bias": (D,),
            p + "conv.pointwise_conv1.weight": (2 * D, D), p + "conv.pointwise_conv1.bias": (2 * D,),
            p + "conv.depthwise_conv.weight": (D, 1, K), p + "conv.depthwise_conv.bias": (D,),
            p + "conv.batch_norm.weight": (D,), p + "conv.batch_norm.bias": (D,),
            p + "conv.batch_norm.running_mean": (D,), p + "conv.batch_norm.running_var": (D,),
            p + "conv.pointwise_conv2.weight": (D, D), p + "conv.pointwise_conv2.bias": (D,),
            p + "norm_feed_forward2.weight": (D,), p + "norm_feed_forward2.bias": (D,),
            p + "feed_forward2.linear1.weight": (FF, D), p + "feed_forward2.linear1.bias": (FF,),
            p + "feed_forward2.linear2.weight": (D, FF), p + "feed_forward2.linear2.bias": (D,),
            p + "norm_out.weight": (D,), p + "norm_out.bias": (D,),
        })
    s.update({"encoder_proj.weight": (X, D), "encoder_proj.bias": (X,)})
    for li in range(c["transformer_layers"]):
        p = f"transformer.layers.{li}."
        s.update({
            p + "first_sub_layer.query_net.weight": (X, X), p + "first_sub_layer.query_net.bias": (X,),
            p + "first_sub_layer.key_net.weight": (X, X), p + "first_sub_layer.key_net.bias": (X,),
            p + "first_sub_layer.value_net.weight": (X, X), p + "first_sub_layer.value_net.bias": (X,),
            p + "first_sub_layer.out_projection.weight": (X, X), p + "first_sub_layer.out_projection.bias": (X,),
            p + "layer_norm_1.weight": (X,), p + "layer_norm_1.bias": (X,),
            p + "second_sub_layer.dense_in.weight": (I, X), p + "second_sub_layer.dense_in.bias": (I,),
            p + "second_sub_layer.dense_out.weight": (X, I), p + "second_sub_layer.dense_out.bias": (X,),
            p + "layer_norm_2.weight": (X,), p + "layer_norm_2.bias": (X,),
        })
    s.update({
        "head.first_hidden_to_hidden.weight": (X, X), "head.first_hidden_to_hidden.bias": (X,),
        "head.single_hidden_to_spks.weight": (SPK, X), "head.single_hidden_to_spks.bias": (SPK,),
        "preprocessor.fb": (F, 257),
    })
    return s


# ---------------------------------------------------------------------------
def test_embed_round_trip():
    """Embedded anchors must equal the sibling sources and cpp blobs."""
    runner_text = (STAGE3 / "m2_stage3_run.py").read_text(encoding="utf-8")
    sys.path.insert(0, str(STAGE3))
    import embed_stage3 as e
    texts, blobs = e.build_embedded(runner_text)
    for anchor, value in texts.items():
        got = e._read_anchor(runner_text, anchor)
        assert got == value, f"{anchor} embedded != source (run embed_stage3.py)"
    got_blobs = e._read_anchor(runner_text, "EMBEDDED_CPP_BLOBS_B64")
    assert got_blobs == blobs, "EMBEDDED_CPP_BLOBS_B64 stale (run embed_stage3.py)"


def test_embed_check_cli():
    import embed_stage3 as e
    assert e.check() == 0


def test_name_mirror_matches_cpp():
    runner = _runner()
    p = subprocess.run([str(runner), "--expected-names"], capture_output=True,
                       text=True, check=True, timeout=60)
    cpp_names = json.loads(p.stdout)
    py_names = w.expected_dfw1_tensor_names(TINY_CFG)
    assert py_names == cpp_names


def _write_tiny_dfw1(tmp_path, seed: int = 777):
    """Tiny non-degenerate DFW1 container (shared by loader/run-mode tests)."""
    runner = _runner()
    shapes = _tiny_shapes()
    names = w.expected_dfw1_tensor_names(TINY_CFG)
    rng = np.random.default_rng(seed)
    tensors = []
    for n in names:
        shape = shapes[n]
        a = rng.standard_normal(shape).astype(np.float32) * 0.4
        if n.endswith("running_var"):
            a = np.abs(a) + 1.0
        if n == "preprocessor.fb":
            # filterbank magnitudes: log(negative power) = NaN timeline
            # (same degenerate-random failure the C++ selftest guards)
            a = np.abs(a) + 0.01
        tensors.append((n, a))
    cfg = [
        ("sortformer.encoder.d_model", "16"),
        ("sortformer.encoder.n_layers", "2"),
        ("sortformer.encoder.n_heads", "4"),
        ("sortformer.encoder.d_ff", "32"),
        ("sortformer.encoder.conv_kernel_size", "9"),
        ("sortformer.encoder.subsampling_factor", "8"),
        ("sortformer.encoder.subsampling_conv_channels", "8"),
        ("sortformer.encoder.feat_in", "128"),
        ("sortformer.encoder.xscaling", "1"),
        ("sortformer.encoder.pos_emb_max_len", "512"),
        ("sortformer.transformer.n_layers", "2"),
        ("sortformer.transformer.hidden_size", "16"),
        ("sortformer.transformer.inner_size", "32"),
        ("sortformer.transformer.n_heads", "4"),
        ("sortformer.num_speakers", "4"),
        ("sortformer.preprocessor.sample_rate", "16000"),
        ("sortformer.preprocessor.window_size", "0.025"),
        ("sortformer.preprocessor.window_stride", "0.01"),
        ("sortformer.preprocessor.n_fft", "512"),
        ("sortformer.preprocessor.features", "128"),
        ("sortformer.preprocessor.preemph", "0.97"),
        ("sortformer.preprocessor.log_zero_guard", w.LOG_ZERO_GUARD_TEXT),
    ] + [
        ("sortformer.scoring.spkcache_sil_frames_per_spk", "3"),
        ("sortformer.scoring.pred_score_threshold", "0.25"),
        ("sortformer.scoring.scores_boost_latest", "0.05"),
        ("sortformer.scoring.sil_threshold", "0.2"),
        ("sortformer.scoring.strong_boost_rate", "0.75"),
        ("sortformer.scoring.weak_boost_rate", "1.5"),
        ("sortformer.scoring.min_pos_scores_rate", "0.5"),
    ]
    path = tmp_path / "tiny.dfw1"
    w.write_dfw1_v2(path, cfg, tensors)
    return path


def test_dfw1_written_by_python_loads_in_cpp(tmp_path):
    runner = _runner()
    path = _write_tiny_dfw1(tmp_path)
    p = subprocess.run([str(runner), "--probe-weights", "--weights", str(path)],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-500:]
    probe = json.loads(p.stdout)
    assert probe["ok"] is True
    assert probe["d_model"] == 16 and probe["encoder_layers"] == 2


def test_full_offline_mode_end_to_end(tmp_path):
    """--full-offline: peak-normalize + one run_chunk, raw probs, no gate.

    Mirrors upstream DiarModel::diarize_offline via the production `--offline`
    CLI semantics: pregate == postgate (no BirthGate on this path), rows ==
    subsampled_len(mel), all finite.
    """
    runner = _runner()
    weights = _write_tiny_dfw1(tmp_path)
    # 2.0 s of 16 kHz mono with headroom so peak-normalize is a no-op-ish gain
    t = np.arange(32000, dtype=np.float32) / 16000.0
    pcm = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    audio = tmp_path / "tone.f32"
    audio.write_bytes(np.ascontiguousarray(pcm, dtype="<f4").tobytes())
    prefix = str(tmp_path / "off")
    p = subprocess.run([str(runner), "--full-offline", "--weights", str(weights),
                        "--audio", str(audio), "--out", prefix],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-500:]
    meta = json.loads(Path(prefix + ".meta.json").read_text())
    assert meta["mode"] == "full-offline"
    assert meta["samples"] == 32000
    assert meta["n_frames"] > 0
    n_spk = TINY_CFG["num_speakers"]
    pre = np.fromfile(prefix + ".pregate.f32", dtype="<f4").reshape(-1, n_spk)
    post = np.fromfile(prefix + ".postgate.f32", dtype="<f4").reshape(-1, n_spk)
    assert pre.shape[0] == meta["n_frames"]
    assert np.isfinite(pre).all(), "non-finite probs (negative fb leaked?)"
    assert np.array_equal(pre, post), "full-offline must not apply BirthGate"
    assert (pre >= 0.0).all() and (pre <= 1.0).all()


def test_dfw1_bad_magic_fails_loud(tmp_path):
    runner = _runner()
    path = tmp_path / "bad.dfw1"
    path.write_bytes(b"XXXX" + b"\0" * 32)
    p = subprocess.run([str(runner), "--probe-weights", "--weights", str(path)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode != 0
    assert "DFW1" in (p.stderr + p.stdout) or "magic" in (p.stderr + p.stdout)


def test_remap_matches_upstream_rules():
    assert w.remap("encoder.layers.0.self_attn.linear_q.weight") == \
        "encoder.layers.0.self_attn.linear_q.weight"
    assert w.remap("transformer_encoder.layers.3.layer_norm_1.weight") == \
        "transformer.layers.3.layer_norm_1.weight"
    assert w.remap("sortformer_modules.first_hidden_to_hidden.weight") == \
        "head.first_hidden_to_hidden.weight"
    assert w.remap("sortformer_modules.encoder_proj.bias") == "encoder_proj.bias"
    assert w.remap("encoder.pos_enc.pe") is None
    assert w.remap("preprocessor.featurizer.window") is None
    assert w.remap("sortformer_modules.hidden_to_spks.weight") is None
    assert w.remap("encoder.layers.0.conv.batch_norm.num_batches_tracked") is None
    assert w.remap("spec_augmentation.something") is None
    assert w.remap("mystery.key") is None


def test_freq_bins_formula():
    # conv_out_len(k=3,s=2,p=1) three times: n -> (n-1)/2+1
    assert w.subsampling_freq_bins(128, 8) == 16
    assert w.subsampling_freq_bins(512, 8) == 64
    assert w.subsampling_freq_bins(80, 8) == 10


def test_k5a_metrics_hand_case():
    import importlib
    k5a = importlib.import_module("m2_stage3_k5a")
    ref = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    got = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
    m = k5a.metrics(ref, got)
    assert m["rows_compared"] == 3
    assert m["max_abs"] == pytest.approx(0.1)
    # band = prob > 0.5 per speaker (multi-label, M1 convention): [0.5,0.5]
    # bands to (0,0) on both sides, so all three rows agree.
    assert m["frame_agreement"] == pytest.approx(1.0)
    assert 0.99 <= m["cosine"] <= 1.0 + 1e-9  # fp round can graze above 1.0


# ---------------------------------------------------------------------------
# Dress rehearsal: run the FULL compare_audio gate flow with a tiny DFW1 and
# a synthetic "NeMo" npz built from the runner's own outputs. Exercises every
# npz key access (total_preds/geometry/n_chunks/compression_chunks/
# spkcache_after/fifo_after/mean_sil_emb_after/n_sil_frames_after) so key
# typos die here, not on Kaggle. All gates must come back green (identity
# comparison) except informational fields.
# ---------------------------------------------------------------------------
def test_k5a_dress_rehearsal(tmp_path):
    import importlib
    k5a = importlib.import_module("m2_stage3_k5a")
    runner = _runner()

    wpath = tmp_path / "tiny.dfw1"
    # Reuse the selftest's own writer through the runner (keeps one writer).
    subprocess.run([str(runner), "--selftest"], check=True, capture_output=True,
                   timeout=120)
    src = Path("/tmp/k5_selftest.dfw1")
    assert src.exists()
    wpath.write_bytes(src.read_bytes())

    # 3 s sine mix: 2 chunks (full 20-frame chunk0 + 18-frame tail), so the
    # G1 20-row gate and multi-chunk G3 walk are both exercised.
    n = 48000
    t = np.arange(n) / 16000.0
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 331 * t + 0.7)
             + 0.1 * np.sin(2 * np.pi * 517 * t + 1.3)).astype(np.float32)

    pcm = tmp_path / "audio.f32"
    pcm.write_bytes(np.ascontiguousarray(audio, dtype="<f4").tobytes())
    prefix = str(tmp_path / "rehearsal")
    p = subprocess.run([str(runner), "--run", "--weights", str(wpath),
                        "--audio", str(pcm), "--out", prefix,
                        "--feed", "whole"],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-400:]

    ours = np.fromfile(prefix + ".pregate.f32", dtype="<f4").reshape(-1, 4)
    ledger = json.loads(Path(prefix + ".ledger.json").read_text())
    aosc_index = json.loads(Path(prefix + ".aosc.json").read_text())
    blob = np.fromfile(prefix + ".aosc.f32", dtype="<f4")
    emb_dim = 16  # tiny cfg d_model

    # Synthetic npz: reference == ours PLUS one python-NeMo phantom pad row
    # (all-zero) on the final chunk, plus the state keys the G5 derivation
    # reads (feat_length/state_lens_before/preds_full on the last chunk).
    phantom = np.zeros((1, 4), dtype=np.float64)
    tp = np.concatenate([ours.astype(np.float64), phantom], axis=0)
    last = ledger[-1]
    ci_last = f"chunk{last['chunk']:03d}"
    # final window rows = state_lens_before[0] + [1] + t3 (window geometry)
    z: dict = {
        "audio": audio,
        "total_preds": tp,
        "geometry": np.array([20, 0, 0, 80, 160, 80]),
        "n_chunks": np.array([len(ledger)]),
        "compression_chunks": np.array([], dtype=np.int64),
    }
    feat_len_last = int(last["t_mel"])
    z[f"{ci_last}/feat_length"] = np.array([feat_len_last])
    z[f"{ci_last}/state_lens_before"] = np.array(
        [0, last["fifo_frames"], last["t3"]])
    z[f"{ci_last}/preds_full"] = np.zeros(
        (last["spkcache_frames"] + last["fifo_frames"] + last["t3"] + 1, 4))
    for i, snap in enumerate(aosc_index):
        ci = f"chunk{i:03d}"
        spk = blob[snap["spk_off"]:snap["spk_off"] + snap["spk_frames"] * emb_dim] \
            .reshape(snap["spk_frames"], emb_dim)
        fifo = blob[snap["fifo_off"]:snap["fifo_off"] + snap["fifo_frames"] * emb_dim] \
            .reshape(snap["fifo_frames"], emb_dim) if snap["fifo_frames"] else \
            np.zeros((0, emb_dim), dtype=np.float32)
        mean = blob[snap["mean_off"]:snap["mean_off"] + emb_dim]
        z[f"{ci}/spkcache_after"] = spk
        z[f"{ci}/fifo_after"] = fifo
        z[f"{ci}/mean_sil_emb_after"] = mean
        z[f"{ci}/n_sil_frames_after"] = np.array([snap["silence_frames"]])

    r = k5a.compare_audio("tiny", z, runner, wpath, tmp_path, n_spk=4)
    assert "error" not in r, r.get("error")
    assert r["G1_chunk0"]["max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert r["G2_timeline"]["overall"]["max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert r["G2_timeline"]["overall"]["frame_agreement"] == pytest.approx(1.0)
    assert r["G3_aosc"]["geometry_ok"] is True
    assert r["G3_aosc"]["spk_max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert r["G3_aosc"]["mean_sil_max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert r["G4_compress"]["match"] is True
    assert r["G5_length"]["diff"] == 0
    assert r["G5_length"]["phantom_rows"] == 1
    assert r["G5_length"]["phantom_all_zero"] is True
    assert r["G6_determinism"]["bit_identical"] is True
