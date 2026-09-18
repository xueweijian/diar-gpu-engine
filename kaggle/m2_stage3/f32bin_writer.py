"""DFW1 v2 writer: .nemo -> our fp32 container (K5-A truth-anchor route).

Byte contract: include/diar/gguf.hpp ("DFW1 is our own fp32 container")
+ src/gguf.cpp F32WeightFile::load. Layout:

  u32 magic 'DFW1', u32 version=2,
  u32 n_cfg, n_cfg x {u32 klen, key, u32 vlen, val}   (decimal text values)
  u32 n_tensors, per tensor: u32 name_len, name, u32 ndim, u32 dims[ndim]
      (row-major, dims[0] = outermost), u64 n_floats, f32 data
  All integers little-endian.

Tensor names + config KVs mirror the upstream GGUF converter
(NVIDIA/NeMo-Speech.cpp a5b6953 conversion/diarization.py) but emit EVERY
tensor as raw fp32 (DFW1 has no quantization) and skip encoder.pos_enc.pe
(the DFW1 route rebuilds the rel-pos table with the pinned formula —
sortformer.hpp PE route pin).

Config values are TEXT; the C++ reader types them on read (kv_u32/kv_f32/
kv_string). Floats must be round-trippable (repr(float)); integers plain
decimal. log_zero_guard must land exactly on 5.9604644775390625e-08 — the
FE parity floor (2**-24).

Local check (no torch needed): tests/test_dfw1_pywriter.py writes a tiny
container and loads it back through tools/k5_runner --probe-weights +
--expected-names (name mirror pinned against the C++ slot list).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"DFW1"
VERSION = 2

# Exact FE floor the C++ parse_config validates against (FrontendConfig pin).
LOG_ZERO_GUARD_TEXT = "5.9604644775390625e-08"

# ---------------------------------------------------------------------------
# .nemo state_dict key -> DFW1 tensor name (conversion/diarization.py remap)
# ---------------------------------------------------------------------------
_SKIP_EXACT = {
    "encoder.pos_enc.pe",  # rebuilt analytically on the DFW1 route
    "preprocessor.featurizer.fb",  # emitted manually as preprocessor.fb
    "preprocessor.featurizer.window",  # FE builds the hann window from config
    "sortformer_modules.hidden_to_spks.weight",  # unused at inference
    "sortformer_modules.hidden_to_spks.bias",
}
_SKIP_PREFIXES = ("spec_augmentation.", "loss.")


def remap(name: str) -> str | None:
    if name in _SKIP_EXACT or name.startswith(_SKIP_PREFIXES):
        return None
    if name.endswith(".num_batches_tracked"):
        return None
    if name.startswith("sortformer_modules.encoder_proj."):
        return name.replace("sortformer_modules.", "", 1)
    if name.startswith("sortformer_modules."):
        # first_hidden_to_hidden / single_hidden_to_spks
        return name.replace("sortformer_modules.", "head.", 1)
    if name.startswith("transformer_encoder."):
        return name.replace("transformer_encoder.", "transformer.", 1)
    if name.startswith("encoder."):
        return name
    return None  # caller warns on unrecognized keys


# ---------------------------------------------------------------------------
# Expected tensor names/shapes for a config (mirrors collect_slots in
# src/sortformer.cpp; pinned against the C++ list by test_dfw1_pywriter via
# k5_runner --expected-names).
# ---------------------------------------------------------------------------
def subsampling_freq_bins(feat_in: int, factor: int) -> int:
    stages = 0
    f = factor
    while f > 1:
        if f & 1:
            raise ValueError(f"subsampling factor {factor} not a power of two")
        stages += 1
        f >>= 1
    length = feat_in
    for _ in range(stages):
        length = (length - 1) // 2 + 1
    return length


def expected_dfw1_tensor_names(c: dict) -> list[str]:
    d = c["d_model"]
    n = c["encoder_layers"]
    ff = c["encoder_d_ff"]
    x = c["transformer_hidden"]
    i = c["transformer_inner"]
    spk = c["num_speakers"]
    names: list[str] = [
        "encoder.pre_encode.conv.0.weight", "encoder.pre_encode.conv.0.bias",
        "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
        "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias",
        "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
        "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias",
        "encoder.pre_encode.out.weight", "encoder.pre_encode.out.bias",
    ]
    for li in range(n):
        p = f"encoder.layers.{li}."
        names += [
            p + "norm_feed_forward1.weight", p + "norm_feed_forward1.bias",
            p + "feed_forward1.linear1.weight", p + "feed_forward1.linear1.bias",
            p + "feed_forward1.linear2.weight", p + "feed_forward1.linear2.bias",
            p + "norm_self_att.weight", p + "norm_self_att.bias",
            p + "self_attn.linear_q.weight", p + "self_attn.linear_q.bias",
            p + "self_attn.linear_k.weight", p + "self_attn.linear_k.bias",
            p + "self_attn.linear_v.weight", p + "self_attn.linear_v.bias",
            p + "self_attn.linear_pos.weight", p + "self_attn.pos_bias_u",
            p + "self_attn.pos_bias_v", p + "self_attn.linear_out.weight",
            p + "self_attn.linear_out.bias",
            p + "norm_conv.weight", p + "norm_conv.bias",
            p + "conv.pointwise_conv1.weight", p + "conv.pointwise_conv1.bias",
            p + "conv.depthwise_conv.weight", p + "conv.depthwise_conv.bias",
            p + "conv.batch_norm.weight", p + "conv.batch_norm.bias",
            p + "conv.batch_norm.running_mean", p + "conv.batch_norm.running_var",
            p + "conv.pointwise_conv2.weight", p + "conv.pointwise_conv2.bias",
            p + "norm_feed_forward2.weight", p + "norm_feed_forward2.bias",
            p + "feed_forward2.linear1.weight", p + "feed_forward2.linear1.bias",
            p + "feed_forward2.linear2.weight", p + "feed_forward2.linear2.bias",
            p + "norm_out.weight", p + "norm_out.bias",
        ]
    names += ["encoder_proj.weight", "encoder_proj.bias"]
    for li in range(c["transformer_layers"]):
        p = f"transformer.layers.{li}."
        names += [
            p + "first_sub_layer.query_net.weight", p + "first_sub_layer.query_net.bias",
            p + "first_sub_layer.key_net.weight", p + "first_sub_layer.key_net.bias",
            p + "first_sub_layer.value_net.weight", p + "first_sub_layer.value_net.bias",
            p + "first_sub_layer.out_projection.weight", p + "first_sub_layer.out_projection.bias",
            p + "layer_norm_1.weight", p + "layer_norm_1.bias",
            p + "second_sub_layer.dense_in.weight", p + "second_sub_layer.dense_in.bias",
            p + "second_sub_layer.dense_out.weight", p + "second_sub_layer.dense_out.bias",
            p + "layer_norm_2.weight", p + "layer_norm_2.bias",
        ]
    names += [
        "head.first_hidden_to_hidden.weight", "head.first_hidden_to_hidden.bias",
        "head.single_hidden_to_spks.weight", "head.single_hidden_to_spks.bias",
        "preprocessor.fb",
    ]
    assert len(names) == len(set(names))
    return names


# ---------------------------------------------------------------------------
# Config KV derivation from a NeMo model_config.yaml dict (same sources and
# defaults as conversion/diarization.py).
# ---------------------------------------------------------------------------
_SCORING_DEFAULTS = {
    "spkcache_sil_frames_per_spk": 3,
    "pred_score_threshold": 0.25,
    "scores_boost_latest": 0.05,
    "sil_threshold": 0.2,
    "strong_boost_rate": 0.75,
    "weak_boost_rate": 1.5,
    "min_pos_scores_rate": 0.5,
}


def dfw1_config_pairs(model_cfg: dict) -> list[tuple[str, str]]:
    enc = model_cfg["encoder"]
    pp = model_cfg["preprocessor"]
    tf = model_cfg["transformer_encoder"]
    sm = model_cfg.get("sortformer_modules", {}) or {}

    def u32(key: str, v: int) -> tuple[str, str]:
        return (key, str(int(v)))

    def f32(key: str, v: float) -> tuple[str, str]:
        return (key, repr(float(v)))

    d_model = int(enc["d_model"])
    d_ff = d_model * int(enc.get("ff_expansion_factor", 4))
    feat_in = int(enc["feat_in"])
    pairs: list[tuple[str, str]] = [
        u32("sortformer.encoder.d_model", d_model),
        u32("sortformer.encoder.n_layers", int(enc["n_layers"])),
        u32("sortformer.encoder.n_heads", int(enc["n_heads"])),
        u32("sortformer.encoder.d_ff", d_ff),
        u32("sortformer.encoder.conv_kernel_size", int(enc["conv_kernel_size"])),
        u32("sortformer.encoder.subsampling_factor", int(enc.get("subsampling_factor", 8))),
        u32("sortformer.encoder.subsampling_conv_channels",
            int(enc.get("subsampling_conv_channels", 256))),
        u32("sortformer.encoder.feat_in", feat_in),
        u32("sortformer.encoder.xscaling", 1 if bool(enc.get("xscaling", True)) else 0),
        u32("sortformer.encoder.pos_emb_max_len", int(enc.get("pos_emb_max_len", 5000))),
        u32("sortformer.transformer.n_layers", int(tf["num_layers"])),
        u32("sortformer.transformer.hidden_size", int(tf["hidden_size"])),
        u32("sortformer.transformer.inner_size", int(tf["inner_size"])),
        u32("sortformer.transformer.n_heads", int(tf["num_attention_heads"])),
        u32("sortformer.num_speakers", int(model_cfg.get("max_num_of_spks", sm.get("num_spks", 4)))),
        u32("sortformer.preprocessor.sample_rate", int(pp.get("sample_rate", 16000))),
        f32("sortformer.preprocessor.window_size", float(pp.get("window_size", 0.025))),
        f32("sortformer.preprocessor.window_stride", float(pp.get("window_stride", 0.01))),
        u32("sortformer.preprocessor.n_fft", int(pp.get("n_fft", 512))),
        u32("sortformer.preprocessor.features", int(pp.get("features", feat_in))),
        f32("sortformer.preprocessor.preemph", float(pp.get("preemph", 0.97) or 0.0)),
        ("sortformer.preprocessor.log_zero_guard", LOG_ZERO_GUARD_TEXT),
    ]
    for key, default in _SCORING_DEFAULTS.items():
        val = sm.get(key, default)
        gkey = f"sortformer.scoring.{key}"
        pairs.append(u32(gkey, int(val)) if isinstance(default, int) else f32(gkey, float(val)))
    return pairs


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _pack_str(b: bytearray, s: str) -> None:
    raw = s.encode("utf-8")
    b += struct.pack("<I", len(raw))
    b += raw


def write_dfw1_v2(path: str | Path, cfg_pairs: list[tuple[str, str]],
                  tensors: list[tuple[str, np.ndarray]]) -> Path:
    b = bytearray()
    b += MAGIC
    b += struct.pack("<I", VERSION)
    b += struct.pack("<I", len(cfg_pairs))
    for k, v in cfg_pairs:
        _pack_str(b, k)
        _pack_str(b, v)
    b += struct.pack("<I", len(tensors))
    for name, arr in tensors:
        a = np.ascontiguousarray(arr, dtype="<f4")
        _pack_str(b, name)
        b += struct.pack("<I", a.ndim)
        for d in a.shape:
            b += struct.pack("<I", int(d))
        b += struct.pack("<Q", a.size)
        b += a.tobytes()
    p = Path(path)
    p.write_bytes(bytes(b))
    return p


def nemo_state_dict_to_dfw1_tensors(sd: dict, report=None) -> tuple[
        list[tuple[str, np.ndarray]], list[str]]:
    """Remap a NeMo state_dict into DFW1 tensor records (fp32 verbatim).

    Returns (tensors, warnings). The mel filterbank is squeezed from
    (1, n_mels, n_freq) to (n_mels, n_freq) exactly like the upstream
    converter (sd["preprocessor.featurizer.fb"]).
    """
    tensors: list[tuple[str, np.ndarray]] = []
    warns: list[str] = []
    fb_done = False
    for key, val in sd.items():
        if key == "preprocessor.featurizer.fb":
            fb = val.detach().cpu().float().numpy()
            fb = np.ascontiguousarray(np.squeeze(fb, axis=0))
            tensors.append(("preprocessor.fb", fb))
            fb_done = True
            continue
        out = remap(key)
        if out is None:
            continue
        arr = np.ascontiguousarray(val.detach().cpu().float().numpy())
        tensors.append((out, arr))
    if not fb_done:
        warns.append("preprocessor.featurizer.fb MISSING from state_dict")
    return tensors, warns
