#!/usr/bin/env python3
"""Estimate engine wall time with SYNTHETIC weights (no real checkpoints).

Kaggle CPU runs of the K5/K6 gates are wall-clock bound by k5_runner: the
sortformer stack is ~35 layers over a 260-frame sliding window per chunk, and
naive C++ is far from real-time. Before pushing a kernel it is worth knowing
whether the paid session will take minutes or hours.

Method: run a SMALLER synthetic model (random weights, real geometry) and
extrapolate by d_model^2 * n_layers — the dominant matmul term. Measured on
this repo's engine (phone-class ARM CPU):
    d=128, 8 layers   ->   566 ms/chunk
    d=256, 8 layers   ->  1878 ms/chunk   (x3.3, close to the x4 d^2 model)
  => real config (d=512, 35 layers) ~= 30-40 s/chunk
  => mid case (224 chunks) ~= 2.2 h per feed.

Usage:
    python3 tools/bench_engine_synth.py [--seconds 60] [--d-model 128]
                                        [--layers 4] [--runner PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle" / "m2_stage3"))
import f32bin_writer as w  # noqa: E402

REAL = {"d_model": 512, "encoder_layers": 17, "encoder_d_ff": 2048,
        "transformer_layers": 18, "transformer_hidden": 512,
        "transformer_inner": 2048, "num_speakers": 4}


def shapes_for(c: dict, subsampling_channels: int = 64) -> dict:
    """name -> shape (formula mirror of SortformerWeights::collect_slots)."""
    D, C, F, K = c["d_model"], subsampling_channels, 128, 9
    FF, X, I, SPK = (c["encoder_d_ff"], c["transformer_hidden"],
                     c["transformer_inner"], c["num_speakers"])
    fbins = w.subsampling_freq_bins(F, 8)
    s: dict = {
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
            p + "self_attn.pos_bias_u": (8, D // 8), p + "self_attn.pos_bias_v": (8, D // 8),
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
    for li in range(c["transformer_layers"]):
        p = f"transformer.layers.{li}."
        s.update({
            p + "layer_norm_1.weight": (X,), p + "layer_norm_1.bias": (X,),
            p + "layer_norm_2.weight": (X,), p + "layer_norm_2.bias": (X,),
            p + "first_sub_layer.query_net.weight": (X, X), p + "first_sub_layer.query_net.bias": (X,),
            p + "first_sub_layer.key_net.weight": (X, X), p + "first_sub_layer.key_net.bias": (X,),
            p + "first_sub_layer.value_net.weight": (X, X), p + "first_sub_layer.value_net.bias": (X,),
            p + "first_sub_layer.out_projection.weight": (X, X), p + "first_sub_layer.out_projection.bias": (X,),
            p + "second_sub_layer.dense_in.weight": (I, X), p + "second_sub_layer.dense_in.bias": (I,),
            p + "second_sub_layer.dense_out.weight": (X, I), p + "second_sub_layer.dense_out.bias": (X,),
        })
    s.update({
        "encoder_proj.weight": (X, D), "encoder_proj.bias": (X,),
        "head.first_hidden_to_hidden.weight": (X, X), "head.first_hidden_to_hidden.bias": (X,),
        "head.single_hidden_to_spks.weight": (SPK, X), "head.single_hidden_to_spks.bias": (SPK,),
        "preprocessor.fb": (F, 257),
    })
    return s


def cfg_pairs(c: dict, subsampling_channels: int = 64) -> list[tuple[str, str]]:
    return [
        ("sortformer.encoder.d_model", str(c["d_model"])),
        ("sortformer.encoder.n_layers", str(c["encoder_layers"])),
        ("sortformer.encoder.n_heads", "8"),
        ("sortformer.encoder.d_ff", str(c["encoder_d_ff"])),
        ("sortformer.encoder.conv_kernel_size", "9"),
        ("sortformer.encoder.subsampling_factor", "8"),
        ("sortformer.encoder.subsampling_conv_channels", str(subsampling_channels)),
        ("sortformer.encoder.feat_in", "128"),
        ("sortformer.encoder.xscaling", "1"),
        ("sortformer.encoder.pos_emb_max_len", "5000"),
        ("sortformer.transformer.n_layers", str(c["transformer_layers"])),
        ("sortformer.transformer.hidden_size", str(c["transformer_hidden"])),
        ("sortformer.transformer.inner_size", str(c["transformer_inner"])),
        ("sortformer.transformer.n_heads", "8"),
        ("sortformer.num_speakers", "4"),
        ("sortformer.preprocessor.sample_rate", "16000"),
        ("sortformer.preprocessor.window_size", "0.025"),
        ("sortformer.preprocessor.window_stride", "0.01"),
        ("sortformer.preprocessor.n_fft", "512"),
        ("sortformer.preprocessor.features", "128"),
        ("sortformer.preprocessor.preemph", "0.97"),
        ("sortformer.preprocessor.log_zero_guard", w.LOG_ZERO_GUARD_TEXT),
        ("sortformer.scoring.spkcache_sil_frames_per_spk", "3"),
        ("sortformer.scoring.pred_score_threshold", "0.25"),
        ("sortformer.scoring.scores_boost_latest", "0.05"),
        ("sortformer.scoring.sil_threshold", "0.2"),
        ("sortformer.scoring.strong_boost_rate", "0.75"),
        ("sortformer.scoring.weak_boost_rate", "1.5"),
        ("sortformer.scoring.min_pos_scores_rate", "0.5"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4,
                    help="per-stack layer count (encoder and transformer)")
    ap.add_argument("--subsampling-channels", type=int, default=64)
    ap.add_argument("--mid-chunks", type=int, default=224,
                    help="chunk count of the mid reference case")
    ap.add_argument("--short-chunks", type=int, default=36)
    ap.add_argument("--runner", type=Path,
                    default=REPO / ".local-build" / "k5_runner")
    args = ap.parse_args()

    d = args.d_model
    bench = {"d_model": d, "encoder_layers": args.layers, "encoder_d_ff": 4 * d,
             "transformer_layers": args.layers, "transformer_hidden": d,
             "transformer_inner": 4 * d, "num_speakers": 4}
    outdir = Path("/tmp/bench-engine")
    outdir.mkdir(parents=True, exist_ok=True)

    names = w.expected_dfw1_tensor_names(bench)
    shapes = shapes_for(bench, args.subsampling_channels)
    rng = np.random.default_rng(7)
    tensors = []
    n_params = 0
    for n in names:
        a = rng.standard_normal(shapes[n]).astype(np.float32) * 0.2
        if n.endswith("running_var"):
            a = np.abs(a) + 1.0
        if n == "preprocessor.fb":
            a = np.abs(a) + 0.01
        n_params += a.size
        tensors.append((n, a))
    path = outdir / f"bench_d{d}_l{args.layers}.dfw1"
    w.write_dfw1_v2(path, cfg_pairs(bench, args.subsampling_channels), tensors)
    print(f"bench model: {n_params/1e6:.2f}M params, file {path.stat().st_size/1e6:.1f}MB")

    n = int(16000 * args.seconds)
    pcm = (rng.standard_normal(n).astype(np.float32) * 0.1)
    af32 = outdir / "bench.f32"
    af32.write_bytes(pcm.tobytes())

    t0 = time.time()
    p = subprocess.run([str(args.runner), "--run", "--weights", str(path),
                        "--audio", str(af32), "--out", str(outdir / "bench")],
                       capture_output=True, text=True, timeout=4 * 3600)
    dt = time.time() - t0
    if p.returncode != 0:
        print("FAILED:", p.stderr[-2000:])
        return 1
    meta = json.loads((outdir / "bench.meta.json").read_text())
    chunks = meta["n_chunks"]
    per_chunk = dt / max(chunks, 1)
    print(f"audio {args.seconds}s -> {chunks} chunks in {dt:.1f}s "
          f"({per_chunk*1000:.0f} ms/chunk)")

    f = (REAL["d_model"] / d) ** 2 * (
        (REAL["encoder_layers"] + REAL["transformer_layers"]) / (2 * args.layers))
    real_chunk = per_chunk * f
    print(f"extrapolation factor d^2*L = {f:.1f}x -> ~{real_chunk:.2f}s/chunk "
          f"at the real config (d=512, 35 layers)")
    print(f"mid case  ({args.mid_chunks} chunks x 2 feeds)  ~ "
          f"{2*args.mid_chunks*real_chunk/3600:.1f} h"
          f"  [4 workers: ~{args.mid_chunks*real_chunk/3600:.1f} h]")
    print(f"short case ({args.short_chunks} chunks x 2 feeds) ~ "
          f"{2*args.short_chunks*real_chunk/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
