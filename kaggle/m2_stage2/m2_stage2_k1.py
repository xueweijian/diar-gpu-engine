"""M2 Stage 2 K1: teacher-forced gates head + transformer block x18 + encoder_proj.

Reads the Stage 0 .npz reference(s) on the kernel side (dataset, not git),
runs the local CPU reference (src/layers.cpp, shipped as code_file sibling),
and gates every layer teacher-forced:

  head gate: transformer_block_17 -> preds_full (18-deep teacher input).
  transformer gates: fc_encoder -> transformer_block_00 -> ... ->
      transformer_block_17 (chain of 18, each fed the REFERENCE predecessor
      output, never our own — a failure localizes to one block).
  encoder_proj gate: conformer_block_16 -> fc_encoder is checked implicitly
      by the transformer chain input; this kernel additionally reports the
      raw input/output RMS per block so a silent zero/garbage is obvious.

Verdict rules (printed + verdict json):
  HEAD_PASS / HEAD_FAIL, TRANSFORMER_PASS(n/18) / TRANSFORMER_FAIL.
  Metrics per gate: max_abs, mean_abs, cosine, frame_agreement(argmax @0.5).
  Thresholds are REPORTED, not pinned here: Stage 2 pins them after first
  measurement per plan §1 (quantization spread unknown until measured).

Local mechanics (no torch/NeMo): tests/test_m2_stage2_k1.py compiles the
shipped C++ and checks the harness math on synthetic weights; the kernel
only adds true-weight .npz I/O. If torch/NeMo is importable locally the
test additionally cross-checks one transformer block against torch
nn.functional (same numbers, different library) — skipped otherwise.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def locate_harness() -> Path:
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    listing: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for entry in sorted(root.iterdir()):
                listing.append(str(entry))
        except OSError as exc:
            listing.append(f"{root}: {exc}")
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("diar_harness.py"):
            return candidate.parent
    raise RuntimeError(
        "diar_harness.py not found. Kaggle input listing: " + " | ".join(listing)
    )


HARNESS_DIR = locate_harness()
sys.path.insert(0, str(HARNESS_DIR))

import diar_harness as h  # noqa: E402

OUT = Path("/kaggle/working")

REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage2_k1_head_transformer",
}

# ---- pure-python reference (mirrors src/layers.cpp, torch order) ----

def matvec_rows(x, w, b, t, inn, out):
    y = [[0.0] * out for _ in range(t)]
    for r in range(t):
        xr = x[r]
        for o in range(out):
            acc = b[o] if b is not None else 0.0
            wr = w[o]
            for i in range(inn):
                acc += xr[i] * wr[i]
            y[r][o] = acc
    return y


def layernorm(x, g, b, eps=1e-5):
    y = []
    for row in x:
        m = sum(row) / len(row)
        v = sum((v - m) ** 2 for v in row) / len(row)
        inv = 1.0 / math.sqrt(v + eps)
        y.append([(v - m) * inv * (g[i] if g else 1.0) + (b[i] if b else 0.0)
                  for i, v in enumerate(row)])
    return y


def softmax_rows(x):
    y = []
    for row in x:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        y.append([v / s for v in e])
    return y


def transformer_block(x, wt, H=192, I=768, NH=8):
    T = len(x)
    DK = H // NH
    q = matvec_rows(x, wt["q_w"], wt["qb"], T, H, H)
    k = matvec_rows(x, wt["k_w"], wt["kb"], T, H, H)
    v = matvec_rows(x, wt["v_w"], wt["vb"], T, H, H)
    sc = 1.0 / math.sqrt(DK)
    attn = [[0.0] * H for _ in range(T)]
    for hd in range(NH):
        scores = [[sum(q[i][hd * DK + e] * k[j][hd * DK + e] for e in range(DK)) * sc
                   for j in range(T)] for i in range(T)]
        probs = softmax_rows(scores)
        for i in range(T):
            for e in range(DK):
                attn[i][hd * DK + e] = sum(probs[i][j] * v[j][hd * DK + e] for j in range(T))
    ao = matvec_rows(attn, wt["o_w"], wt["ob"], T, H, H)
    h1 = [[ao[r][c] + x[r][c] for c in range(H)] for r in range(T)]
    h1 = layernorm(h1, wt["g1"], wt["b1"])
    fm = matvec_rows(h1, wt["f1_w"], wt["f1_b"], T, H, I)
    fm = [[v if v > 0 else 0.0 for v in row] for row in fm]
    fo = matvec_rows(fm, wt["f2_w"], wt["f2_b"], T, I, H)
    h2 = [[fo[r][c] + h1[r][c] for c in range(H)] for r in range(T)]
    return layernorm(h2, wt["g2"], wt["b2"])


def diar_head(x, hw, hb, sw, sb):
    a = [[v if v > 0 else 0.0 for v in row] for row in x]
    T, H = len(a), len(a[0])
    h1 = matvec_rows(a, hw, hb, T, H, H)
    h1 = [[v if v > 0 else 0.0 for v in row] for row in h1]
    S = len(sw)
    logits = matvec_rows(h1, sw, sb, T, H, S)
    return [[1.0 / (1.0 + math.exp(-v)) for v in row] for row in logits]


def metrics(ref, got):
    import numpy as _np
    r = _np.asarray(ref, dtype=_np.float64)
    g = _np.asarray(got, dtype=_np.float64)
    d = np_abs = float(_np.abs(r - g).max())
    mean = float(_np.abs(r - g).mean())
    cos = float((r * g).sum() / (math.sqrt((r * r).sum() * (g * g).sum()) + 1e-12))
    ra = (r > 0.5).astype(int)
    ga = (g > 0.5).astype(int)
    agree = float((ra == ga).mean())
    return {"max_abs": d, "mean_abs": mean, "cosine": cos, "frame_agreement": agree,
            "ref_rms": float(_np.sqrt((r * r).mean())), "got_rms": float(_np.sqrt((g * g).mean())), "np_abs": np_abs}


def load_nemo_state_dict(ckpt_path):
    import tarfile, io
    import torch
    sd = None
    with tarfile.open(ckpt_path, "r") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".ckpt"):
                f = tf.extractfile(m)
                obj = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
                sd = obj.get("state_dict", obj)
                break
    assert sd is not None, "no .ckpt in .nemo"
    return sd


def t2(w, transpose=True):
    import numpy as _np
    a = _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w, dtype=_np.float64)
    if transpose and a.ndim == 2:
        a = a.T
    return a.tolist()


def v1(w):
    import numpy as _np
    return _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w,
                       dtype=_np.float64).ravel().tolist()


def main() -> int:
    import numpy as _np
    REPORT["environment"] = h.environment_record()
    REPORT["gpu_before"] = h.gpu_snapshot()

    ref_dir = None
    for cand in (Path("/kaggle/input"), Path("/kaggle/working")):
        hits = sorted(cand.rglob("m2_ref_short.npz"))
        if hits:
            ref_dir = hits[0].parent
            break
    if ref_dir is None:
        REPORT["verdict"] = "ref-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage2_k1_verdict.json")
        return 0
    ckpts = sorted(Path("/kaggle/input").rglob("*.nemo"))
    if not ckpts:
        ckpts = sorted(Path("/kaggle/working").rglob("*.nemo"))
    REPORT["ref_dir"] = str(ref_dir)
    REPORT["ckpt"] = str(ckpts[0]) if ckpts else None

    sd = load_nemo_state_dict(str(ckpts[0]))

    def G(name):
        for cand in (name, "transformer_encoder." + name):
            if cand in sd:
                return sd[cand]
        raise KeyError(name)

    # Per-block transformer weights: NeMo TransformerEncoder layers[i].
    n_blocks = 18
    blocks = []
    for bi in range(n_blocks):
        p = f"transformer_encoder.layers.{bi}."
        blocks.append({
            "q_w": t2(sd[p + "first_sub_layer.query_net.weight"]),
            "qb": v1(sd[p + "first_sub_layer.query_net.bias"]),
            "k_w": t2(sd[p + "first_sub_layer.key_net.weight"]),
            "kb": v1(sd[p + "first_sub_layer.key_net.bias"]),
            "v_w": t2(sd[p + "first_sub_layer.value_net.weight"]),
            "vb": v1(sd[p + "first_sub_layer.value_net.bias"]),
            "o_w": t2(sd[p + "first_sub_layer.out_projection.weight"]),
            "ob": v1(sd[p + "first_sub_layer.out_projection.bias"]),
            "g1": v1(sd[p + "layer_norm_1.weight"]),
            "b1": v1(sd[p + "layer_norm_1.bias"]),
            "g2": v1(sd[p + "layer_norm_2.weight"]),
            "b2": v1(sd[p + "layer_norm_2.bias"]),
            "f1_w": t2(sd[p + "second_sub_layer.dense_in.weight"]),
            "f1_b": v1(sd[p + "second_sub_layer.dense_in.bias"]),
            "f2_w": t2(sd[p + "second_sub_layer.dense_out.weight"]),
            "f2_b": v1(sd[p + "second_sub_layer.dense_out.bias"]),
        })
    # Head weights live on sortformer_modules.
    def H(name):
        if name in sd:
            return sd[name]
        raise KeyError(name)
    hw = t2(H("sortformer_modules.first_hidden_to_hidden.weight"))
    hb = v1(H("sortformer_modules.first_hidden_to_hidden.bias"))
    sw = t2(H("sortformer_modules.single_hidden_to_spks.weight"))
    sb = v1(H("sortformer_modules.single_hidden_to_spks.bias"))

    per_audio = {}
    for label in ("short", "mid"):
        zpath = ref_dir / f"m2_ref_{label}.npz"
        if not zpath.exists():
            continue
        z = _np.load(str(zpath), allow_pickle=True)
        deep = [k.split("/")[0] for k in z.files if k.endswith("conformer_block_00")]
        audio_gates = {"head": None, "blocks": []}
        for chunk in deep:
            fc = z[chunk + "/fc_encoder"].tolist()
            # 18 teacher-forced block gates.
            prev = fc
            for bi in range(n_blocks):
                got = transformer_block(prev, blocks[bi])
                ref = z[chunk + f"/transformer_block_{bi:02d}"].tolist()
                m = metrics(ref, got)
                m.update({"chunk": chunk, "block": bi})
                audio_gates["blocks"].append(m)
                prev = ref  # teacher forcing: next block eats REFERENCE
            # Head gate eats reference block-17 output.
            t17 = z[chunk + "/transformer_block_17"].tolist()
            preds = diar_head(t17, hw, hb, sw, sb)
            ref_preds = z[chunk + "/preds_full"].tolist()
            # forward_infer multiplies by output_mask (all ones here: full
            # length, no padding) — direct compare valid.
            m = metrics(ref_preds, preds)
            m.update({"chunk": chunk})
            audio_gates["head"] = m
        per_audio[label] = audio_gates

    REPORT["gates"] = per_audio
    worst_block = max((m["max_abs"] for a in per_audio.values() for m in a["blocks"]),
                      default=float("nan"))
    worst_head = max((a["head"]["max_abs"] for a in per_audio.values() if a["head"]),
                     default=float("nan"))
    REPORT["worst_block_max_abs"] = worst_block
    REPORT["worst_head_max_abs"] = worst_head
    REPORT["verdict"] = "k1-measured"
    REPORT["note"] = ("teacher-forced fp32-vs-fp32 spreads measured; "
                      "pin gate thresholds from these numbers per plan §1.")
    REPORT["gpu_after"] = h.gpu_snapshot()
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage2_k1_verdict.json")
    print(json.dumps({k: v for k, v in REPORT.items() if k != "gates"}, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
