"""M2 Stage 2 K2: teacher-forced gates conformer conv + FF + full layer x17.

Same harness pattern as K1 (pure-python reference mirrored from
src/conv.cpp + src/layers.cpp + src/conformer.cpp; local mechanics tests
pin the python-vs-C++ transcription). Kernel-side: load .nemo state dict,
build per-layer weights, and gate teacher-forced on deep chunks:

  conv gate:  norm_conv(conformer input) -> conv module -> compare vs a
      locally recomputed reference is NOT available (no intermediate dump),
      so the conv gate runs INSIDE the full-layer gate: the full layer is
      computed with reference submodule inputs at each stage? No — without
      per-submodule dumps only the full-layer boundary is gateable. The K2
      verdict therefore gates full conformer_layer_00..16 (input =
      reference predecessor block output; layer 00 input = xscaled concat
      [spkcache|fifo|pre_encode] reconstructed from state_lens_before +
      fifo_after/spkcache_after? No — chunk000 has empty state, so layer-00
      input = xscale(pre_encode) + pos table; later chunks need state which
      IS in the dump (fifo_after/spkcache_after of the PREVIOUS chunk).

  Practical K2 gate set (no new dump needed):
  - layer00/chunk000..002: input = concat([], [], pre_encode) xscaled;
    pos table built locally (posenc formula, pinned by test_posenc).
  - layerNN/chunk000..002: input = reference conformer_block_{NN-1}
    (teacher forcing through the stack).
  - FF/conv submodule gates: synthetic-weight python-vs-C++ already green
    locally; true-weight submodule isolation is impossible without new
    dumps, so a full-layer FAIL localizes by re-running the layer with
    reference NORM outputs? Also unavailable. Localization fallback: swap
    in reference MHA-vs-local per K1-style recompute is possible because
    MHA inputs ARE reproducible (normed input). The kernel therefore ALSO
    reports per-stage recompute metrics (norm+MHA vs reference? no
    reference). Honest scope: K2 gates FULL layers; submodule truth waits
    for Stage 3 free-running assembly. A K2 FAIL triages by which layer
    index and which chunk length first diverges (length-dependent =
    pos-table/shift suspect; layer-0-only = xscale/concat suspect).

Verdict: K2-measured with worst_layer_max_abs. Thresholds pinned after.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path


def locate_harness() -> Path:
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("diar_harness.py"):
            return candidate.parent
    raise RuntimeError("diar_harness.py not found")


HARNESS_DIR = locate_harness()
sys.path.insert(0, str(HARNESS_DIR))

import diar_harness as h  # noqa: E402

OUT = Path("/kaggle/working")
REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage2_k2_conformer",
}

D_MODEL, D_FF, N_HEADS, KERNEL = 512, 2048, 8, 9


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


def relpos_table(L, D):
    div = [math.exp(i * -(math.log(10000.0) / D)) for i in range(D // 2)]
    pe = []
    for r in range(2 * L - 1):
        pos = (L - 1) - r
        row = []
        for i in range(D // 2):
            row += [math.sin(pos * div[i]), math.cos(pos * div[i])]
        pe.append(row)
    return pe


def shift_torch(bd, T):
    # bd[P][T] (rows relpos, cols query); S[q][k] = bd[k-q+T-1][q].
    P = 2 * T - 1
    return [[bd[k - q + T - 1][q] for k in range(T)] for q in range(T)]


def relpos_mha(x, pe, wt, C, NH):
    T = len(x)
    DK = C // NH
    q = matvec_rows(x, wt["q_w"], wt["qb"], T, C, C)
    k = matvec_rows(x, wt["k_w"], wt["kb"], T, C, C)
    v = matvec_rows(x, wt["v_w"], wt["vb"], T, C, C)
    p = matvec_rows(pe, wt["pos_w"], None, 2 * T - 1, C, C)
    sdk = math.sqrt(DK)
    merged = [[0.0] * C for _ in range(T)]
    for hd in range(NH):
        bd = [[sum((q[i][hd * DK + e] + wt["bv"][hd * DK + e]) * p[r][hd * DK + e]
                   for e in range(DK)) for i in range(T)] for r in range(2 * T - 1)]
        sh = shift_torch(bd, T)
        for i in range(T):
            sc = [(sum((q[i][hd * DK + e] + wt["bu"][hd * DK + e]) * k[j][hd * DK + e]
                       for e in range(DK)) + sh[i][j]) / sdk for j in range(T)]
            m = max(sc)
            e = [math.exp(z - m) for z in sc]
            s = sum(e)
            pr = [z / s for z in e]
            for dd in range(DK):
                merged[i][hd * DK + dd] = sum(pr[j] * v[j][hd * DK + dd] for j in range(T))
    return matvec_rows(merged, wt["o_w"], wt["ob"], T, C, C)


def conformer_ff(x, w1, b1, w2, b2):
    mid = matvec_rows(x, w1, b1, len(x), len(x[0]), len(w1))
    act = [[z / (1.0 + math.exp(-z)) for z in row] for row in mid]
    return matvec_rows(act, w2, b2, len(x), len(w1), len(x[0]))


def conformer_conv(x, wt, D, K=9):
    T = len(x)
    e = matvec_rows(x, wt["pw1_w"], wt["pw1_b"], T, D, 2 * D)
    g = [[a / (1.0 + math.exp(-b)) for a, b in zip(row[:D], row[D:])] for row in e]
    pad = (K - 1) // 2
    dw = []
    for r in range(T):
        row = []
        for c in range(D):
            acc = wt["dw_b"][c] if wt["dw_b"] else 0.0
            for kk in range(K):
                src = r - pad + kk
                acc += ((0.0 if src < 0 or src >= T else g[src][c]) * wt["dw_w"][c][kk])
            row.append(acc)
        dw.append(row)
    n = [[(dw[r][c] - wt["mean"][c]) / math.sqrt(wt["var"][c] + 1e-5) * wt["bn_w"][c] + wt["bn_b"][c]
          for c in range(D)] for r in range(T)]
    a = [[z / (1.0 + math.exp(-z)) for z in row] for row in n]
    return matvec_rows(a, wt["pw2_w"], wt["pw2_b"], T, D, D)


def conformer_layer(x, pe, wt, D, F, NH, K=9):
    res = [row[:] for row in x]
    f1 = conformer_ff(layernorm(res, wt["g1"], wt["b1"]), wt["w1"], wt["bb1"], wt["w2"], wt["bb2"])
    res = [[r + 0.5 * f for r, f in zip(rr, ff)] for rr, ff in zip(res, f1)]
    at = relpos_mha(layernorm(res, wt["ga"], wt["ba"]), pe, wt["attn"], D, NH)
    res = [[r + a for r, a in zip(rr, aa)] for rr, aa in zip(res, at)]
    cv = conformer_conv(layernorm(res, wt["gc"], wt["bc"]), wt["conv"], D, K)
    res = [[r + c for r, c in zip(rr, cc)] for rr, cc in zip(res, cv)]
    f2 = conformer_ff(layernorm(res, wt["g2"], wt["b2"]), wt["w3"], wt["bb3"], wt["w4"], wt["bb4"])
    res = [[r + 0.5 * f for r, f in zip(rr, ff)] for rr, ff in zip(res, f2)]
    return layernorm(res, wt["go"], wt["bo"])


def metrics(ref, got):
    import numpy as _np
    r = _np.asarray(ref, dtype=_np.float64)
    g = _np.asarray(got, dtype=_np.float64)
    ra = (r > 0.5).astype(int)
    ga = (g > 0.5).astype(int)
    return {"max_abs": float(_np.abs(r - g).max()), "mean_abs": float(_np.abs(r - g).mean()),
            "cosine": float((r * g).sum() / (math.sqrt((r * r).sum() * (g * g).sum()) + 1e-12)),
            "frame_agreement": float((ra == ga).mean()),
            "ref_rms": float(_np.sqrt((r * r).mean())),
            "got_rms": float(_np.sqrt((g * g).mean()))}


def load_sd(ckpt_path):
    import tarfile, io
    import torch
    with tarfile.open(ckpt_path, "r") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".ckpt"):
                f = tf.extractfile(m)
                obj = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
                return obj.get("state_dict", obj)
    raise RuntimeError("no .ckpt in .nemo")


def t2(w):
    import numpy as _np
    a = _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w, dtype=_np.float64)
    return a.T.tolist() if a.ndim == 2 else a.tolist()


def v1(w):
    import numpy as _np
    return _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w,
                       dtype=_np.float64).ravel().tolist()


def main() -> int:
    import numpy as _np
    import json
    REPORT["environment"] = h.environment_record()
    REPORT["gpu_before"] = h.gpu_snapshot()
    ref_hits = sorted(Path("/kaggle/input").rglob("m2_ref_short.npz")) or \
        sorted(Path("/kaggle/working").rglob("m2_ref_short.npz"))
    ckpts = sorted(Path("/kaggle/input").rglob("*.nemo")) or \
        sorted(Path("/kaggle/working").rglob("*.nemo"))
    if not ref_hits or not ckpts:
        REPORT["verdict"] = "ref-or-ckpt-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage2_k2_verdict.json")
        return 0
    ref_dir = ref_hits[0].parent
    REPORT["ref_dir"] = str(ref_dir)
    REPORT["ckpt"] = str(ckpts[0])
    sd = load_sd(str(ckpts[0]))

    layers = []
    for li in range(17):
        p = f"encoder.layers.{li}."
        # NeMo ConformerLayer submodule names (conformer_modules.py):
        #   norm_feed_forward1/feed_forward1(linear1/linear2) +
        #   norm_self_att/self_attn(linear_q/k/v/pos/out, pos_bias_u/v shared) +
        #   norm_conv/conv(pointwise_conv1/depthwise_conv/batch_norm/pointwise_conv2) +
        #   norm_feed_forward2/feed_forward2 + norm_out.
        # Full key list per layer (resolved against ckpt at runtime):
        #   {p}norm_feed_forward1.weight/bias, {p}feed_forward1.linear1/2.weight/bias,
        #   {p}norm_self_att.*, {p}self_attn.linear_q/k/v/pos/out.*,
        #   {p}norm_conv.*, {p}conv.pointwise_conv1/depthwise_conv/batch_norm/pointwise_conv2.*,
        #   {p}norm_feed_forward2.*, {p}feed_forward2.*, {p}norm_out.*.
        attn = {"q_w": t2(sd[p + "self_attn.linear_q.weight"]),
                "qb": v1(sd[p + "self_attn.linear_q.bias"]),
                "k_w": t2(sd[p + "self_attn.linear_k.weight"]),
                "kb": v1(sd[p + "self_attn.linear_k.bias"]),
                "v_w": t2(sd[p + "self_attn.linear_v.weight"]),
                "vb": v1(sd[p + "self_attn.linear_v.bias"]),
                "pos_w": t2(sd[p + "self_attn.linear_pos.weight"]),
                "bu": None,  # shared encoder pos_bias_u (resolved below)
                "bv": None,  # shared encoder pos_bias_v (resolved below)
                "o_w": t2(sd[p + "self_attn.linear_out.weight"]),
                "ob": v1(sd[p + "self_attn.linear_out.bias"])}
        layers.append((p, attn))
    # NOTE: assembled fully below (pos_bias shared tensors + conv names
    # resolved against the actual checkpoint key list, printed in REPORT).
    REPORT["ckpt_keys_sample"] = sorted([k for k in sd if "encoder.layers.0." in k])[:40]
    REPORT["verdict"] = "k2-scaffold"
    REPORT["note"] = ("key schema probed; full 17-layer gate assembly runs "
                      "once key names are confirmed from this probe.")
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage2_k2_verdict.json")
    print(json.dumps(REPORT, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
