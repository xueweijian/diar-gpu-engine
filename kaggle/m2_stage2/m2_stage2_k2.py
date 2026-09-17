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
import os
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


def find_ckpt() -> Path | None:
    # Runner shares the resolved/downloaded ckpt via env (see
    # m2_stage2_run._ensure_ckpt); fall back to dataset/working scan when
    # a gate runs standalone (local mechanics never reach here).
    env = os.environ.get("M2_STAGE2_CKPT")
    if env and Path(env).exists():
        return Path(env)
    hits = sorted(Path("/kaggle/input").rglob("*.nemo"))
    if not hits:
        hits = sorted(Path("/kaggle/working").rglob("*.nemo"))
    return hits[0] if hits else None
REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage2_k2_conformer",
}

D_MODEL, D_FF, N_HEADS, KERNEL = 512, 2048, 8, 9
# pos_bias_u/v are SHARED across all conformer layers (single nn.Parameter on
# the encoder, passed by reference into each ConformerLayer — see NeMo
# ConformerEncoder.__init__ `if not untie_biases` branch; diar keeps the
# default untie_biases=True... which takes the SHARED branch: the `if not`
# is True, one (n_heads, d_head) pair is created and handed to every layer).
# NeMo key names (state dict of the .nemo): the shared pair lives at
# encoder.pos_bias_u / encoder.pos_bias_v (encoder-level, NO layer index);
# per-layer projections are encoder.layers.{li}.self_attn.linear_{q,k,v,pos,out}.*.
POS_BIAS_U_KEY = "encoder.pos_bias_u"
POS_BIAS_V_KEY = "encoder.pos_bias_v"


def layer_prefix(li):
    return f"encoder.layers.{li}."


def matvec_rows(x, w, b, t, inn, out):
    # Legacy helper retained for the mechanics transcription tests only;
    # the kernel path uses lin_np/layernorm_np (numpy). See K1 header note.
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
    # Legacy helper retained for the mechanics tests only (test files
    # reference nothing else by this name; harmless). Kernel path uses
    # inline numpy softmax. Do NOT delete without updating tests.
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
    # Numpy-vectorized (same perf rewrite as K1): identical math to the C++
    # (contiguous heads, query-major shift S[q,k]=bd[k-q+T-1][q], plain
    # sqrt(Dk) scale, merge, linear_out). Nested-list I/O for the mechanics
    # tests; the kernel gate converts once outside the layer loop.
    import numpy as _np
    xa = _np.asarray(x, dtype=_np.float64)
    pa = _np.asarray(pe, dtype=_np.float64)
    T = xa.shape[0]
    DK = C // NH

    def lin(a, w, b):
        y = a @ _np.asarray(w, dtype=_np.float64).T
        return y if b is None else y + _np.asarray(b, dtype=_np.float64)

    q, k, v = lin(xa, wt["q_w"], wt["qb"]), lin(xa, wt["k_w"], wt["kb"]), lin(
        xa, wt["v_w"], wt["vb"])
    p = lin(pa, wt["pos_w"], None)
    # NeMo pos_bias_{u,v} are (H, Dk); the kernel stores them via v1()
    # (raveled [C], head-contiguous) and slices flat per head below — same
    # convention as the C++ (reshape(d_k,1,n_head,1) over the same bytes).
    # A transposed future checkpoint fails LOUDLY on the size assert.
    bu = _np.asarray(wt["bu"], dtype=_np.float64).ravel()
    bv = _np.asarray(wt["bv"], dtype=_np.float64).ravel()
    assert bu.size == C and bv.size == C, (bu.shape, bv.shape, C)
    sdk = math.sqrt(DK)
    heads = []
    for hd in range(NH):
        s = slice(hd * DK, (hd + 1) * DK)
        # bd[q, r] = (q[q]+bv) . p[r] over head dims; S[q,k] = bd[q, k-q+T-1].
        # bd here is [T, P] query-major — the TRANSPOSE of the old nested-list
        # bd[P][T] that shift_torch consumes. The identity-trace proof still
        # holds: S[q,k] = (q[q]+bv).p[k-q+T-1] either way; only the storage
        # orientation changed with the numpy rewrite.
        bd = (q[:, s] + bv[s]) @ p[:, s].T  # [T, P]
        P = 2 * T - 1
        sh = _np.stack([bd[_np.arange(T), k - _np.arange(T) + T - 1] for k in range(T)],
                       axis=1)  # S[q,k] = bd[q, k-q+T-1]
        ac = (q[:, s] + bu[s]) @ k[:, s].T  # [T, T]
        sc = (ac + sh) / sdk
        sc = sc - sc.max(axis=1, keepdims=True)
        e = _np.exp(sc)
        pr = e / e.sum(axis=1, keepdims=True)
        heads.append(pr @ v[:, s])
    merged = _np.concatenate(heads, axis=1)
    return lin(merged, wt["o_w"], wt["ob"]).tolist()


def conformer_ff(x, w1, b1, w2, b2):
    import numpy as _np
    xa = _np.asarray(x, dtype=_np.float64)
    mid = xa @ _np.asarray(w1, dtype=_np.float64).T + _np.asarray(b1, dtype=_np.float64)
    act = mid / (1.0 + _np.exp(-mid))  # SiLU == Swish
    return (act @ _np.asarray(w2, dtype=_np.float64).T + _np.asarray(
        b2, dtype=_np.float64)).tolist()


def layernorm_np(xa, g, b, eps=1e-5):
    import numpy as _np
    m = xa.mean(axis=1, keepdims=True)
    v = ((xa - m) ** 2).mean(axis=1, keepdims=True)
    y = (xa - m) / _np.sqrt(v + eps)
    if g is not None:
        y = y * _np.asarray(g, dtype=_np.float64)
    if b is not None:
        y = y + _np.asarray(b, dtype=_np.float64)
    return y


def lin_np(xa, w, b):
    import numpy as _np
    y = xa @ _np.asarray(w, dtype=_np.float64).T
    return y if b is None else y + _np.asarray(b, dtype=_np.float64)


def conformer_conv(x, wt, D, K=9):
    import numpy as _np
    xa = _np.asarray(x, dtype=_np.float64)
    T = xa.shape[0]
    e = lin_np(xa, wt["pw1_w"], wt["pw1_b"])  # [T, 2D]
    a, b = e[:, :D], e[:, D:]
    g = a / (1.0 + _np.exp(-b))  # GLU, a = FIRST half (see K1 header note)
    # Depthwise k, symmetric pad == offline CausalConv1D full context.
    pad = (K - 1) // 2
    gp = _np.pad(g, ((pad, pad), (0, 0)))
    dww = _np.asarray(wt["dw_w"], dtype=_np.float64)  # [D, K]
    # gp[idx]: [T, K, D] window stack; einsum over taps -> [T, D].
    idx = _np.arange(T)[:, None] + _np.arange(K)[None, :]
    dw = _np.einsum("tkd,dk->td", gp[idx], dww) \
        + _np.asarray(wt["dw_b"], dtype=_np.float64)[None, :]
    mean = _np.asarray(wt["mean"], dtype=_np.float64)[None, :]
    var = _np.asarray(wt["var"], dtype=_np.float64)[None, :]
    n = ((dw - mean) / _np.sqrt(var + 1e-5) * _np.asarray(wt["bn_w"], dtype=_np.float64)[None, :]
         + _np.asarray(wt["bn_b"], dtype=_np.float64)[None, :])
    act = n / (1.0 + _np.exp(-n))  # Swish == SiLU
    return lin_np(act, wt["pw2_w"], wt["pw2_b"]).tolist()


def conformer_layer(x, pe, wt, D, F, NH, K=9):
    import numpy as _np
    res = _np.asarray(x, dtype=_np.float64)
    pea = _np.asarray(pe, dtype=_np.float64)
    f1 = _np.asarray(conformer_ff(
        layernorm_np(res, wt["g1"], wt["b1"]).tolist(),
        wt["w1"], wt["bb1"], wt["w2"], wt["bb2"]), dtype=_np.float64)
    res = res + 0.5 * f1
    at = _np.asarray(relpos_mha(
        layernorm_np(res, wt["ga"], wt["ba"]).tolist(), pea.tolist(),
        wt["attn"], D, NH), dtype=_np.float64)
    res = res + at
    cv = _np.asarray(conformer_conv(
        layernorm_np(res, wt["gc"], wt["bc"]).tolist(), wt["conv"], D, K),
        dtype=_np.float64)
    res = res + cv
    f2 = _np.asarray(conformer_ff(
        layernorm_np(res, wt["g2"], wt["b2"]).tolist(),
        wt["w3"], wt["bb3"], wt["w4"], wt["bb4"]), dtype=_np.float64)
    res = res + 0.5 * f2
    return layernorm_np(res, wt["go"], wt["bo"]).tolist()


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
    # torch Linear weight [out,in] -> nested lists, SAME layout (NO transpose).
    # Python mirrors + C++ nn::linear_forward both index w[o][i]; a transpose
    # here is a silent corruption (square) or shape crash (rectangular) —
    # pinned by test_t2_keeps_torch_layout below.
    import numpy as _np
    a = _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w, dtype=_np.float64)
    return a.tolist()


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
    env_ckpt = os.environ.get("M2_STAGE2_CKPT")
    if env_ckpt and Path(env_ckpt).exists():
        ckpts = [Path(env_ckpt)]
    if not ref_hits or not ckpts:
        REPORT["verdict"] = "ref-or-ckpt-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage2_k2_verdict.json")
        return 0
    ref_dir = ref_hits[0].parent
    REPORT["ref_dir"] = str(ref_dir)
    REPORT["ckpt"] = str(ckpts[0])
    sd = load_sd(str(ckpts[0]))

    def resolve_bias(key):
        # Shared encoder-level pair; fall back to per-layer copies if a
        # future checkpoint unties them (same (H,Dk) shape either way).
        if key in sd:
            return v1(sd[key])
        per_layer = [v1(sd[layer_prefix(li) + "self_attn." + key.split(".")[-1]])
                     for li in range(17)
                     if layer_prefix(li) + "self_attn." + key.split(".")[-1] in sd]
        if per_layer:
            REPORT.setdefault("pos_bias_untied", True)
            return per_layer[0]
        raise KeyError(key)

    bu_shared = resolve_bias(POS_BIAS_U_KEY)
    bv_shared = resolve_bias(POS_BIAS_V_KEY)
    REPORT["pos_bias_source"] = ("shared:" + POS_BIAS_U_KEY if POS_BIAS_U_KEY in sd
                                 else "per-layer-fallback")

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
                "bu": bu_shared,  # shared encoder pair (see POS_BIAS_*_KEY)
                "bv": bv_shared,  # (same values for every layer)
                "o_w": t2(sd[p + "self_attn.linear_out.weight"]),
                "ob": v1(sd[p + "self_attn.linear_out.bias"])}
        # FF norms + bodies (ConformerFeedForward.linear1/linear2).
        ff1 = {"w1": t2(sd[p + "feed_forward1.linear1.weight"]),
               "bb1": v1(sd[p + "feed_forward1.linear1.bias"]),
               "w2": t2(sd[p + "feed_forward1.linear2.weight"]),
               "bb2": v1(sd[p + "feed_forward1.linear2.bias"])}
        ff2 = {"w1": t2(sd[p + "feed_forward2.linear1.weight"]),
               "bb1": v1(sd[p + "feed_forward2.linear1.bias"]),
               "w2": t2(sd[p + "feed_forward2.linear2.weight"]),
               "bb2": v1(sd[p + "feed_forward2.linear2.bias"])}
        norms = {"g1": v1(sd[p + "norm_feed_forward1.weight"]),
                 "b1": v1(sd[p + "norm_feed_forward1.bias"]),
                 "ga": v1(sd[p + "norm_self_att.weight"]),
                 "ba": v1(sd[p + "norm_self_att.bias"]),
                 "gc": v1(sd[p + "norm_conv.weight"]),
                 "bc": v1(sd[p + "norm_conv.bias"]),
                 "g2": v1(sd[p + "norm_feed_forward2.weight"]),
                 "b2": v1(sd[p + "norm_feed_forward2.bias"]),
                 "go": v1(sd[p + "norm_out.weight"]),
                 "bo": v1(sd[p + "norm_out.bias"])}
        # Conv module (ConformerConvolution, batch_norm flavour). Torch Conv1d
        # weights are 3D: pointwise [out,in,1] -> squeeze trailing dim to
        # [out,in]; depthwise [D,1,K] -> squeeze dim-1 to [D,K] (groups=D).
        # Depthwise dim fork (see conformer_conv docstring): [D,K] expected;
        # a [2D,K] checkpoint would mean the 'glu_' registry branch — report
        # LOUDLY via verdict below, never silently reshape.
        _dw = _np.asarray(sd[p + "conv.depthwise_conv.weight"])
        REPORT.setdefault("dw_shape_" + str(li), list(_dw.shape))
        _pw1 = t2(sd[p + "conv.pointwise_conv1.weight"])
        _pw2 = t2(sd[p + "conv.pointwise_conv2.weight"])
        _dwl = t2(sd[p + "conv.depthwise_conv.weight"])
        pw1_w = [[v[0] for v in row] for row in _pw1]  # [2D,D,1] -> [2D,D]
        pw2_w = [[v[0] for v in row] for row in _pw2]  # [D,D,1] -> [D,D]
        dw_w = [ch[0] for ch in _dwl]  # [D,1,K] -> [D,K]
        conv = {"pw1_w": pw1_w,
                "pw1_b": v1(sd[p + "conv.pointwise_conv1.bias"]),
                "dw_w": dw_w,
                "dw_b": v1(sd[p + "conv.depthwise_conv.bias"]),
                "bn_w": v1(sd[p + "conv.batch_norm.weight"]),
                "bn_b": v1(sd[p + "conv.batch_norm.bias"]),
                "mean": v1(sd[p + "conv.batch_norm.running_mean"]),
                "var": v1(sd[p + "conv.batch_norm.running_var"]),
                "pw2_w": pw2_w,
                "pw2_b": v1(sd[p + "conv.pointwise_conv2.bias"])}
        layers.append({"attn": attn, "ff1": ff1, "ff2": ff2, "norms": norms, "conv": conv})
    # Concat->xscale->pos-table input for layer 00 comes from the reference
    # dump itself (no new dump needed): chunk000 has empty streaming state,
    # so layer-00 input = xscale(pre_encode); later layers teacher-force on
    # reference conformer_block_{NN-1}. The pos_emb table is the analytic
    # NeMo RelPositionalEncoding window (relpos_table, pinned by test_posenc).
    xs = math.sqrt(D_MODEL)
    per_audio = {}
    for label in ("short", "mid"):
        zpath = ref_dir / f"m2_ref_{label}.npz"
        if not zpath.exists():
            continue
        z = _np.load(str(zpath), allow_pickle=True)
        deep = [k.split("/")[0] for k in z.files if k.endswith("conformer_block_00")]
        gates = []
        for chunk in deep:
            L = int(z[chunk + "/pre_encode"].shape[0])  # == layer-00 T for chunk000
            # Layer-00 full-length input: concat([], [], pre_encode) xscaled.
            # For chunks >000 the reference fc path already includes fifo/
            # spkcache accumulation, so teacher-force through the stack on
            # the FULL fc length: layer li eats reference block li-1, and
            # layer 00 eats the reference fc input? No — fc ENCODER output
            # is the stack OUTPUT. Reconstruct layer-00 input as
            # xscale(concat(prev_fifo, prev_pre, pre)) from *_after of the
            # previous chunk (state_lens_before gives the split points).
            prev_idx = int(chunk.replace("chunk", "")) - 1
            if prev_idx < 0:
                l00_in = [[v * xs for v in row]
                          for row in z[chunk + "/pre_encode"].tolist()]
                T = L
            else:
                prev = f"chunk{prev_idx:03d}/"
                n_spk, n_fifo, n_pre = (int(v) for v in z[chunk + "/state_lens_before"])
                parts = []
                if n_spk:
                    parts += z[prev + "spkcache_after"].tolist()[-n_spk:]
                if n_fifo:
                    parts += z[prev + "fifo_after"].tolist()[-n_fifo:]
                parts += z[chunk + "/pre_encode"].tolist()
                assert len(parts) == n_spk + n_fifo + n_pre, (chunk, len(parts))
                l00_in = [[v * xs for v in row] for row in parts]
                T = len(l00_in)
            pe = relpos_table(T, D_MODEL)
            prev_out = l00_in
            for li in range(17):
                wt = {"g1": layers[li]["norms"]["g1"], "b1": layers[li]["norms"]["b1"],
                      "w1": layers[li]["ff1"]["w1"], "bb1": layers[li]["ff1"]["bb1"],
                      "w2": layers[li]["ff1"]["w2"], "bb2": layers[li]["ff1"]["bb2"],
                      "ga": layers[li]["norms"]["ga"], "ba": layers[li]["norms"]["ba"],
                      "attn": layers[li]["attn"],
                      "gc": layers[li]["norms"]["gc"], "bc": layers[li]["norms"]["bc"],
                      "conv": layers[li]["conv"],
                      "g2": layers[li]["norms"]["g2"], "b2": layers[li]["norms"]["b2"],
                      "w3": layers[li]["ff2"]["w1"], "bb3": layers[li]["ff2"]["bb1"],
                      "w4": layers[li]["ff2"]["w2"], "bb4": layers[li]["ff2"]["bb2"],
                      "go": layers[li]["norms"]["go"], "bo": layers[li]["norms"]["bo"]}
                got = conformer_layer(prev_out, pe, wt, D_MODEL, D_FF, N_HEADS, KERNEL)
                ref = z[chunk + f"/conformer_block_{li:02d}"].tolist()
                m = metrics(ref, got)
                m.update({"chunk": chunk, "layer": li, "T": T})
                gates.append(m)
                prev_out = ref  # teacher forcing through the stack
        per_audio[label] = gates
    REPORT["gates"] = per_audio
    worst = max((m["max_abs"] for gates in per_audio.values() for m in gates),
                default=float("nan"))
    REPORT["worst_layer_max_abs"] = worst
    REPORT["verdict"] = "k2-measured"
    REPORT["note"] = ("17-layer teacher-forced fp32-vs-fp32 spreads; "
                      "pin gate thresholds from these numbers per plan §1.")
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage2_k2_verdict.json")
    print(json.dumps({k: v for k, v in REPORT.items() if k != "gates"}, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
