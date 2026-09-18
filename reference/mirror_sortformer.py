"""Numpy mirror of src/sortformer.cpp (M2 Stage 3.2b assembly).

Independent re-transcription of the C++ contracts documented in
include/diar/{subsampling,posenc,mha,conv,conformer,layers,nn}.hpp — the same
math the K1/K2/K3 Kaggle kernels verified against live NeMo fp32 (K1 gate
1.2e-6, K2 1.2e-5, K3 8e-5). Computed here in float64; the C++ engine is
fp32, so taps are compared at a 1e-4 absolute gate (tests/test_stage32_mirror.py).

Input is the dump written by tools/dump_forward.cpp: manifest.txt (flat
line-based) + raw .f32 payload files, all row-major outermost-first.
"""
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- manifest
def read_dump(dump_dir):
    d = Path(dump_dir)
    manifest = {"config": {}, "input": {}, "tensor": {}, "tap": {}}
    for line in (d / "manifest.txt").read_text().splitlines():
        if line == "END" or not line.strip():
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "config":
            manifest["config"][parts[1]] = parts[2]
        elif kind in ("input", "tensor", "tap"):
            name = parts[1]
            dims = [int(x) for x in parts[2:-1]]
            fname = parts[-1]
            arr = np.fromfile(d / fname, dtype="<f4").astype(np.float64)
            if arr.size != int(np.prod(dims)):
                raise ValueError(f"{fname}: {arr.size} floats, manifest says {dims}")
            manifest[kind][name] = arr.reshape(dims)
    return manifest


# ---------------------------------------------------------------- nn ops
def linear(x, w, b=None):
    y = x @ np.asarray(w).T
    return y + b if b is not None else y


def layernorm(x, g, b, eps=1e-5):
    # biased variance (divide by n), eps INSIDE sqrt (ggml_norm semantics)
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def batchnorm_infer(x, w, b, mean, var, eps=1e-5):
    # per-channel affine; x rows are frames, columns channels
    return (x - mean) / np.sqrt(var + eps) * w + b


def softmax_rows(x):
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def relu(x):
    return np.maximum(x, 0.0)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def glu(x):
    # part_a * sigmoid(part_b), a = FIRST half (src/nn.cpp glu_forward)
    half = x.shape[-1] // 2
    return x[..., :half] * sigmoid(x[..., half:])


# ---------------------------------------------------------------- stem
def conv_out_len(n, k=3, s=2, p=1):
    return (n + 2 * p - k) // s + 1


def _pad(x):
    # symmetric zero pad 1 on H,W of [C,H,W]
    return np.pad(x, ((0, 0), (1, 1), (1, 1)))


def conv2d_relu(x, w, b):
    # x [C,H,W], w [O,I,3,3], stride 2 pad 1, relu (src conv2d_relu)
    xp = _pad(x)
    c_out, h_out, w_out = w.shape[0], conv_out_len(x.shape[1]), conv_out_len(x.shape[2])
    y = np.empty((c_out, h_out, w_out))
    for ho in range(h_out):
        for wo in range(w_out):
            patch = xp[:, ho * 2 : ho * 2 + 3, wo * 2 : wo * 2 + 3]  # [C,I,3,3]
            y[:, ho, wo] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
    return np.maximum(y, 0.0)


def dw_conv2d(x, w, b):
    # grouped depthwise: w [C,3,3] (GGUF keeps [C,1,3,3] — same bytes), stride 2 pad 1
    c = w.shape[0]
    w = w.reshape(c, 3, 3)
    xp = _pad(x)
    c, h_out, w_out = x.shape[0], conv_out_len(x.shape[1]), conv_out_len(x.shape[2])
    y = np.empty((c, h_out, w_out))
    for ho in range(h_out):
        for wo in range(w_out):
            patch = xp[:, ho * 2 : ho * 2 + 3, wo * 2 : wo * 2 + 3]  # [C,3,3]
            y[:, ho, wo] = (patch * w).sum(axis=(1, 2)) + b
    return y


def pointwise_relu(d, w, b):
    # d [C,T2,F2], w [C,C,1,1] -> per-position Linear over channels + relu
    # (src pointwise_relu: x[i*hw+q], i channel, q = t*F2+f position)
    c, t2, f2 = d.shape
    x_hw = d.transpose(1, 2, 0).reshape(t2 * f2, c)  # [HW, C]
    y = relu(linear(x_hw, w.reshape(c, c), b))  # [HW, C]
    return y.reshape(t2, f2, c).transpose(2, 0, 1)  # [C,T2,F2]


def zero_time_rows(v, l):
    # v [C,T,F]: time rows >= l zeroed (v13 MaskedConv per-stage mask)
    v[:, l:, :] = 0.0


def subsampling(mel, wt, feat_len):
    # mel [T,F] frame-major == C++ x [1,T,F] channel-first view.
    if feat_len <= 0 or feat_len > mel.shape[0]:
        feat_len = mel.shape[0]
    l1 = conv_out_len(feat_len)
    l2 = conv_out_len(l1)
    l3 = conv_out_len(l2)
    mel = mel.copy()
    mel[feat_len:, :] = 0.0
    p = "encoder.pre_encode."
    s0 = conv2d_relu(mel[None, :, :], wt[p + "conv.0.weight"], wt[p + "conv.0.bias"])
    zero_time_rows(s0, l1)
    s1 = pointwise_relu(dw_conv2d(s0, wt[p + "conv.2.weight"], wt[p + "conv.2.bias"]),
        wt[p + "conv.3.weight"], wt[p + "conv.3.bias"])
    zero_time_rows(s1, l2)
    s2 = pointwise_relu(dw_conv2d(s1, wt[p + "conv.5.weight"], wt[p + "conv.5.bias"]),
        wt[p + "conv.6.weight"], wt[p + "conv.6.bias"])
    zero_time_rows(s2, l3)
    c_ch, t3, f3 = s2.shape
    flat = s2.transpose(1, 0, 2).reshape(t3, c_ch * f3)  # frame t: s2[:,t,:] concat
    return linear(flat, wt[p + "out.weight"], wt[p + "out.bias"])


# ---------------------------------------------------------------- pos table
def relpos_table(l, d_model):
    # src/posenc.cpp: row r <-> pos = (l-1) - r; div_term exp(2i * -ln(1e4)/d)
    r = np.arange(2 * l - 1, dtype=np.float64)
    pos = (l - 1) - r
    i = np.arange(0, d_model, 2, dtype=np.float64)
    div = np.exp(i * -(LOG10000 / d_model))
    ang = pos[:, None] * div[None, :]
    pe = np.empty((2 * l - 1, d_model))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


# ---------------------------------------------------------------- mha
def relpos_mha(x, pe, wt, p, n_heads):
    # src/mha.cpp relpos_mha_forward
    t, c = x.shape
    dk = c // n_heads
    q = linear(x, wt[p + "linear_q.weight"], wt[p + "linear_q.bias"])
    k = linear(x, wt[p + "linear_k.weight"], wt[p + "linear_k.bias"])
    v = linear(x, wt[p + "linear_v.weight"], wt[p + "linear_v.bias"])
    pr = linear(pe, wt[p + "linear_pos.weight"])
    bu = wt[p + "pos_bias_u"].reshape(n_heads, dk)
    bv = wt[p + "pos_bias_v"].reshape(n_heads, dk)
    y = np.empty((t, c))
    s_dk = np.sqrt(float(dk))
    for h in range(n_heads):
        sl = slice(h * dk, (h + 1) * dk)
        qh, kh, vh = q[:, sl] + bu[h], k[:, sl], v[:, sl]
        ph = pr[:, sl]
        scores = qh @ kh.T  # content term [T,T]
        bd = (qh @ ph.T)  # [T_q, P rows] — C++ bd[r,i] = (q_i+bv)·p_r
        bd = bd.T  # [P, T_q]
        # torch S[i,j] += BD[j - i + T - 1, i] (C++ shifted[j*T+i] with
        # rel_shift S_ggml[k,j] = BD[k-j+T-1, j])
        rel = np.empty((t, t))
        for i in range(t):
            for j in range(t):
                rel[i, j] = bd[j - i + t - 1, i]
        probs = softmax_rows((scores + rel) / s_dk)
        y[:, sl] = probs @ vh
    return linear(y, wt[p + "linear_out.weight"], wt[p + "linear_out.bias"])


def plain_mha(x, wt, p, n_heads):
    # src/layers.cpp transformer_block_forward MHA (scale 1/sqrt(DK), biased)
    t, h = x.shape
    dk = h // n_heads
    q = linear(x, wt[p + "query_net.weight"], wt[p + "query_net.bias"])
    k = linear(x, wt[p + "key_net.weight"], wt[p + "key_net.bias"])
    v = linear(x, wt[p + "value_net.weight"], wt[p + "value_net.bias"])
    y = np.empty((t, h))
    scale = 1.0 / np.sqrt(float(dk))
    for hd in range(n_heads):
        sl = slice(hd * dk, (hd + 1) * dk)
        probs = softmax_rows((q[:, sl] @ k[:, sl].T) * scale)
        y[:, sl] = probs @ v[:, sl]
    return linear(y, wt[p + "out_projection.weight"], wt[p + "out_projection.bias"])


# ---------------------------------------------------------------- modules
def conformer_ff(x, wt, p):
    # linear1 -> SiLU -> linear2 (src conformer_ff_forward)
    return linear(silu(linear(x, wt[p + "linear1.weight"], wt[p + "linear1.bias"])),
        wt[p + "linear2.weight"], wt[p + "linear2.bias"])


def conformer_conv(x, wt, p, kernel, eps=1e-5):
    # pw1 -> GLU -> depthwise(sym pad) -> BN -> SiLU -> pw2 (src conv.cpp)
    t, d = x.shape
    e = linear(x, wt[p + "pointwise_conv1.weight"], wt[p + "pointwise_conv1.bias"])
    g = glu(e)
    pad = (kernel - 1) // 2
    gp = np.pad(g, ((0, 0), (pad, pad)))  # (kept for clarity; indexing is explicit below)
    dw = wt[p + "depthwise_conv.weight"].reshape(d, kernel)  # GGUF keeps [D,1,K]
    db = wt[p + "depthwise_conv.bias"]
    y = np.empty((t, d))
    for tt in range(t):
        # y[t,o] = sum_k g[t-pad+k-? ...] — C++: ti = t - pad + k, skip OOB
        acc = db.copy()
        for kk in range(kernel):
            ti = tt - pad + kk
            if 0 <= ti < t:
                acc += g[ti] * dw[:, kk]
        y[tt] = acc
    n = batchnorm_infer(y, wt[p + "batch_norm.weight"], wt[p + "batch_norm.bias"],
        wt[p + "batch_norm.running_mean"], wt[p + "batch_norm.running_var"], eps)
    return linear(silu(n), wt[p + "pointwise_conv2.weight"], wt[p + "pointwise_conv2.bias"])


def conformer_layer(x, pe, wt, p, d_ff, n_heads, kernel):
    # src/conformer.cpp: macaron FF1 -> rel-pos attn -> conv -> FF2 -> LN
    res = x.copy()
    res = res + 0.5 * conformer_ff(layernorm(res, wt[p + "norm_feed_forward1.weight"],
        wt[p + "norm_feed_forward1.bias"]), wt, p + "feed_forward1.")
    res = res + relpos_mha(layernorm(res, wt[p + "norm_self_att.weight"],
        wt[p + "norm_self_att.bias"]), pe, wt, p + "self_attn.", n_heads)
    res = res + conformer_conv(layernorm(res, wt[p + "norm_conv.weight"],
        wt[p + "norm_conv.bias"]), wt, p + "conv.", kernel)
    res = res + 0.5 * conformer_ff(layernorm(res, wt[p + "norm_feed_forward2.weight"],
        wt[p + "norm_feed_forward2.bias"]), wt, p + "feed_forward2.")
    return layernorm(res, wt[p + "norm_out.weight"], wt[p + "norm_out.bias"])


def transformer_block(x, wt, p, inner, n_heads):
    # src/layers.cpp transformer_block_forward (post-LN, FF activation ReLU)
    h = x + plain_mha(x, wt, p + "first_sub_layer.", n_heads)
    h_ln = layernorm(h, wt[p + "layer_norm_1.weight"], wt[p + "layer_norm_1.bias"])
    ff = relu(linear(h_ln, wt[p + "second_sub_layer.dense_in.weight"],
        wt[p + "second_sub_layer.dense_in.bias"]))
    ff_out = linear(ff, wt[p + "second_sub_layer.dense_out.weight"],
        wt[p + "second_sub_layer.dense_out.bias"])
    return layernorm(ff_out + h_ln, wt[p + "layer_norm_2.weight"], wt[p + "layer_norm_2.bias"])


def diar_head(x, wt):
    # src/layers.cpp diar_head_forward
    a = relu(x)
    h1 = relu(linear(a, wt["head.first_hidden_to_hidden.weight"],
        wt["head.first_hidden_to_hidden.bias"]))
    logits = linear(h1, wt["head.single_hidden_to_spks.weight"],
        wt["head.single_hidden_to_spks.bias"])
    return sigmoid(logits)


# ---------------------------------------------------------------- assembly
def run_chunk(mel, t_mel, feat_len, spkcache, fifo, wt, cfg):
    """Mirror of sortformer_run_chunk — returns {tap_name: array}."""
    d = int(cfg["sortformer.encoder.d_model"])
    x = bool(int(cfg.get("sortformer.encoder.xscaling", "1")))
    n_conf = int(cfg["sortformer.encoder.n_layers"])
    n_xf = int(cfg["sortformer.transformer.n_layers"])
    d_ff = int(cfg["sortformer.encoder.d_ff"])
    heads = int(cfg["sortformer.encoder.n_heads"])
    kernel = int(cfg["sortformer.encoder.conv_kernel_size"])
    inner = int(cfg["sortformer.transformer.inner_size"])
    x_heads = int(cfg["sortformer.transformer.n_heads"])
    pe_max = int(cfg["sortformer.encoder.pos_emb_max_len"])
    factor = int(cfg.get("sortformer.encoder.subsampling_factor", "8"))
    stages = 0
    for f in range(factor, 1, factor // 2 if factor > 1 else 2):
        stages += 1
    stages = factor.bit_length() - 1
    # subsampled length: per-stage ceil chain (sortformer_subsampled_len)
    t3 = t_mel
    for _ in range(stages):
        t3 = (t3 - 1) // 2 + 1

    taps = {}
    chunk = subsampling(mel, wt, feat_len)
    taps["stem.out"] = chunk
    l1s, l2s = (0 if spkcache is None else spkcache.shape[0]), (0 if fifo is None else fifo.shape[0])
    big_l = l1s + l2s + t3
    xc = np.zeros((big_l, d))
    if l1s:
        xc[:l1s] = spkcache
    if l2s:
        xc[l1s : l1s + l2s] = fifo
    xc[l1s + l2s :] = chunk
    taps["concat.raw"] = xc.copy()
    if x:
        xc = xc * np.sqrt(float(d))
    taps["xscaled"] = xc.copy()
    # stored pe sliced exactly like production; fixture pe IS the formula
    # table for L=pe_max (see test fixture), so slice == formula up to fp32.
    pe_store = wt["encoder.pos_enc.pe"]
    pe = pe_store[pe_max - big_l : pe_max + big_l - 1]
    taps["pos_emb"] = pe
    for i in range(n_conf):
        xc = conformer_layer(xc, pe, wt, f"encoder.layers.{i}.", d_ff, heads, kernel)
        taps[f"conformer.{i}"] = xc.copy()
    px = linear(xc, wt["encoder_proj.weight"], wt["encoder_proj.bias"])
    taps["proj.out"] = px
    for i in range(n_xf):
        px = transformer_block(px, wt, f"transformer.layers.{i}.", inner, x_heads)
        taps[f"transformer.{i}"] = px.copy()
    taps["preds"] = diar_head(px, wt)
    return taps


GATE = 1e-4


def compare(dump_dir, gate=GATE):
    man = read_dump(dump_dir)
    cfg = man["config"]
    wt = man["tensor"]
    mel = man["input"]["mel"]
    sc = man["input"]["spkcache"]
    fifo = man["input"]["fifo"]
    got = run_chunk(mel, mel.shape[0], int(cfg.get("__feat_len", "-1")), sc if sc.size else None,
        fifo if fifo.size else None, wt, cfg)
    # NOTE: dump tool ran with feat_len from its CLI; manifest records only
    # shapes, so the test passes the same t_mel/feat_len it invoked with.
    worst = {}
    for name, want in got.items():
        have = man["tap"][name]
        if want.shape != have.shape:
            raise ValueError(f"{name}: mirror {want.shape} vs cpp {have.shape}")
        worst[name] = float(np.abs(want - have).max())
    return worst


if __name__ == "__main__":
    diffs = compare(sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage32_dump")
    for k in sorted(diffs, key=lambda k: -diffs[k]):
        print(f"{k:20s} max_abs={diffs[k]:.3e}")
    if max(diffs.values(), default=0.0) > GATE:
        print(f"FAIL: taps above {GATE}")
        sys.exit(1)
    print(f"PASS: {len(diffs)} taps within {GATE} (mirror fp64 vs C++ fp32)")
