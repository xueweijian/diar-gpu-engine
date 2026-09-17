"""M2 Stage 2 K4: single-chunk NeMo-vs-local conformer probe (short/chunk000).

Runs INSIDE the kernel where torch+NeMo exist: rebuilds layer-00 input
exactly as the diar streaming path does (concat fifo/spkcache/pre ->
frontend_encoder with bypass_pre_encode=True -> xscale inside pos_enc),
then compares, stage by stage through layer 00 only:

  P0  xscaled input:        local (pre*xs) vs NeMo pos_enc output x
  P1  pos table:            local relpos_table vs NeMo pos_enc.pe window
  P2  norm_ff1 output
  P3  MHA output            (uses NeMo's OWN q/k/v/p tensors recombined
                             with the LOCAL shift/softmax formula -> isolates
                             formula-vs-weights; plus a full-local MHA on the
                             same input -> isolates input-vs-formula)
  P4  conv output
  P5  full layer output     (= old K2 L00/chunk000 gate, 0.45 anchor)

Every comparison is teacher-forced on NeMo's own intermediate inputs
(hooks observe, forward path untouched), so a FAIL localizes to one stage.
Hooks capture per-submodule in/out on the REAL streaming forward, not a
reconstructed call: P2-P5 inputs are NeMo's own tensors.

Verdict: k4-measured. A stage going 0.45 -> ~1e-6 while the next stays red
names the guilty submodule. If P0 is already red, the bug is in the input
chain (xscale/concat), not the layer. teacher-forced.
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

REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage2_k4_probe",
}

D_MODEL, N_HEADS = 512, 8


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


def relpos_mha(x, pe, wt, C, NH):
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
    bu = _np.asarray(wt["bu"], dtype=_np.float64).ravel()
    bv = _np.asarray(wt["bv"], dtype=_np.float64).ravel()
    sdk = math.sqrt(DK)
    heads = []
    for hd in range(NH):
        s = slice(hd * DK, (hd + 1) * DK)
        bd = (q[:, s] + bv[s]) @ p[:, s].T  # [T, P]
        P = 2 * T - 1
        sh = _np.stack([bd[_np.arange(T), k - _np.arange(T) + T - 1] for k in range(T)],
                       axis=1)
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
    act = mid / (1.0 + _np.exp(-mid))
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
    g = a / (1.0 + _np.exp(-b))  # GLU, a = FIRST half
    pad = (K - 1) // 2
    gp = _np.pad(g, ((pad, pad), (0, 0)))
    dww = _np.asarray(wt["dw_w"], dtype=_np.float64)  # [D, K]
    idx = _np.arange(T)[:, None] + _np.arange(K)[None, :]
    dw = _np.einsum("tkd,dk->td", gp[idx], dww) \
        + _np.asarray(wt["dw_b"], dtype=_np.float64)[None, :]
    mean = _np.asarray(wt["mean"], dtype=_np.float64)[None, :]
    var = _np.asarray(wt["var"], dtype=_np.float64)[None, :]
    n = ((dw - mean) / _np.sqrt(var + 1e-5) * _np.asarray(wt["bn_w"], dtype=_np.float64)[None, :]
         + _np.asarray(wt["bn_b"], dtype=_np.float64)[None, :])
    act = n / (1.0 + _np.exp(-n))
    return lin_np(act, wt["pw2_w"], wt["pw2_b"]).tolist()


def cmp(ref, got):
    import numpy as _np
    r = _np.asarray(ref, dtype=_np.float64)
    g = _np.asarray(got, dtype=_np.float64)
    return {"max_abs": float(_np.abs(r - g).max()),
            "mean_abs": float(_np.abs(r - g).mean()),
            "cosine": float((r * g).sum() / (math.sqrt((r * r).sum() * (g * g).sum()) + 1e-12)),
            "ref_rms": float(_np.sqrt((r * r).mean())),
            "got_rms": float(_np.sqrt((g * g).mean())),
            "shape": [int(v) for v in r.shape]}


def t2(w):
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
    env_ckpt = os.environ.get("M2_STAGE2_CKPT")
    ckpts = [Path(env_ckpt)] if env_ckpt and Path(env_ckpt).exists() else (
        sorted(Path("/kaggle/input").rglob("*.nemo")) or
        sorted(Path("/kaggle/working").rglob("*.nemo")))
    if not ref_hits or not ckpts:
        REPORT["verdict"] = "ref-or-ckpt-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage2_k4_verdict.json")
        return 0
    ref_dir = ref_hits[0].parent
    REPORT["ref_dir"] = str(ref_dir)
    REPORT["ckpt"] = str(ckpts[0])

    import torch
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError:
        # K1/K2/K3 need only torch; the probe needs NeMo itself. Same
        # S0-b recipe as m2_stage0_refdump.py (quiet pip, internet on).
        import subprocess as _sp
        print("[k4] nemo_toolkit missing, pip installing ...", flush=True)
        p = _sp.run(["pip", "install", "--quiet", "nemo_toolkit[asr]"],
                    capture_output=True, text=True, timeout=3600)
        print("[k4] pip rc=", p.returncode, (p.stderr or p.stdout)[-500:], flush=True)
        from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: E402
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(ckpts[0]), map_location=torch.device("cpu"))
    model.eval()
    enc = model.encoder
    REPORT["self_attention_model"] = enc.self_attention_model
    REPORT["xscale"] = float(enc.xscale) if enc.xscale else None
    REPORT["att_context_size"] = list(enc.att_context_size)
    REPORT["att_context_style"] = enc.att_context_style
    REPORT["n_layers"] = len(enc.layers)
    L0 = enc.layers[0]
    REPORT["l0_mha_class"] = type(L0.self_attn).__name__
    REPORT["l0_conv_norm"] = L0.conv.norm_type
    REPORT["l0_conv_kernel"] = int(L0.conv.kernel_size)
    REPORT["l0_pw1_shape"] = list(L0.conv.pointwise_conv1.weight.shape)
    REPORT["l0_dw_shape"] = list(L0.conv.depthwise_conv.weight.shape)
    REPORT["l0_pw1_is_glu_default"] = (L0.conv.pointwise_activation == "glu_")
    REPORT["l0_ff_act"] = type(L0.feed_forward1.activation).__name__

    # Rebuild the chunk000 streaming input on the REAL path: zeros state +
    # pre_encode window through frontend_encoder(bypass=True), hooks observe.
    z = _np.load(str(ref_dir / "m2_ref_short.npz"), allow_pickle=True)
    chunk = "chunk000"
    pre_ref = _np.asarray(z[chunk + "/pre_encode"], dtype=_np.float32)
    T_pre = int(pre_ref.shape[0])
    device = torch.device("cpu")
    sm = model.sortformer_modules
    state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=device)
    with torch.inference_mode():
        # Feed the STORED pre_encode window (== what dump's pre_encode call
        # produced) straight into the real streaming path: concat state +
        # frontend_encoder(bypass=True). No mel orientation involved.
        pre_embs = torch.from_numpy(pre_ref).unsqueeze(0).to(device)
        pre_lens = torch.tensor([T_pre], device=device)
        REPORT["probe_pre_lens"] = [int(v) for v in pre_lens.tolist()]
        REPORT["probe_pre_shape"] = list(pre_embs.shape)
        concat_embs = sm.concat_embs([state.spkcache, state.fifo, pre_embs], dim=1,
                                     device=device)
        concat_lens = (state.spkcache.shape[1] + state.fifo.shape[1] + pre_lens)

        cap: dict[str, object] = {}
        handles = []
        lay0 = enc.layers[0]

        def rec(name):
            def _fn(mod, args, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                cap[name] = o.detach().cpu().numpy()
            return _fn

        handles.append(enc.pos_enc.register_forward_hook(rec("posenc_out_x")))
        # pos_emb is the 2nd element of pos_enc's output tuple: capture via
        # a FORWARD HOOK (sees (x_out, pos_emb); a method wrapper would NOT
        # fire because forward_internal calls self.pos_enc(...) — keep both
        # belt and suspenders, but the hook is authoritative).
        def pos_hook(mod, args, out):
            xo, pe = out
            cap["pos_emb"] = pe.detach().cpu().numpy()
            cap["pos_pe_store_rows"] = int(mod.pe.size(1))

        handles.append(enc.pos_enc.register_forward_hook(pos_hook))
        handles.append(lay0.norm_feed_forward1.register_forward_hook(rec("n_ff1")))
        handles.append(lay0.feed_forward1.register_forward_hook(rec("ff1")))
        handles.append(lay0.norm_self_att.register_forward_hook(rec("n_sa")))
        # MHA internals: recombine NeMo's own q/k/v/p with local formula.
        # Hooks on linear_q/k/v/pos (nn.Linear) capture (B, T, C) — squeeze
        # batch in P3b. (NOT forward_qkv's (B,H,T,Dk): that method's output
        # is internal, hook the linears, not the method.)
        mh = lay0.self_attn
        qkv: dict[str, object] = {}

        def qkv_hook(name):
            def _fn(mod, args, out):
                qkv[name] = out.detach().cpu().numpy()
            return _fn

        handles.append(mh.linear_q.register_forward_hook(qkv_hook("q")))
        handles.append(mh.linear_k.register_forward_hook(qkv_hook("k")))
        handles.append(mh.linear_v.register_forward_hook(qkv_hook("v")))
        handles.append(mh.linear_pos.register_forward_hook(qkv_hook("p")))
        handles.append(mh.register_forward_hook(rec("mha")))
        handles.append(lay0.norm_conv.register_forward_hook(rec("n_conv")))
        handles.append(lay0.conv.register_forward_hook(rec("conv")))
        handles.append(lay0.norm_feed_forward2.register_forward_hook(rec("n_ff2")))
        handles.append(lay0.feed_forward2.register_forward_hook(rec("ff2")))
        handles.append(lay0.norm_out.register_forward_hook(rec("layer_out")))

        try:
            with torch.inference_mode():
                fc_embs, fc_lens = model.frontend_encoder(
                    processed_signal=concat_embs,
                    processed_signal_length=concat_lens,
                    bypass_pre_encode=True)
        finally:
            for hd in handles:
                hd.remove()
        REPORT["probe_fc_shape"] = list(fc_embs.shape)
        REPORT["probe_fc_lens"] = [int(v) for v in fc_lens.tolist()]

    probes: dict[str, object] = {}
    xs = math.sqrt(D_MODEL)
    pre_ref = z[chunk + "/pre_encode"].tolist()
    local_xscaled = [[v * xs for v in row] for row in pre_ref]
    nemo_x = _np.asarray(cap["posenc_out_x"])
    assert nemo_x.shape[0] == 1, nemo_x.shape  # (B, T, C), B == 1
    nemo_x = nemo_x.reshape(-1, D_MODEL)
    REPORT["probe_T"] = int(nemo_x.shape[0])
    probes["P0_xscaled_input"] = cmp(nemo_x, _np.asarray(local_xscaled)[:nemo_x.shape[0]])

    pe_nemo = _np.asarray(cap["pos_emb"])
    assert pe_nemo.shape[0] == 1, pe_nemo.shape  # (1, P, C)
    pe_nemo = pe_nemo.reshape(-1, D_MODEL)
    REPORT["probe_pe_shape"] = [int(v) for v in pe_nemo.shape]
    T = int(nemo_x.shape[0])
    pe_local = _np.asarray(relpos_table(T, D_MODEL))
    n_cmp = min(pe_nemo.shape[0], pe_local.shape[0])
    probes["P1_pos_table"] = cmp(pe_nemo[:n_cmp], pe_local[:n_cmp])
    # Row-0/center diagnostics: which local row matches NeMo row 0?
    d0 = ((pe_nemo[:1, :8] - pe_local[:, :8]) ** 2).mean(axis=1)
    REPORT["probe_pe_row0_best"] = int(_np.argmin(d0))
    REPORT["probe_pe_row0_best_mse"] = float(d0.min())
    REPORT["probe_pe_row0_mse_vs_same"] = float(d0[0])
    # Store geometry pin: exact window arithmetic needs the live store rows
    # (hook-captured above). The K2 fix mirrors THIS slice, not extend_pe(L).
    REPORT["probe_store_rows"] = int(cap.get("pos_pe_store_rows", -1))

    p = f"encoder.layers.0."
    sd = model.state_dict()
    attn_wt = {"q_w": t2(sd[p + "self_attn.linear_q.weight"]),
               "qb": v1(sd[p + "self_attn.linear_q.bias"]),
               "k_w": t2(sd[p + "self_attn.linear_k.weight"]),
               "kb": v1(sd[p + "self_attn.linear_k.bias"]),
               "v_w": t2(sd[p + "self_attn.linear_v.weight"]),
               "vb": v1(sd[p + "self_attn.linear_v.bias"]),
               "pos_w": t2(sd[p + "self_attn.linear_pos.weight"]),
               "bu": v1(sd[p + "self_attn.pos_bias_u"]),
               "bv": v1(sd[p + "self_attn.pos_bias_v"]),
               "o_w": t2(sd[p + "self_attn.linear_out.weight"]),
               "ob": v1(sd[p + "self_attn.linear_out.bias"])}
    n_sa = _np.asarray(cap["n_sa"]).reshape(-1, D_MODEL)  # (B,T,C) -> (T,C)
    mha_nemo = _np.asarray(cap["mha"]).reshape(-1, D_MODEL)
    probes["P2_norm_ff1"] = cmp(_np.asarray(cap["n_ff1"]).reshape(-1, D_MODEL),
                                layernorm_np(_np.asarray(local_xscaled)[:T],
                                             v1(sd[p + "norm_feed_forward1.weight"]),
                                             v1(sd[p + "norm_feed_forward1.bias"])))
    # P3a: LOCAL full MHA on NeMo's own normed input + local pe window
    mha_local = _np.asarray(relpos_mha(n_sa.tolist(), pe_local.tolist(), attn_wt,
                                       D_MODEL, N_HEADS))
    probes["P3a_mha_full_local"] = cmp(mha_nemo, mha_local)
    # P3b: LOCAL shift/softmax recombined on NeMo's OWN q/k/v/p.
    # Linear hooks capture (B, T, C)/(Bp, P, C) -> reshape to (T, H, Dk).
    # (B is 1 here; assert it rather than silently squeezing a real batch.)
    DK = D_MODEL // N_HEADS
    q = _np.asarray(qkv["q"]).reshape(1, T, D_MODEL).reshape(T, N_HEADS, DK)
    k = _np.asarray(qkv["k"]).reshape(1, T, D_MODEL).reshape(T, N_HEADS, DK)
    v = _np.asarray(qkv["v"]).reshape(1, T, D_MODEL).reshape(T, N_HEADS, DK)
    _pp = _np.asarray(qkv["p"])
    assert _pp.shape[0] == 1, _pp.shape
    pp = _pp.reshape(-1, N_HEADS, DK)
    P = pp.shape[0]
    bu = _np.asarray(attn_wt["bu"]).reshape(N_HEADS, DK)
    bv = _np.asarray(attn_wt["bv"]).reshape(N_HEADS, DK)
    heads = []
    for hd in range(N_HEADS):
        bd = (q[:, hd, :] + bv[hd]) @ pp[:, hd, :].T  # [T, P]
        sh = _np.stack([bd[_np.arange(T), kk - _np.arange(T) + T - 1]
                        for kk in range(T)], axis=1)
        ac = (q[:, hd, :] + bu[hd]) @ k[:, hd, :].T
        sc = (ac + sh) / math.sqrt(DK)
        sc = sc - sc.max(axis=1, keepdims=True)
        e = _np.exp(sc)
        pr = e / e.sum(axis=1, keepdims=True)
        heads.append(pr @ v[:, hd, :])
    merged = _np.concatenate(heads, axis=1)
    o_w = _np.asarray(attn_wt["o_w"]); o_b = _np.asarray(attn_wt["ob"])
    probes["P3b_mha_nemo_qkvp_local_formula"] = cmp(mha_nemo, merged @ o_w.T + o_b)
    q0 = _np.asarray(qkv["q"]).reshape(-1)
    REPORT["probe_q_span"] = [float(q0.min()), float(q0.max())]

    n_conv = _np.asarray(cap["n_conv"]).reshape(-1, D_MODEL)
    conv_nemo = _np.asarray(cap["conv"]).reshape(-1, D_MODEL)
    _dw = _np.asarray(sd[p + "conv.depthwise_conv.weight"])
    REPORT["probe_dw_shape"] = list(_dw.shape)
    _pw1 = t2(sd[p + "conv.pointwise_conv1.weight"])
    _pw2 = t2(sd[p + "conv.pointwise_conv2.weight"])
    _dwl = t2(sd[p + "conv.depthwise_conv.weight"])
    conv_wt = {"pw1_w": [[vv[0] for vv in row] for row in _pw1],
               "pw1_b": v1(sd[p + "conv.pointwise_conv1.bias"]),
               "dw_w": [ch[0] for ch in _dwl],
               "dw_b": v1(sd[p + "conv.depthwise_conv.bias"]),
               "bn_w": v1(sd[p + "conv.batch_norm.weight"]),
               "bn_b": v1(sd[p + "conv.batch_norm.bias"]),
               "mean": v1(sd[p + "conv.batch_norm.running_mean"]),
               "var": v1(sd[p + "conv.batch_norm.running_var"]),
               "pw2_w": [[vv[0] for vv in row] for row in _pw2],
               "pw2_b": v1(sd[p + "conv.pointwise_conv2.bias"])}
    probes["P4_conv_on_nemo_input"] = cmp(conv_nemo,
                                          _np.asarray(conformer_conv(
                                              n_conv.tolist(), conv_wt, D_MODEL, 9)))

    # P5: full local layer on the local input (== old K2 L00/chunk000 anchor)
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
    res = _np.asarray(local_xscaled)[:T]
    f1 = _np.asarray(conformer_ff(layernorm_np(res, norms["g1"], norms["b1"]).tolist(),
                                  ff1["w1"], ff1["bb1"], ff1["w2"], ff1["bb2"]))
    res = res + 0.5 * f1
    at = _np.asarray(relpos_mha(layernorm_np(res, norms["ga"], norms["ba"]).tolist(),
                                pe_local.tolist(), attn_wt, D_MODEL, N_HEADS))
    res = res + at
    cv = _np.asarray(conformer_conv(layernorm_np(res, norms["gc"], norms["bc"]).tolist(),
                                    conv_wt, D_MODEL, 9))
    res = res + cv
    f2 = _np.asarray(conformer_ff(layernorm_np(res, norms["g2"], norms["b2"]).tolist(),
                                  ff2["w1"], ff2["bb1"], ff2["w2"], ff2["bb2"]))
    res = res + 0.5 * f2
    layer_local = layernorm_np(res, norms["go"], norms["bo"])
    probes["P5_full_layer_local"] = cmp(_np.asarray(cap["layer_out"]).reshape(-1, D_MODEL),
                                          layer_local)
    ref_b0 = _np.asarray(z[chunk + "/conformer_block_00"].tolist())
    probes["P5_nemo_vs_dump"] = cmp(ref_b0[:T],
                                    _np.asarray(cap["layer_out"]).reshape(-1, D_MODEL))
    REPORT["probes"] = probes
    REPORT["verdict"] = "k4-measured"
    REPORT["note"] = ("NeMo-vs-local stage probe on short/chunk000 layer 00; "
                      "the stage where max_abs collapses to ~1e-6 while the "
                      "next stays red names the guilty submodule.")
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage2_k4_verdict.json")
    print(json.dumps({k: v for k, v in REPORT.items() if k != "probes"}, indent=1)[:3000])
    for name, m in probes.items():
        print(f"{name}: max_abs={m['max_abs']:.3e} mean={m['mean_abs']:.3e} "
              f"cos={m['cosine']:.6f} shape={m['shape']}")
    return 0


def _unused_mel_note():
    # K4 feeds the STORED pre_encode window (not mel) into the streaming
    # path, so no mel orientation helper is needed. Kept as a named stub so
    # a future mel-level probe has one obvious place to hang it.
    raise NotImplementedError("K4 probes from pre_encode, not mel")


if __name__ == "__main__":
    raise SystemExit(main())
