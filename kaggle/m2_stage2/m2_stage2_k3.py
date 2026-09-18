"""M2 Stage 2 K3: teacher-forced gate pre_encode stem (mel_window -> pre_encode).

Numpy-vectorized mirror of src/subsampling.cpp (mechanics tests pin the
transcription through the SAME helpers the kernel calls: conv2d/dw/pw/
flat_cf/lin below now take/return nested lists but compute in numpy —
identical values, ~50x faster than the old triple loops; a 160x128 mel
window drops from ~60s to ~1s, so the full 260-chunk gate fits the kernel
budget). Kernel-side: weights from the .nemo state dict
(encoder.pre_encode.conv.* + encoder.pre_encode.out.*), gate on every chunk
(all 36 short + all 224 mid).

Verdict: K3-measured with worst_max_abs. Thresholds pinned after.
dw_striding index layout (subsampling.py dw_striding branch): conv.0/pw…
resolved against the checkpoint key list at runtime; any schema fork fails
LOUDLY in main() (no silent index).
"""
from __future__ import annotations

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
    "job": "m2_stage2_k3_preencode",
}


def ool(x):
    return (x + 2 - 3) // 2 + 1


def conv2d(x, w, b, ci, hi, wi, co):
    # Nested-list facade over the numpy stage (kept for the mechanics
    # transcription test, which calls THIS name). x: [ci][H][W] nested,
    # w: [co][ci][3][3] nested. Stride 2, pad 1, floor, +ReLU (baked, as in
    # the torch dw_striding conv0 and the C++ conv2d_relu(relu=true)).
    import numpy as _np
    xa = _np.asarray(x, dtype=_np.float64).reshape(ci, hi, wi)
    wa = _np.asarray(w, dtype=_np.float64).reshape(co, ci, 3, 3)
    xp = _np.pad(xa, ((0, 0), (1, 1), (1, 1)))
    ho, wo = (hi + 2 - 3) // 2 + 1, (wi + 2 - 3) // 2 + 1
    cols = _np.stack([xp[:, h:h + 2 * ho:2][:, :, ww:ww + 2 * wo:2]
                      for h in range(3) for ww in range(3)], axis=1)
    y = (wa.reshape(co, -1) @ cols.reshape(ci * 9, -1)) \
        .reshape(co, ho, wo) + _np.asarray(b, dtype=_np.float64)[:, None, None]
    return _np.maximum(y, 0.0).tolist()


def dw(x, w, b, c, hi, wi):
    # Nested-list facade, grouped DW (NO relu — matches torch dw_striding
    # depthwise stages and C++ conv2d_dw_relu(relu=false)).
    import numpy as _np
    xa = _np.asarray(x, dtype=_np.float64).reshape(c, hi, wi)
    wa = _np.asarray(w, dtype=_np.float64).reshape(c, 3, 3)
    xp = _np.pad(xa, ((0, 0), (1, 1), (1, 1)))
    ho, wo = (hi + 2 - 3) // 2 + 1, (wi + 2 - 3) // 2 + 1
    cols = _np.stack([xp[:, h:h + 2 * ho:2][:, :, ww:ww + 2 * wo:2]
                      for h in range(3) for ww in range(3)], axis=1)
    # cols: [C, 9, ho, wo]; w is [C,3,3] here; reshape taps kh-outer to [C,9]
    # to match the stack order (h outer, ww inner) — "ck" over the raw 3D
    # array would throw (fixed 2026-09-17, caught by static audit).
    y = _np.einsum("ck,ckhw->chw", wa.reshape(c, 9), cols)
    return (y + _np.asarray(b, dtype=_np.float64)[:, None, None]).tolist()


def pw(x, w, b, c):
    # Nested-list facade, pointwise +ReLU (baked).
    import numpy as _np
    xa = _np.asarray(x, dtype=_np.float64)
    y = _np.asarray(w, dtype=_np.float64) @ xa + _np.asarray(b, dtype=_np.float64)[:, None]
    return _np.maximum(y, 0.0).tolist()


def flat_cf(x, c, t, f):
    # x[C][T][F] -> rows[T][C*F] (torch reshape(b,t,-1) of (b,c,t,f).T(1,2)).
    return [[x[cc][t][ff] for cc in range(c) for ff in range(f)] for t in range(t)]


def lin(x, w, b):
    # Nested-list facade, Linear [out,in] (NO transpose) + bias, no act.
    import numpy as _np
    y = _np.asarray(x, dtype=_np.float64) @ _np.asarray(w, dtype=_np.float64).T
    if b is not None:
        y = y + _np.asarray(b, dtype=_np.float64)[None, :]
    return y.tolist()


def pre_encode(mel_tf, wt, F=128, C=256, D=512, feat_len=None):
    # Numpy-vectorized stem (same perf rewrite as K1/K2): identical math to
    # src/subsampling.cpp — conv0+ReLU, (DW/no-ReLU, pw+ReLU) x2, flat, out.
    # Nested-list I/O preserved for the mechanics transcription tests.
    # dw_striding index layout (subsampling.py dw_striding branch):
    #   conv.0 (Conv2d 1->C) / conv.1 (ReLU) / conv.2 (DW) / conv.3 (pw) /
    #   conv.4 (ReLU) / conv.5 (DW) / conv.6 (pw) / conv.7 (ReLU),
    #   then pre_encode.out (Linear C*F3 -> D). Resolved against the
    #   checkpoint key list in main(); any schema fork fails LOUDLY there.
    #
    # v13 masked-tail semantics (NeMo 3.0.0 subsampling.py MaskedConvSequential,
    # pinned 2026-09-18 from the v12 P6 probe): the conv stack runs on the
    # FULL window grid (T_full = window/8 rows), but a multiplicative time
    # mask is applied around every layer using per-stage floor lengths
    # L_{k+1} = (L_k + 2 - 3)//2 + 1 iterated from feat_len (NOT window):
    # 86 -> 43 -> 22 -> 11 == state_lens_before[2]. Rows >= Lk are zeroed
    # after level k's activation (bit-equivalent to NeMo's per-layer mask:
    # only zero-vs-nonzero matters to the next strided consumer). Consequence
    # 1: boundary valid rows tap masked zero rows instead of bias chains
    # (short/chunk035 row10 divergence, v12 34.8 -> fixed). Consequence 2:
    # final rows >= L3 have all-zero features -> out(0) == out.bias EXACTLY —
    # the bit-identical fill rows (rms 0.337, sha 8f441a7ac60126dc) seen
    # across both audios' tail chunks.
    import numpy as _np
    mel = _np.array(mel_tf, dtype=_np.float64)  # [T, F] (fresh copy, masked below)
    T = mel.shape[0]
    if feat_len is None:
        feat_len = T
    if not (0 < feat_len <= T):
        raise ValueError(f"feat_len {feat_len} outside (0, {T}]")
    mel[int(feat_len):, :] = 0.0  # input time mask (mask before conv.0)
    L1, L2, L3 = ool(feat_len), ool(ool(feat_len)), ool(ool(ool(feat_len)))

    def conv_stage(x, w, b, relu):
        # x: [Ci, H, W] (numpy), w: [Co, Ci, 3, 3]. Stride 2, pad 1, floor.
        Ci, H, W = x.shape
        Co = w.shape[0]
        xp = _np.pad(x, ((0, 0), (1, 1), (1, 1)))
        ho, wo = (H + 2 - 3) // 2 + 1, (W + 2 - 3) // 2 + 1
        # Strided windows: xp[:, h:h+2*ho:2] picks rows h, h+2, ... (ho rows).
        cols = _np.stack([xp[:, h:h + 2 * ho:2][:, :, ww:ww + 2 * wo:2]
                          for h in range(3) for ww in range(3)], axis=1)
        # cols: [Ci, 9, ho, wo] -> [Ci*9, ho*wo]; GEMM vs [Co, Ci*9].
        y = (w.reshape(Co, -1) @ cols.reshape(Ci * 9, -1)) \
            .reshape(Co, ho, wo) + _np.asarray(b, dtype=_np.float64)[:, None, None]
        return _np.maximum(y, 0.0) if relu else y

    def dw_stage(x, w, b):
        # x: [C, H, W], w: [C, 3, 3] grouped. Stride 2, pad 1, NO relu.
        C, H, W = x.shape
        xp = _np.pad(x, ((0, 0), (1, 1), (1, 1)))
        ho, wo = (H + 2 - 3) // 2 + 1, (W + 2 - 3) // 2 + 1
        cols = _np.stack([xp[:, h:h + 2 * ho:2][:, :, ww:ww + 2 * wo:2]
                          for h in range(3) for ww in range(3)], axis=1)
        # cols: [C, 9, ho, wo]; contract taps per channel. NOTE w arrives as
        # [C,3,3] (kept 3D by t2c); reshape to [C,9] with kh-outer order to
        # match the stack order (h outer, ww inner) — fixed 2026-09-17,
        # "ck" over a 3D array would throw.
        y = _np.einsum("ck,ckhw->chw", _np.asarray(w, dtype=_np.float64).reshape(C, 9), cols)
        return y + _np.asarray(b, dtype=_np.float64)[:, None, None]

    def pw_stage(x, w, b):
        # x: [C, HW] flattened tap grid; w: [C, C]. +ReLU (baked).
        y = _np.asarray(w, dtype=_np.float64) @ x \
            + _np.asarray(b, dtype=_np.float64)[:, None]
        return _np.maximum(y, 0.0)

    c0_w = _np.asarray(wt["c0_w"], dtype=_np.float64)  # [C,1,3,3]
    # Channel-first input [1, T, F] (mel rows = time, cols = freq).
    xc = mel[None, :, :]  # [1, T, F]
    s0 = conv_stage(xc, c0_w, wt["c0_b"], True)  # [C, T1, F1]
    s0[:, L1:, :] = 0.0  # v13: per-stage length mask (rows >= L1)
    C_ = s0.shape[0]
    d1 = dw_stage(s0, _np.asarray(wt["dw1_w"], dtype=_np.float64), wt["dw1_b"])
    t2_, f2_ = d1.shape[1], d1.shape[2]
    s1 = pw_stage(d1.reshape(C_, -1), _np.asarray(wt["pw1_w"], dtype=_np.float64),
                  wt["pw1_b"]).reshape(C_, t2_, f2_)
    s1[:, L2:, :] = 0.0  # v13: per-stage length mask (rows >= L2)
    d2 = dw_stage(s1, _np.asarray(wt["dw2_w"], dtype=_np.float64), wt["dw2_b"])
    t3, f3 = d2.shape[1], d2.shape[2]
    s2 = pw_stage(d2.reshape(C_, -1), _np.asarray(wt["pw2_w"], dtype=_np.float64),
                  wt["pw2_b"]).reshape(C_, t3, f3)
    s2[:, L3:, :] = 0.0  # v13: final mask -> rows >= L3 flatten to zero features
    flat = s2.transpose(1, 0, 2).reshape(t3, -1)  # torch (b,t,c*f)
    out = flat @ _np.asarray(wt["out_w"], dtype=_np.float64).T \
        + _np.asarray(wt["out_b"], dtype=_np.float64)[None, :]
    return out.tolist()


def t2c(w):
    # Conv2d weight [out,in,kH,kW] -> nested [o][i][kh][kw] (NO transpose).
    import numpy as _np
    return _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w,
                       dtype=_np.float64).tolist()


def t2(w):
    # Linear weight [out,in] -> nested [o][i] (NO transpose; same fix as K1).
    import numpy as _np
    return _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w,
                       dtype=_np.float64).tolist()


def v1(w):
    import numpy as _np
    return _np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w,
                       dtype=_np.float64).ravel().tolist()


# v13 pinned gates. valid-row threshold: v12 measured 259/260 green at
# worst ~5-8e-05 (fp32-vs-fp64 spread); the 260th (short/chunk035 row10)
# was the masked-tail bug fixed in v13. fill rows: out.bias on BOTH sides,
# compared EXACTLY (0*w sums are exact zeros in fp64 and fp32 alike).
GATE_VALID_MAX_ABS = 8e-05
GATE_FILL_MAX_ABS = 0.0


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
        h.emit_report(REPORT, name="m2_stage2_k3_verdict.json")
        return 0
    import tarfile, io
    import torch
    with tarfile.open(str(ckpts[0]), "r") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".ckpt"):
                f = tf.extractfile(m)
                obj = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
                sd = obj.get("state_dict", obj)
                break
    keys = sorted([k for k in sd if "pre_encode" in k])
    REPORT["pre_encode_keys"] = keys
    ref_dir = ref_hits[0].parent
    REPORT["ref_dir"] = str(ref_dir)
    REPORT["ckpt"] = str(ckpts[0])
    # Resolve dw_striding index layout against the checkpoint (loud on fork).
    p = "encoder.pre_encode."
    wt = {"c0_w": t2c(sd[p + "conv.0.weight"]), "c0_b": v1(sd[p + "conv.0.bias"]),
          "dw1_w": t2c(sd[p + "conv.2.weight"]), "dw1_b": v1(sd[p + "conv.2.bias"]),
          "pw1_w": t2(sd[p + "conv.3.weight"]), "pw1_b": v1(sd[p + "conv.3.bias"]),
          "dw2_w": t2c(sd[p + "conv.5.weight"]), "dw2_b": v1(sd[p + "conv.5.bias"]),
          "pw2_w": t2(sd[p + "conv.6.weight"]), "pw2_b": v1(sd[p + "conv.6.bias"]),
          "out_w": t2(sd[p + "out.weight"]), "out_b": v1(sd[p + "out.bias"])}
    # pw 1x1 weights are [C,C,1,1] -> squeeze to [C,C] for the pw() helper.
    for k in ("pw1_w", "pw2_w"):
        w = wt[k]
        if isinstance(w[0][0], list):
            wt[k] = [[v[0][0] if isinstance(v[0], list) else v[0] for v in row] for row in w]
    # DW weights are [C,1,3,3] -> squeeze dim-1 to [C,3,3] for the dw() helper
    # (fixed 2026-09-17: t2c keeps [o][i][kh][kw]; groups=C means i==0 only).
    for k in ("dw1_w", "dw2_w"):
        w = wt[k]
        if isinstance(w[0][0], list) and len(w[0]) == 1:
            wt[k] = [ch[0] for ch in w]
    # c0_w is [C,1,3,3] and conv2d() indexes w[o][i][kh][kw] with ci==1:
    # w[o][0][kh][kw] resolves correctly, no squeeze needed.
    per_audio = {}
    for label in ("short", "mid"):
        zpath = ref_dir / f"m2_ref_{label}.npz"
        if not zpath.exists():
            continue
        z = _np.load(str(zpath), allow_pickle=True)
        chunks = sorted(set(k.split("/")[0] for k in z.files if k.startswith("chunk")))
        gates = []
        for chunk in chunks:
            mel = z[chunk + "/mel_window"].tolist()  # [T_mel, 128]
            feat_len = int(z[chunk + "/feat_length"][0])
            got = pre_encode(mel, wt, feat_len=feat_len)
            ref = z[chunk + "/pre_encode"].tolist()
            import math as _m
            # v13: the mirror now reproduces NeMo's masked-tail semantics, so
            # the gate covers ALL rows: valid rows (< n_valid) against fp32
            # spread thresholds, tail rows (>= n_valid) against out.bias
            # EXACTLY (both sides compute out(0-features) == out.bias).
            # n_valid = state_lens_before[2] must equal our iterated floor
            # formula L3 (NeMo calc_length repeat_num=3) — asserted loudly.
            n_valid = int(z[chunk + "/state_lens_before"][2])
            assert 0 < n_valid <= len(ref), (chunk, n_valid, len(ref))
            L3 = ool(ool(ool(feat_len)))
            assert n_valid == L3, (chunk, "length formula drift", n_valid, L3)
            r_all = _np.asarray(ref, dtype=_np.float64)
            g_all = _np.asarray(got, dtype=_np.float64)
            r, g = r_all[:n_valid], g_all[:n_valid]
            out_b = _np.asarray(wt["out_b"], dtype=_np.float64)
            tail_r, tail_g = r_all[n_valid:], g_all[n_valid:]
            gates.append({"chunk": chunk,
                          "max_abs": float(_np.abs(r - g).max()),
                          "mean_abs": float(_np.abs(r - g).mean()),
                          "cosine": float((r * g).sum() / (_m.sqrt((r * r).sum() * (g * g).sum()) + 1e-12)),
                          "fill_max_abs": (float(_np.abs(tail_r - tail_g).max())
                                           if tail_r.shape[0] else 0.0),
                          "fill_vs_out_bias": (float(_np.abs(tail_r - out_b[None, :]).max())
                                               if tail_r.shape[0] else 0.0),
                          "T": n_valid,
                          "T_full": len(ref)})
        per_audio[label] = gates
    REPORT["gates"] = per_audio
    REPORT["worst_max_abs"] = max((m["max_abs"] for gates in per_audio.values() for m in gates),
                                  default=float("nan"))
    REPORT["worst_fill_max_abs"] = max((m["fill_max_abs"] for gates in per_audio.values()
                                        for m in gates), default=0.0)
    REPORT["worst_fill_vs_out_bias"] = max((m["fill_vs_out_bias"] for gates in per_audio.values()
                                            for m in gates), default=0.0)
    ok = (REPORT["worst_max_abs"] <= GATE_VALID_MAX_ABS
          and REPORT["worst_fill_max_abs"] <= GATE_FILL_MAX_ABS
          and REPORT["worst_fill_vs_out_bias"] <= GATE_FILL_MAX_ABS)
    REPORT["gate"] = {"valid_max_abs": GATE_VALID_MAX_ABS, "fill_max_abs": GATE_FILL_MAX_ABS}
    REPORT["verdict"] = "k3-green" if ok else "k3-red"
    REPORT["note"] = ("v13 masked-tail semantics: valid rows gated at "
                      f"{GATE_VALID_MAX_ABS:g}, tail rows must equal out.bias exactly.")
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage2_k3_verdict.json")
    print(json.dumps({k: v for k, v in REPORT.items() if k != "gates"}, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
