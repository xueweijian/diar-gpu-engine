"""M2 Stage 2 K3 mechanics tests (no torch/NeMo/.npz needed).

Pins what can be pinned locally:
- kernel script compiles;
- pure-python K3 conv2d/dw/pw/flat/lin helpers match the C++
  subsampling path on synthetic weights via a compiled driver
  (transcription check);
- pre_encode key fragments documented in the kernel.
"""
from __future__ import annotations

import py_compile
import struct
import subprocess
import tempfile
import os
from pathlib import Path

K3 = Path(__file__).parent.parent / "kaggle" / "m2_stage2" / "m2_stage2_k3.py"
ROOT = Path(__file__).parent.parent


def _load():
    src = K3.read_text()
    src = src.replace("HARNESS_DIR = locate_harness()\nsys.path.insert(0, str(HARNESS_DIR))\n\nimport diar_harness as h  # noqa: E402\n",
                      "class _H:\n    @staticmethod\n    def environment_record(): return {}\n    @staticmethod\n    def gpu_snapshot(): return ''\n    @staticmethod\n    def emit_report(*a, **k): pass\nh = _H()\n")
    ns: dict = {"__name__": "m2_stage2_k3_test"}
    exec(compile(src, str(K3), "exec"), ns)
    return ns


def test_k3_compiles() -> None:
    assert K3.exists(), "k3 script missing"
    py_compile.compile(str(K3), doraise=True)


def test_k3_verdict_is_gated() -> None:
    # K3 is a full pre_encode gate now, verdict green/red against pinned
    # thresholds (v13 masked-tail fix).
    text = K3.read_text()
    assert "k3-green" in text and "k3-red" in text
    assert "k3-measured" not in text
    assert "k3-scaffold" not in text


def test_gate_constants_pinned_v13() -> None:
    # v13: the three-gate thresholds are pinned constants — changing one is
    # a deliberate test edit, never a silent kernel drift.
    ns = _load()
    assert ns["GATE_VALID_MAX_ABS"] == 8e-05
    assert ns["GATE_FILL_MAX_ABS"] == 0.0
    assert "GATE_MAX_ABS = 1.2e-06" in (ROOT / "kaggle" / "m2_stage2" /
                                        "m2_stage2_k1.py").read_text()
    assert "GATE_MAX_ABS = 1.2e-05" in (ROOT / "kaggle" / "m2_stage2" /
                                        "m2_stage2_k2.py").read_text()


def test_k3_t2_helpers_keep_layout() -> None:
    # Same transpose trap family: t2/t2c must not transpose.
    ns = _load()
    assert ns["t2"]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert ns["t2c"]([[[[1.0]]]]) == [[[[1.0]]]]


def test_k3_pre_encode_end_to_end() -> None:
    # K3 pre_encode() full path on synthetic weights (mirrors the C++
    # TestTranscription path in test_helpers_match_cpp_subsampling).
    import random
    ns = _load()
    rng = random.Random(33)
    TM, F, C, D = 8, 8, 4, 6
    mel = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]

    def mat4(co, ci):
        return [[[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                 for _ in range(ci)] for _ in range(co)]

    def mat3(c):
        return [[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                for _ in range(c)]

    def mat2(r, c):
        return [[rng.uniform(-0.3, 0.3) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.2, 0.2) for _ in range(n)]

    f3 = ns["ool"](ns["ool"](ns["ool"](F)))
    wt = {"c0_w": mat4(C, 1), "c0_b": vec(C),
          "dw1_w": mat3(C), "dw1_b": vec(C), "pw1_w": mat2(C, C), "pw1_b": vec(C),
          "dw2_w": mat3(C), "dw2_b": vec(C), "pw2_w": mat2(C, C), "pw2_b": vec(C),
          "out_w": mat2(D, C * f3), "out_b": vec(D)}
    got = ns["pre_encode"](mel, wt, F=F, C=C, D=D)
    t3 = ns["ool"](ns["ool"](ns["ool"](TM)))
    assert len(got) == t3, (len(got), t3)
    assert len(got[0]) == D
    assert all(all(__import__("math").isfinite(v) for v in row) for row in got)


def test_helpers_match_cpp_subsampling() -> None:
    import random
    ns = _load()
    rng = random.Random(31)
    TM, F, C, D = 12, 8, 4, 6

    def mat4(co, ci):
        return [[[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                 for _ in range(ci)] for _ in range(co)]

    def mat3(c):
        return [[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                for _ in range(c)]

    def mat2(r, c):
        return [[rng.uniform(-0.3, 0.3) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.2, 0.2) for _ in range(n)]

    mel = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]
    c0_w, c0_b = mat4(C, 1), vec(C)
    dw1_w = mat3(C)
    dw1_b = vec(C)
    pw1_w, pw1_b = mat2(C, C), vec(C)
    dw2_w = mat3(C)
    dw2_b = vec(C)
    pw2_w, pw2_b = mat2(C, C), vec(C)
    f3 = ns["ool"](ns["ool"](ns["ool"](F)))
    out_w, out_b = mat2(D, C * f3), vec(D)

    # Python path through the K3 helpers.
    x = [[list(row) for row in mel]]
    s0 = ns["conv2d"](x, c0_w, c0_b, 1, TM, F, C)
    t1, f1 = ns["ool"](TM), ns["ool"](F)
    d1 = ns["dw"](s0, dw1_w, dw1_b, C, t1, f1)
    t2, f2 = ns["ool"](t1), ns["ool"](f1)
    d1f = [[d1[cc][t][f] for t in range(t2) for f in range(f2)] for cc in range(C)]
    s1 = ns["pw"](d1f, pw1_w, pw1_b, C)
    s1c = [[[s1[cc][t * f2 + f] for f in range(f2)] for t in range(t2)] for cc in range(C)]
    d2 = ns["dw"](s1c, dw2_w, dw2_b, C, t2, f2)
    t3, f3b = ns["ool"](t2), ns["ool"](f2)
    assert f3b == f3
    d2f = [[d2[cc][t][f] for t in range(t3) for f in range(f3)] for cc in range(C)]
    s2 = ns["pw"](d2f, pw2_w, pw2_b, C)
    s2c = [[[s2[cc][t * f3 + f] for f in range(f3)] for t in range(t3)] for cc in range(C)]
    flat = ns["flat_cf"](s2c, C, t3, f3)
    py = ns["lin"](flat, out_w, out_b)

    # C++ path.
    td = tempfile.mkdtemp()
    files = {}
    for name, vals, shape in (
            ("mel", [v for row in mel for v in row], None),
            ("c0w", [v for o in c0_w for i in o for r in i for v in r], None),
            ("c0b", c0_b, None), ("dw1w", [v for c in dw1_w for r in c for v in r], None),
            ("dw1b", dw1_b, None), ("pw1w", [v for r in pw1_w for v in r], None),
            ("pw1b", pw1_b, None), ("dw2w", [v for c in dw2_w for r in c for v in r], None),
            ("dw2b", dw2_b, None), ("pw2w", [v for r in pw2_w for v in r], None),
            ("pw2b", pw2_b, None), ("outw", [v for r in out_w for v in r], None),
            ("outb", out_b, None)):
        p = os.path.join(td, name + ".bin")
        with open(p, "wb") as f:
            f.write(struct.pack(f"{len(vals)}f", *vals))
        files[name] = p
    yp = os.path.join(td, "y.bin")
    drv = os.path.join(td, "drv.cpp")
    Path(drv).write_text(
        '#include <cstdio>\n#include <vector>\n#include "diar/subsampling.hpp"\n'
        f"int main(int argc,char**argv){{const int TM={TM},F={F},C={C},D={D};\n"
        "auto load=[&](const char*p){FILE*f=fopen(p,\"rb\");std::vector<float>v;float x;"
        "while(fread(&x,4,1,f)==1)v.push_back(x);fclose(f);return v;};\n"
        "auto mel=load(argv[1]);auto c0w=load(argv[2]);auto c0b=load(argv[3]);\n"
        "auto dw1w=load(argv[4]);auto dw1b=load(argv[5]);auto pw1w=load(argv[6]);auto pw1b=load(argv[7]);\n"
        "auto dw2w=load(argv[8]);auto dw2b=load(argv[9]);auto pw2w=load(argv[10]);auto pw2b=load(argv[11]);\n"
        "auto outw=load(argv[12]);auto outb=load(argv[13]);\n"
        "diar::SubsamplingWeights W;c0w.size();\n"
        "W.c0_w=c0w.data();W.c0_b=c0b.data();W.dw1_w=dw1w.data();W.dw1_b=dw1b.data();\n"
        "W.pw1_w=pw1w.data();W.pw1_b=pw1b.data();W.dw2_w=dw2w.data();W.dw2_b=dw2b.data();\n"
        "W.pw2_w=pw2w.data();W.pw2_b=pw2b.data();W.out_w=outw.data();W.out_b=outb.data();\n"
        "std::vector<float>y(2*D,0);diar::subsampling_forward(mel.data(),W,y.data(),TM,F,C,D);\n"
        "FILE*f=fopen(argv[14],\"wb\");fwrite(y.data(),4,2*D,f);fclose(f);return 0;}\n")
    exe = os.path.join(td, "drv")
    r = subprocess.run(["g++", "-std=c++17", "-O2", f"-I{ROOT}/include", drv,
                        f"{ROOT}/src/nn.cpp", f"{ROOT}/src/subsampling.cpp", "-o", exe],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    args = [exe, files["mel"], files["c0w"], files["c0b"], files["dw1w"], files["dw1b"],
            files["pw1w"], files["pw1b"], files["dw2w"], files["dw2b"], files["pw2w"],
            files["pw2b"], files["outw"], files["outb"], yp]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cpp = struct.unpack(f"{t3 * D}f", open(yp, "rb").read())
    for i in range(t3):
        for j in range(D):
            assert abs(py[i][j] - cpp[i * D + j]) < 2e-4, f"pre_encode [{i},{j}]"


def test_key_fragments_documented() -> None:
    text = K3.read_text()
    for frag in ("pre_encode", "conv.0", "pre_encode.out", "dw_striding"):
        assert frag in text, f"kernel missing key fragment {frag}"


def test_tail_masks_to_valid_rows() -> None:
    # v6 triage (2026-09-17): tail chunks carry zero-padded mel rows whose
    # stem outputs are padding artifacts (short/chunk035 valid 11/12 rows,
    # mid/chunk223 valid 6/8, ref tail row rms 0.34 vs ~13-22 valid). The
    # gate must slice to state_lens_before[2] before metrics.
    text = K3.read_text()
    assert "state_lens_before" in text, "tail masking missing"
    assert "n_valid" in text, "tail masking missing"


def test_pre_encode_matches_staged_helpers() -> None:
    # pre_encode() must equal the staged conv2d/dw/pw/flat_cf/lin path on
    # the SAME synthetic weights (guards the vectorized rewrite against
    # stage-order or reshape drift; the staged path itself is pinned to
    # C++ by test_helpers_match_cpp_subsampling).
    import random
    ns = _load()
    rng = random.Random(37)
    TM, F, C, D = 12, 8, 4, 6

    def mat4(co, ci):
        return [[[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                 for _ in range(ci)] for _ in range(co)]

    def mat3(c):
        return [[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                for _ in range(c)]

    def mat2(r, c):
        return [[rng.uniform(-0.3, 0.3) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.2, 0.2) for _ in range(n)]

    mel = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]
    wt = {"c0_w": mat4(C, 1), "c0_b": vec(C),
          "dw1_w": mat3(C), "dw1_b": vec(C), "pw1_w": mat2(C, C), "pw1_b": vec(C),
          "dw2_w": mat3(C), "dw2_b": vec(C), "pw2_w": mat2(C, C), "pw2_b": vec(C),
          "out_w": mat2(D, C * ns["ool"](ns["ool"](ns["ool"](F)))), "out_b": vec(D)}
    got = ns["pre_encode"](mel, wt, F=F, C=C, D=D)
    # Staged path through the facades.
    x = [[list(row) for row in mel]]
    s0 = ns["conv2d"](x, wt["c0_w"], wt["c0_b"], 1, TM, F, C)
    t1, f1 = ns["ool"](TM), ns["ool"](F)
    d1 = ns["dw"](s0, wt["dw1_w"], wt["dw1_b"], C, t1, f1)
    t2, f2 = ns["ool"](t1), ns["ool"](f1)
    d1f = [[d1[cc][t][f] for t in range(t2) for f in range(f2)] for cc in range(C)]
    s1 = ns["pw"](d1f, wt["pw1_w"], wt["pw1_b"], C)
    s1c = [[[s1[cc][t * f2 + f] for f in range(f2)] for t in range(t2)] for cc in range(C)]
    d2 = ns["dw"](s1c, wt["dw2_w"], wt["dw2_b"], C, t2, f2)
    t3, f3 = ns["ool"](t2), ns["ool"](f2)
    d2f = [[d2[cc][t][f] for t in range(t3) for f in range(f3)] for cc in range(C)]
    s2 = ns["pw"](d2f, wt["pw2_w"], wt["pw2_b"], C)
    s2c = [[[s2[cc][t * f3 + f] for f in range(f3)] for t in range(t3)] for cc in range(C)]
    want = ns["lin"](ns["flat_cf"](s2c, C, t3, f3), wt["out_w"], wt["out_b"])
    assert len(got) == len(want) == t3
    for i in range(t3):
        for j in range(D):
            assert abs(got[i][j] - want[i][j]) < 1e-9, f"staged [{i},{j}]"


# ---------------------------------------------------------------------------
# v13 masked-tail semantics (NeMo MaskedConvSequential, subsampling.py 3.0.0)
# ---------------------------------------------------------------------------

def _naive_masked_pre_encode(mel, wt, feat_len, F, C, D):
    """Literal transcription of NeMo MaskedConvSequential: per-layer
    multiplicative time masks, lengths iterated from feat_len via the floor
    formula. Independent hand loops — deliberately NOT the kernel helpers."""
    T = len(mel)
    L = [feat_len]
    for _ in range(3):
        L.append((L[-1] + 2 - 3) // 2 + 1)

    x = [[[0.0] * F for _ in range(T)]]
    for t in range(T):
        for f in range(F):
            x[0][t][f] = mel[t][f] if t < feat_len else 0.0  # input mask

    def conv(x, w, b, relu):
        H, W = len(x[0]), len(x[0][0])
        ho, wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        y = [[[0.0] * wo for _ in range(ho)] for _ in range(len(w))]
        for o in range(len(w)):
            for h in range(ho):
                for ww in range(wo):
                    acc = b[o]
                    for i in range(len(x)):
                        for kh in range(3):
                            for kw in range(3):
                                hi, wi = h * 2 - 1 + kh, ww * 2 - 1 + kw
                                if 0 <= hi < H and 0 <= wi < W:
                                    acc += x[i][hi][wi] * w[o][i][kh][kw]
                    y[o][h][ww] = max(acc, 0.0) if relu else acc
        return y

    def dw(x, w, b):
        H, W = len(x[0]), len(x[0][0])
        ho, wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        y = [[[0.0] * wo for _ in range(ho)] for _ in range(len(w))]
        for ch in range(len(w)):
            for h in range(ho):
                for ww in range(wo):
                    acc = b[ch]
                    for kh in range(3):
                        for kw in range(3):
                            hi, wi = h * 2 - 1 + kh, ww * 2 - 1 + kw
                            if 0 <= hi < H and 0 <= wi < W:
                                acc += x[ch][hi][wi] * w[ch][kh][kw]
                    y[ch][h][ww] = acc
        return y

    def pw(x, w, b):
        H, W = len(x[0]), len(x[0][0])
        y = [[[0.0] * W for _ in range(H)] for _ in range(len(w))]
        for o in range(len(w)):
            for h in range(H):
                for ww in range(W):
                    acc = b[o]
                    for i in range(len(x)):
                        acc += x[i][h][ww] * w[o][i]
                    y[o][h][ww] = acc
        return y

    def relu(x):
        return [[[max(v, 0.0) for v in row] for row in ch] for ch in x]

    def mask_rows(x, l):
        for ch in x:
            for h in range(l, len(ch)):
                for ww in range(len(ch[h])):
                    ch[h][ww] = 0.0

    # NeMo layer order with per-layer masks (strided: conv.0, conv.2, conv.5).
    a0 = conv(x, wt["c0_w"], wt["c0_b"], relu=False)
    mask_rows(a0, L[1])
    s0 = relu(a0)
    d1 = dw(s0, wt["dw1_w"], wt["dw1_b"])
    mask_rows(d1, L[2])
    p1 = pw(d1, wt["pw1_w"], wt["pw1_b"])
    mask_rows(p1, L[2])
    s1 = relu(p1)
    d2 = dw(s1, wt["dw2_w"], wt["dw2_b"])
    mask_rows(d2, L[3])
    p2 = pw(d2, wt["pw2_w"], wt["pw2_b"])
    mask_rows(p2, L[3])
    s2 = relu(p2)
    mask_rows(s2, L[3])  # final mask
    t3, f3 = len(s2[0]), len(s2[0][0])
    out = []
    for t in range(t3):
        row = []
        for o in range(D):
            acc = wt["out_b"][o]
            i = 0
            for cc in range(C):
                for ff in range(f3):
                    acc += s2[cc][t][ff] * wt["out_w"][o][i]
                    i += 1
            row.append(acc)
        out.append(row)
    return out, L


def _rand_wt(rng, F, C, D, f3):
    def mat4(co, ci):
        return [[[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                 for _ in range(ci)] for _ in range(co)]

    def mat3(c):
        return [[[rng.uniform(-0.3, 0.3) for _ in range(3)] for _ in range(3)]
                for _ in range(c)]

    def mat2(r, c):
        return [[rng.uniform(-0.3, 0.3) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.2, 0.2) for _ in range(n)]

    return {"c0_w": mat4(C, 1), "c0_b": vec(C),
            "dw1_w": mat3(C), "dw1_b": vec(C), "pw1_w": mat2(C, C), "pw1_b": vec(C),
            "dw2_w": mat3(C), "dw2_b": vec(C), "pw2_w": mat2(C, C), "pw2_b": vec(C),
            "out_w": mat2(D, C * f3), "out_b": vec(D)}


def test_masked_matches_naive_reference() -> None:
    # The kernel's post-activation zeroing must be BIT-equivalent to NeMo's
    # literal per-layer masking on random weights, including a boundary row
    # whose receptive field crosses masked rows (the v12 chunk035 row10
    # failure mode) and tail fill rows.
    import random
    ns = _load()
    rng = random.Random(4141)
    TM, F, C, D, feat_len = 24, 8, 4, 6, 16  # grid 12/6/3, masks 8/4/2 -> tail row 2
    mel = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]
    f3 = ns["ool"](ns["ool"](ns["ool"](F)))
    wt = _rand_wt(rng, F, C, D, f3)
    # Precondition: non-degenerate weights (else the test passes vacuously).
    assert ns["pre_encode"](mel, wt, F=F, C=C, D=D)[0] != wt["out_b"]
    want, L = _naive_masked_pre_encode(mel, wt, feat_len, F, C, D)
    assert L == [16, 8, 4, 2], L
    got = ns["pre_encode"](mel, wt, F=F, C=C, D=D, feat_len=feat_len)
    assert len(got) == len(want) == 3
    for i in range(3):
        for j in range(D):
            assert abs(got[i][j] - want[i][j]) < 1e-9, f"masked naive [{i},{j}]"


def test_masked_fill_rows_exactly_out_bias() -> None:
    # Rows >= L3 must be EXACTLY out.bias (both sides compute out(0)), and
    # therefore bit-identical across different inputs — the v12 fingerprint
    # (fill sha 8f441a7ac60126dc, rms 0.337, identical across audios).
    import random
    ns = _load()
    rng = random.Random(55)
    TM, F, C, D, feat_len = 24, 8, 4, 6, 16
    f3 = ns["ool"](ns["ool"](ns["ool"](F)))
    wt = _rand_wt(rng, F, C, D, f3)
    mel1 = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]
    mel2 = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]
    # Precondition: these weights are non-degenerate (unmasked row 0 is a
    # computed row, not a collapsed out.bias — seed 777 collapses via an
    # all-negative pw2 pre-activation and proves nothing).
    pre = ns["pre_encode"](mel1, wt, F=F, C=C, D=D)
    assert pre[0] != wt["out_b"], "degenerate weights: pick another seed"
    got1 = ns["pre_encode"](mel1, wt, F=F, C=C, D=D, feat_len=feat_len)
    got2 = ns["pre_encode"](mel2, wt, F=F, C=C, D=D, feat_len=feat_len)
    assert got1[2] == wt["out_b"], "tail row must equal out.bias exactly"
    assert got2[2] == wt["out_b"], "tail row must equal out.bias exactly"
    assert got1[2] == got2[2], "fill rows must be input-independent"
    # Sanity: valid rows differ across inputs (rows 0-1 are computed).
    assert got1[0] != got2[0], "valid rows must be input-dependent"


def test_full_window_mask_is_noop() -> None:
    # feat_len == T (full chunks, e.g. chunk000: 160/160) must reproduce the
    # historical unmasked behavior bit-for-bit (regression guard).
    import random
    ns = _load()
    rng = random.Random(99)
    TM, F, C, D = 24, 8, 4, 6
    f3 = ns["ool"](ns["ool"](ns["ool"](F)))
    wt = _rand_wt(rng, F, C, D, f3)
    mel = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]
    a = ns["pre_encode"](mel, wt, F=F, C=C, D=D)
    b = ns["pre_encode"](mel, wt, F=F, C=C, D=D, feat_len=TM)
    assert a == b


def test_masked_matches_cpp_subsampling() -> None:
    # Python mirror vs C++ src/subsampling.cpp on the SAME masked case
    # (transcription check across languages, incl. tail fill rows).
    import random
    ns = _load()
    rng = random.Random(2121)
    TM, F, C, D, feat_len = 24, 8, 4, 6, 16
    f3 = ns["ool"](ns["ool"](ns["ool"](F)))
    wt = _rand_wt(rng, F, C, D, f3)
    mel = [[rng.uniform(-0.5, 0.5) for _ in range(F)] for _ in range(TM)]
    # Precondition: non-degenerate weights (else comparisons pass vacuously).
    assert ns["pre_encode"](mel, wt, F=F, C=C, D=D)[0] != wt["out_b"]
    got = ns["pre_encode"](mel, wt, F=F, C=C, D=D, feat_len=feat_len)
    t3 = ns["ool"](ns["ool"](ns["ool"](TM)))

    td = tempfile.mkdtemp()
    names = {"mel": [v for row in mel for v in row],
             "c0w": [v for o in wt["c0_w"] for i in o for r in i for v in r],
             "c0b": wt["c0_b"],
             "dw1w": [v for c in wt["dw1_w"] for r in c for v in r], "dw1b": wt["dw1_b"],
             "pw1w": [v for r in wt["pw1_w"] for v in r], "pw1b": wt["pw1_b"],
             "dw2w": [v for c in wt["dw2_w"] for r in c for v in r], "dw2b": wt["dw2_b"],
             "pw2w": [v for r in wt["pw2_w"] for v in r], "pw2b": wt["pw2_b"],
             "outw": [v for r in wt["out_w"] for v in r], "outb": wt["out_b"]}
    files = {}
    for name, vals in names.items():
        p = os.path.join(td, name + ".bin")
        with open(p, "wb") as f:
            f.write(struct.pack(f"{len(vals)}f", *vals))
        files[name] = p
    yp = os.path.join(td, "y.bin")
    drv = os.path.join(td, "drv.cpp")
    Path(drv).write_text(
        '#include <cstdio>\n#include <vector>\n#include "diar/subsampling.hpp"\n'
        f"int main(int argc,char**argv){{const int TM={TM},F={F},C={C},D={D},FL={feat_len};\n"
        "(void)argc;\n"
        "auto load=[&](const char*p){FILE*f=fopen(p,\"rb\");std::vector<float>v;float x;"
        "while(fread(&x,4,1,f)==1)v.push_back(x);fclose(f);return v;};\n"
        "auto mel=load(argv[1]);auto c0w=load(argv[2]);auto c0b=load(argv[3]);\n"
        "auto dw1w=load(argv[4]);auto dw1b=load(argv[5]);auto pw1w=load(argv[6]);auto pw1b=load(argv[7]);\n"
        "auto dw2w=load(argv[8]);auto dw2b=load(argv[9]);auto pw2w=load(argv[10]);auto pw2b=load(argv[11]);\n"
        "auto outw=load(argv[12]);auto outb=load(argv[13]);\n"
        "diar::SubsamplingWeights W;\n"
        "W.c0_w=c0w.data();W.c0_b=c0b.data();W.dw1_w=dw1w.data();W.dw1_b=dw1b.data();\n"
        "W.pw1_w=pw1w.data();W.pw1_b=pw1b.data();W.dw2_w=dw2w.data();W.dw2_b=dw2b.data();\n"
        "W.pw2_w=pw2w.data();W.pw2_b=pw2b.data();W.out_w=outw.data();W.out_b=outb.data();\n"
        "std::vector<float>y(3*D,0);diar::subsampling_forward(mel.data(),W,y.data(),TM,F,C,D,FL);\n"
        "FILE*f=fopen(argv[14],\"wb\");fwrite(y.data(),4,3*D,f);fclose(f);return 0;}\n")
    exe = os.path.join(td, "drv")
    r = subprocess.run(["g++", "-std=c++17", "-O2", f"-I{ROOT}/include", drv,
                        f"{ROOT}/src/nn.cpp", f"{ROOT}/src/subsampling.cpp", "-o", exe],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    args = [exe] + [files[k] for k in ("mel", "c0w", "c0b", "dw1w", "dw1b", "pw1w",
                                       "pw1b", "dw2w", "dw2b", "pw2w", "pw2b",
                                       "outw", "outb")] + [yp]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cpp = struct.unpack(f"{t3 * D}f", open(yp, "rb").read())
    for i in range(t3):
        for j in range(D):
            assert abs(got[i][j] - cpp[i * D + j]) < 2e-4, f"masked cpp [{i},{j}]"
    # Fill row agreement with out.bias must hold on the C++ side exactly too
    # (compare against the float32 round-trip of the python float64 bias —
    # the .bin roundtrip is what production uses on both sides).
    ob32 = struct.unpack(f"{D}f", struct.pack(f"{D}f", *wt["out_b"]))
    for j in range(D):
        assert cpp[2 * D + j] == ob32[j], f"cpp fill [{j}] != out.bias(fp32)"
