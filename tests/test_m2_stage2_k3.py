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


def test_k3_verdict_is_measured() -> None:
    # K3 is a full pre_encode gate now, not a probe scaffold.
    text = K3.read_text()
    assert "k3-measured" in text
    assert "k3-scaffold" not in text


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
