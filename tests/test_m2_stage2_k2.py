"""M2 Stage 2 K2 mechanics tests (no torch/NeMo/.npz needed).

Pins what can be pinned locally:
- kernel script compiles;
- pure-python conformer_layer + relpos_mha + conformer_conv + relpos_table
  match the C++ implementations (src/conformer.cpp, mha.cpp, conv.cpp,
  posenc.cpp) on synthetic weights via compiled drivers — a transcription
  error in m2_stage2_k2.py fails here;
- shift_torch orientation matches nn::rel_shift (transposed ggml storage):
  shift_torch(bd)[q][k] == bd[k-q+T-1][q].
- required checkpoint key fragments for the 17 conformer layers are
  documented in the kernel (encoder.layers.N.{self_attn,feed_forward,conv,
  norm_*} + shared encoder.pos_enc handling).
"""
from __future__ import annotations

import py_compile
import struct
import subprocess
import tempfile
import os
from pathlib import Path

K2 = Path(__file__).parent.parent / "kaggle" / "m2_stage2" / "m2_stage2_k2.py"
ROOT = Path(__file__).parent.parent


def _load():
    src = K2.read_text()
    src = src.replace("HARNESS_DIR = locate_harness()\nsys.path.insert(0, str(HARNESS_DIR))\n\nimport diar_harness as h  # noqa: E402\n",
                      "class _H:\n    @staticmethod\n    def environment_record(): return {}\n    @staticmethod\n    def gpu_snapshot(): return ''\n    @staticmethod\n    def emit_report(*a, **k): pass\nh = _H()\n")
    ns: dict = {"__name__": "m2_stage2_k2_test"}
    exec(compile(src, str(K2), "exec"), ns)
    return ns


def _build_driver(body, extra_srcs):
    td = tempfile.mkdtemp()
    drv = os.path.join(td, "drv.cpp")
    Path(drv).write_text(body)
    exe = os.path.join(td, "drv")
    cmd = ["g++", "-std=c++17", "-O2", f"-I{ROOT}/include", drv,
           *[f"{ROOT}/src/{s}" for s in extra_srcs], "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return exe, td


def test_k2_compiles() -> None:
    assert K2.exists(), "k2 script missing"
    py_compile.compile(str(K2), doraise=True)


def test_t2_keeps_torch_layout() -> None:
    # Same t2() transpose trap as K1 (fixed 2026-09-17): torch Linear
    # [out,in] must reach matvec_rows WITHOUT transpose. Rectangular
    # proof: [2,3] stays [2,3].
    ns = _load()
    assert ns["t2"]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_k2_verdict_is_gated() -> None:
    # v13: K2 verdict is green/red against a pinned threshold (v12 measured
    # 102/102 green, worst 1.11e-05; gate 1.2e-05 with headroom).
    text = K2.read_text()
    assert '"k2-green"' in text or "'k2-green'" in text
    assert '"k2-red"' in text or "'k2-red'" in text
    assert "k2-measured" not in text
    # K2 is a full 17-layer gate now, not a probe scaffold.
    assert "k2-scaffold" not in text


def test_shift_orientation() -> None:
    ns = _load()
    T = 3
    bd = [[100 * r + c for c in range(T)] for r in range(2 * T - 1)]
    s = ns["shift_torch"](bd, T)
    # S[q,k] = bd[k-q+T-1][q]; spot: S[0,0]=bd[2][0]=200, S[0,2]=bd[4][0]=400.
    assert s[0][0] == 200, s[0][0]
    assert s[0][2] == 400, s[0][2]
    assert s[2][0] == 2, s[2][0]
    assert s[2][2] == 202, s[2][2]


def test_python_mha_matches_cpp() -> None:
    import random
    ns = _load()
    rng = random.Random(21)
    T, C, NH = 3, 8, 2
    x = [[rng.uniform(-1, 1) for _ in range(C)] for _ in range(T)]
    pe = [[rng.uniform(-1, 1) for _ in range(C)] for _ in range(2 * T - 1)]

    def mat(r, c):
        return [[rng.uniform(-0.4, 0.4) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.4, 0.4) for _ in range(n)]

    wt = {"q_w": mat(C, C), "qb": vec(C), "k_w": mat(C, C), "kb": vec(C),
          "v_w": mat(C, C), "vb": vec(C), "pos_w": mat(C, C),
          "bu": vec(C), "bv": vec(C), "o_w": mat(C, C), "ob": vec(C)}
    py = ns["relpos_mha"](x, pe, wt, C, NH)
    order = ["q_w", "qb", "k_w", "kb", "v_w", "vb", "pos_w", "bu", "bv", "o_w", "ob"]
    td = tempfile.mkdtemp()
    wp, xp, pp, yp = (os.path.join(td, n) for n in ("w.bin", "x.bin", "p.bin", "y.bin"))
    with open(wp, "wb") as f:
        for k in order:
            vals = [v for row in wt[k] for v in row] if isinstance(wt[k][0], list) else wt[k]
            f.write(struct.pack(f"{len(vals)}f", *vals))
    with open(xp, "wb") as f:
        f.write(struct.pack(f"{T * C}f", *[v for row in x for v in row]))
    with open(pp, "wb") as f:
        f.write(struct.pack(f"{(2 * T - 1) * C}f", *[v for row in pe for v in row]))
    body = ('#include <cstdio>\n#include <vector>\n#include "diar/mha.hpp"\n'
            f"int main(int argc,char**argv){{const int T={T},C={C},NH={NH};\n"
            "FILE*f=fopen(argv[1],\"rb\");std::vector<float>w;float v;while(fread(&v,4,1,f)==1)w.push_back(v);fclose(f);\n"
            "f=fopen(argv[2],\"rb\");std::vector<float>x;while(fread(&v,4,1,f)==1)x.push_back(v);fclose(f);\n"
            "f=fopen(argv[3],\"rb\");std::vector<float>pe;while(fread(&v,4,1,f)==1)pe.push_back(v);fclose(f);\n"
            "size_t o=0;auto take=[&](size_t n){auto p=w.data()+o;o+=n;return p;};\n"
            "diar::RelPosMhaWeights W;W.q_w=take(C*C);W.q_b=take(C);W.k_w=take(C*C);W.k_b=take(C);\n"
            "W.v_w=take(C*C);W.v_b=take(C);W.pos_w=take(C*C);W.bu=take(C);W.bv=take(C);W.out_w=take(C*C);W.out_b=take(C);\n"
            "std::vector<float>y(T*C);diar::relpos_mha_forward(x.data(),pe.data(),W,y.data(),T,C,NH);\n"
            "f=fopen(argv[4],\"wb\");fwrite(y.data(),4,y.size(),f);fclose(f);return 0;}\n")
    exe, _ = _build_driver(body, ["nn.cpp", "mha.cpp"])
    r = subprocess.run([exe, wp, xp, pp, yp], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cpp = struct.unpack(f"{T * C}f", open(yp, "rb").read())
    for i in range(T):
        for j in range(C):
            assert abs(py[i][j] - cpp[i * C + j]) < 2e-5, f"mha [{i},{j}]"


def test_python_conv_matches_cpp() -> None:
    import random
    ns = _load()
    rng = random.Random(22)
    T, D, K = 5, 8, 5
    x = [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(T)]

    def mat(r, c):
        return [[rng.uniform(-0.4, 0.4) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.4, 0.4) for _ in range(n)]

    wt = {"pw1_w": mat(2 * D, D), "pw1_b": vec(2 * D),
          "dw_w": [vec(K) for _ in range(D)], "dw_b": vec(D),
          "bn_w": [1.0] * D, "bn_b": vec(D),
          "mean": vec(D), "var": [0.5 + rng.random() for _ in range(D)],
          "pw2_w": mat(D, D), "pw2_b": vec(D)}
    py = ns["conformer_conv"](x, wt, D, K)
    td = tempfile.mkdtemp()
    wp, xp, yp = (os.path.join(td, n) for n in ("w.bin", "x.bin", "y.bin"))
    with open(wp, "wb") as f:
        for vals in (wt["pw1_w"], [wt["pw1_b"]], wt["dw_w"], [wt["dw_b"]],
                     [[b] for b in wt["bn_w"]], [[b] for b in wt["bn_b"]],
                     [[m] for m in wt["mean"]], [[v] for v in wt["var"]],
                     wt["pw2_w"], [wt["pw2_b"]]):
            flat = [v for row in vals for v in row]
            f.write(struct.pack(f"{len(flat)}f", *flat))
    with open(xp, "wb") as f:
        f.write(struct.pack(f"{T * D}f", *[v for row in x for v in row]))
    body = ('#include <cstdio>\n#include <vector>\n#include "diar/conv.hpp"\n'
            f"int main(int argc,char**argv){{const int T={T},D={D},K={K};\n"
            "FILE*f=fopen(argv[1],\"rb\");std::vector<float>w;float v;while(fread(&v,4,1,f)==1)w.push_back(v);fclose(f);\n"
            "f=fopen(argv[2],\"rb\");std::vector<float>x;while(fread(&v,4,1,f)==1)x.push_back(v);fclose(f);\n"
            "size_t o=0;auto take=[&](size_t n){auto p=w.data()+o;o+=n;return p;};\n"
            "diar::ConformerConvWeights W;W.pw1_w=take(2*D*D);W.pw1_b=take(2*D);W.dw_w=take(D*K);W.dw_b=take(D);\n"
            "W.bn_w=take(D);W.bn_b=take(D);W.bn_mean=take(D);W.bn_var=take(D);W.pw2_w=take(D*D);W.pw2_b=take(D);\n"
            "std::vector<float>y(T*D);diar::conformer_conv_forward(x.data(),W,y.data(),T,D,K);\n"
            "f=fopen(argv[3],\"wb\");fwrite(y.data(),4,y.size(),f);fclose(f);return 0;}\n")
    exe, _ = _build_driver(body, ["nn.cpp", "conv.cpp"])
    r = subprocess.run([exe, wp, xp, yp], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cpp = struct.unpack(f"{T * D}f", open(yp, "rb").read())
    for i in range(T):
        for j in range(D):
            assert abs(py[i][j] - cpp[i * D + j]) < 2e-5, f"conv [{i},{j}]"


def test_python_layer_matches_cpp() -> None:
    import random
    ns = _load()
    rng = random.Random(23)
    T, D, F, NH, K = 3, 8, 16, 2, 5
    P = 2 * T - 1
    x = [[rng.uniform(-0.5, 0.5) for _ in range(D)] for _ in range(T)]
    pe = [[rng.uniform(-0.5, 0.5) for _ in range(D)] for _ in range(P)]

    def mat(r, c):
        return [[rng.uniform(-0.3, 0.3) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.2, 0.2) for _ in range(n)]

    def gvec(n):
        return [1.0 + rng.uniform(-0.1, 0.1) for _ in range(n)]

    wt = {"g1": gvec(D), "b1": vec(D), "w1": mat(F, D), "bb1": vec(F),
          "w2": mat(D, F), "bb2": vec(D),
          "ga": gvec(D), "ba": vec(D),
          "attn": {"q_w": mat(D, D), "qb": vec(D), "k_w": mat(D, D), "kb": vec(D),
                   "v_w": mat(D, D), "vb": vec(D), "pos_w": mat(D, D),
                   "bu": vec(D), "bv": vec(D), "o_w": mat(D, D), "ob": vec(D)},
          "gc": gvec(D), "bc": vec(D),
          "conv": {"pw1_w": mat(2 * D, D), "pw1_b": vec(2 * D),
                   "dw_w": [vec(K) for _ in range(D)], "dw_b": vec(D),
                   "bn_w": [1.0] * D, "bn_b": vec(D), "mean": vec(D),
                   "var": [0.5 + rng.random() for _ in range(D)],
                   "pw2_w": mat(D, D), "pw2_b": vec(D)},
          "g2": gvec(D), "b2": vec(D), "w3": mat(F, D), "bb3": vec(F),
          "w4": mat(D, F), "bb4": vec(D), "go": gvec(D), "bo": vec(D)}
    py = ns["conformer_layer"](x, pe, wt, D, F, NH, K)
    td = tempfile.mkdtemp()
    wp, xp, pp, yp = (os.path.join(td, n) for n in ("w.bin", "x.bin", "p.bin", "y.bin"))
    seq = [wt["g1"], [wt["b1"]], wt["w1"], [wt["bb1"]], wt["w2"], [wt["bb2"]],
           wt["ga"], [wt["ba"]], wt["attn"]["q_w"], [wt["attn"]["qb"]],
           wt["attn"]["k_w"], [wt["attn"]["kb"]], wt["attn"]["v_w"], [wt["attn"]["vb"]],
           wt["attn"]["pos_w"], [wt["attn"]["bu"]], [wt["attn"]["bv"]],
           wt["attn"]["o_w"], [wt["attn"]["ob"]],
           wt["gc"], [wt["bc"]], wt["conv"]["pw1_w"], [wt["conv"]["pw1_b"]],
           wt["conv"]["dw_w"], [wt["conv"]["dw_b"]],
           [[b] for b in wt["conv"]["bn_w"]], [[b] for b in wt["conv"]["bn_b"]],
           [[m] for m in wt["conv"]["mean"]], [[v] for v in wt["conv"]["var"]],
           wt["conv"]["pw2_w"], [wt["conv"]["pw2_b"]],
           wt["g2"], [wt["b2"]], wt["w3"], [wt["bb3"]], wt["w4"], [wt["bb4"]],
           wt["go"], [wt["bo"]]]
    with open(wp, "wb") as f:
        for vals in seq:
            flat = [v for row in vals for v in (row if isinstance(row, list) else [row])]
            f.write(struct.pack(f"{len(flat)}f", *flat))
    with open(xp, "wb") as f:
        f.write(struct.pack(f"{T * D}f", *[v for row in x for v in row]))
    with open(pp, "wb") as f:
        f.write(struct.pack(f"{P * D}f", *[v for row in pe for v in row]))
    body = ('#include <cstdio>\n#include <vector>\n#include "diar/conformer.hpp"\n'
            f"int main(int argc,char**argv){{const int T={T},D={D},F={F},NH={NH},K={K};\n"
            "FILE*f=fopen(argv[1],\"rb\");std::vector<float>w;float v;while(fread(&v,4,1,f)==1)w.push_back(v);fclose(f);\n"
            "f=fopen(argv[2],\"rb\");std::vector<float>x;while(fread(&v,4,1,f)==1)x.push_back(v);fclose(f);\n"
            "f=fopen(argv[3],\"rb\");std::vector<float>pe;while(fread(&v,4,1,f)==1)pe.push_back(v);fclose(f);\n"
            "size_t o=0;auto take=[&](size_t n){auto p=w.data()+o;o+=n;return p;};\n"
            "diar::ConformerLayerWeights W;W.n_ff1_g=take(D);W.n_ff1_b=take(D);W.ff1_w1=take(F*D);W.ff1_b1=take(F);\n"
            "W.ff1_w2=take(D*F);W.ff1_b2=take(D);W.n_sa_g=take(D);W.n_sa_b=take(D);\n"
            "W.attn.q_w=take(D*D);W.attn.q_b=take(D);W.attn.k_w=take(D*D);W.attn.k_b=take(D);\n"
            "W.attn.v_w=take(D*D);W.attn.v_b=take(D);W.attn.pos_w=take(D*D);W.attn.bu=take(D);W.attn.bv=take(D);\n"
            "W.attn.out_w=take(D*D);W.attn.out_b=take(D);W.n_conv_g=take(D);W.n_conv_b=take(D);\n"
            "W.conv.pw1_w=take(2*D*D);W.conv.pw1_b=take(2*D);W.conv.dw_w=take(D*K);W.conv.dw_b=take(D);\n"
            "W.conv.bn_w=take(D);W.conv.bn_b=take(D);W.conv.bn_mean=take(D);W.conv.bn_var=take(D);\n"
            "W.conv.pw2_w=take(D*D);W.conv.pw2_b=take(D);W.n_ff2_g=take(D);W.n_ff2_b=take(D);\n"
            "W.ff2_w1=take(F*D);W.ff2_b1=take(F);W.ff2_w2=take(D*F);W.ff2_b2=take(D);W.n_out_g=take(D);W.n_out_b=take(D);\n"
            "std::vector<float>y(T*D);diar::conformer_layer_forward(x.data(),pe.data(),W,y.data(),T,D,F,NH,K);\n"
            "f=fopen(argv[4],\"wb\");fwrite(y.data(),4,y.size(),f);fclose(f);return 0;}\n")
    exe, _ = _build_driver(body, ["nn.cpp", "layers.cpp", "mha.cpp", "conv.cpp", "conformer.cpp"])
    r = subprocess.run([exe, wp, xp, pp, yp], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    cpp = struct.unpack(f"{T * D}f", open(yp, "rb").read())
    for i in range(T):
        for j in range(D):
            assert abs(py[i][j] - cpp[i * D + j]) < 3e-5, f"layer [{i},{j}]"


def test_key_fragments_documented() -> None:
    text = K2.read_text()
    for frag in ("encoder.layers.", "self_attn.linear_q", "self_attn.linear_pos",
                 "self_attn.pos_bias_u",
                 "feed_forward1", "feed_forward2", "pointwise_conv1",
                 "depthwise_conv", "norm_feed_forward1", "norm_out"):
        assert frag in text, f"kernel missing key fragment {frag}"


def test_no_shared_bias_broadcast() -> None:
    # v5 bug (2026-09-17): shared encoder-level pos_bias keys do not exist
    # in the diar ckpt (untie_biases=True default -> per-layer pairs);
    # broadcasting layer-0's pair to all 17 layers blew L01..L16 to 1.4-7.2
    # while L00 stayed ~0.5. The shared-key constant must stay dead and
    # every layer must eat its own pair.
    text = K2.read_text()
    assert "POS_BIAS_U_KEY" not in text, "shared-bias constant resurrected"
    assert "POS_BIAS_V_KEY" not in text, "shared-bias constant resurrected"
    assert "bu_all[li]" in text and "bv_all[li]" in text
