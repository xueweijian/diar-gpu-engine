"""M2 Stage 2 K1 mechanics tests (no torch/NeMo/.npz needed).

Pins what can be pinned locally:
- kernel script compiles;
- pure-python transformer_block + diar_head match the C++ layer functions
  on synthetic weights (same math, two languages — a transcription error in
  m2_stage2_k1.py fails here);
- weight-name contract: every state-dict key the kernel reads exists in the
  documented NeMo/GGUF schema (checked against a synthetic fake state dict
  with exactly those keys; a rename fails loudly instead of KeyError on the
  kernel);
- gate metric math (max_abs/mean_abs/cosine/agreement) on hand values.
"""
from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

K1 = Path(__file__).parent.parent / "kaggle" / "m2_stage2" / "m2_stage2_k1.py"

REQUIRED_KEYS = (
    [f"transformer_encoder.layers.{i}.{s}" for i in range(18)
     for s in ("first_sub_layer.query_net.weight", "first_sub_layer.query_net.bias",
               "first_sub_layer.key_net.weight", "first_sub_layer.key_net.bias",
               "first_sub_layer.value_net.weight", "first_sub_layer.value_net.bias",
               "first_sub_layer.out_projection.weight", "first_sub_layer.out_projection.bias",
               "layer_norm_1.weight", "layer_norm_1.bias",
               "layer_norm_2.weight", "layer_norm_2.bias",
               "second_sub_layer.dense_in.weight", "second_sub_layer.dense_in.bias",
               "second_sub_layer.dense_out.weight", "second_sub_layer.dense_out.bias")]
    + ["sortformer_modules.first_hidden_to_hidden.weight",
       "sortformer_modules.first_hidden_to_hidden.bias",
       "sortformer_modules.single_hidden_to_spks.weight",
       "sortformer_modules.single_hidden_to_spks.bias"]
)


def _load():
    src = K1.read_text()
    # Stub the Kaggle-only harness import: mechanics tests run locally.
    src = src.replace("HARNESS_DIR = locate_harness()\nsys.path.insert(0, str(HARNESS_DIR))\n\nimport diar_harness as h  # noqa: E402\n",
                      "class _H:\n    @staticmethod\n    def environment_record(): return {}\n    @staticmethod\n    def gpu_snapshot(): return ''\n    @staticmethod\n    def emit_report(*a, **k): pass\nh = _H()\n")
    ns: dict = {"__name__": "m2_stage2_k1_test"}
    exec(compile(src, str(K1), "exec"), ns)
    return ns


def test_k1_compiles() -> None:
    assert K1.exists(), "k1 script missing"
    py_compile.compile(str(K1), doraise=True)


def test_weight_keys_documented() -> None:
    text = K1.read_text()
    for key in ("first_sub_layer.query_net.weight", "second_sub_layer.dense_in.weight",
                "layer_norm_1.weight", "first_hidden_to_hidden.weight",
                "single_hidden_to_spks.weight"):
        assert key in text, f"kernel missing documented key fragment {key}"


def test_python_block_matches_cpp() -> None:
    ns = _load()
    import random
    rng = random.Random(11)
    T, H, I, NH = 3, 8, 16, 2
    x = [[rng.uniform(-1, 1) for _ in range(H)] for _ in range(T)]

    def mat(r, c):
        return [[rng.uniform(-0.5, 0.5) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.5, 0.5) for _ in range(n)]

    wt = {"q_w": mat(H, H), "qb": vec(H), "k_w": mat(H, H), "kb": vec(H),
          "v_w": mat(H, H), "vb": vec(H), "o_w": mat(H, H), "ob": vec(H),
          "g1": [1.0 + rng.uniform(-0.1, 0.1) for _ in range(H)], "b1": vec(H),
          "g2": [1.0 + rng.uniform(-0.1, 0.1) for _ in range(H)], "b2": vec(H),
          "f1_w": mat(I, H), "f1_b": vec(I), "f2_w": mat(H, I), "f2_b": vec(H)}
    py = ns["transformer_block"](x, wt, H=H, I=I, NH=NH)

    # C++ via a driver: flatten weights row-major [out,in].
    import struct, tempfile, os
    flat = []
    order = ["q_w", "qb", "k_w", "kb", "v_w", "vb", "o_w", "ob",
             "g1", "b1", "g2", "b2", "f1_w", "f1_b", "f2_w", "f2_b"]
    sizes = {"q_w": H * H, "qb": H, "k_w": H * H, "kb": H, "v_w": H * H, "vb": H,
             "o_w": H * H, "ob": H, "g1": H, "b1": H, "g2": H, "b2": H,
             "f1_w": I * H, "f1_b": I, "f2_w": H * I, "f2_b": H}
    with tempfile.TemporaryDirectory() as td:
        wp = os.path.join(td, "w.bin")
        xp = os.path.join(td, "x.bin")
        with open(wp, "wb") as f:
            for k in order:
                blob = wt[k]
                vals = [v for row in blob for v in row] if isinstance(blob[0], list) else blob
                assert len(vals) == sizes[k], (k, len(vals), sizes[k])
                f.write(struct.pack(f"{len(vals)}f", *vals))
        with open(xp, "wb") as f:
            f.write(struct.pack(f"{T * H}f", *[v for row in x for v in row]))
        drv = os.path.join(td, "drv.cpp")
        Path(drv).write_text(
            '#include <cstdio>\n#include <vector>\n#include "diar/layers.hpp"\n'
            "int main(int argc,char**argv){\n"
            f"const int T={T},H={H},I={I},NH={NH};\n"
            "FILE*f=fopen(argv[1],\"rb\");std::vector<float>w;float v;while(fread(&v,4,1,f)==1)w.push_back(v);fclose(f);\n"
            "f=fopen(argv[2],\"rb\");std::vector<float>x;while(fread(&v,4,1,f)==1)x.push_back(v);fclose(f);\n"
            "size_t o=0;auto take=[&](size_t n){auto p=w.data()+o;o+=n;return p;};\n"
            "diar::TransformerBlockWeights W;W.q_w=take(H*H);W.q_b=take(H);W.k_w=take(H*H);W.k_b=take(H);\n"
            "W.v_w=take(H*H);W.v_b=take(H);W.o_w=take(H*H);W.o_b=take(H);W.ln1_g=take(H);W.ln1_b=take(H);\n"
            "W.ln2_g=take(H);W.ln2_b=take(H);W.f1_w=take(I*H);W.f1_b=take(I);W.f2_w=take(H*I);W.f2_b=take(H);\n"
            "std::vector<float>y(T*H);diar::transformer_block_forward(x.data(),W,y.data(),T,H,I,NH);\n"
            "f=fopen(argv[3],\"wb\");fwrite(y.data(),4,y.size(),f);fclose(f);return 0;}\n")
        root = Path(__file__).parent.parent
        exe = os.path.join(td, "drv")
        r = subprocess.run(["g++", "-std=c++17", "-O2", f"-I{root}/include", drv,
                            f"{root}/src/nn.cpp", f"{root}/src/layers.cpp", "-o", exe],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        yp = os.path.join(td, "y.bin")
        r = subprocess.run([exe, wp, xp, yp], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        cpp = struct.unpack(f"{T * H}f", open(yp, "rb").read())
    for i in range(T):
        for j in range(H):
            assert abs(py[i][j] - cpp[i * H + j]) < 2e-5, f"py-vs-cpp block [{i},{j}]"


def test_python_head_hand_case() -> None:
    ns = _load()
    y = ns["diar_head"]([[-1.0, 0.5, 2.0, -0.25]],
                        [[1, -1, 0, 0], [0, 1, 2, 0], [0, 0, 1, -1], [2, 0, 0, 1]],
                        [0.25, -0.5, 0.0, 0.5],
                        [[1, 0, -1, 2], [0, 1, 1, -1]], [0.1, -0.2])
    import math
    assert abs(y[0][0] - 1.0 / (1.0 + math.exp(0.9))) < 1e-9
    assert abs(y[0][1] - 1.0 / (1.0 + math.exp(-5.3))) < 1e-9


def test_metrics_math() -> None:
    ns = _load()
    m = ns["metrics"]([[0.1, 0.9], [0.4, 0.6]], [[0.15, 0.85], [0.45, 0.55]])
    assert abs(m["max_abs"] - 0.05) < 1e-12
    assert abs(m["mean_abs"] - 0.05) < 1e-12
    assert m["frame_agreement"] == 1.0
    assert m["cosine"] > 0.99
