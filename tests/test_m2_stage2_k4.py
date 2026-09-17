"""M2 Stage 2 K4 probe test (local, no torch/NeMo/.npz needed).

Pins the probe contract that a kernel round-trip cannot cheaply re-verify:
- K4 compiles and carries its job marker (embed guard needs it);
- the K4-local MHA/conv/FF/LN helpers agree with the K2 gate helpers on
  synthetic weights (same transcription, no fork);
- every probe stage P0..P5 is present in main() (a deleted stage would
  silently narrow the probe);
- embed round-trip covers K4 (anchor registered + marker guarded).
"""
from __future__ import annotations

import py_compile
from pathlib import Path

K4 = Path(__file__).parent.parent / "kaggle" / "m2_stage2" / "m2_stage2_k4.py"
K2 = Path(__file__).parent.parent / "kaggle" / "m2_stage2" / "m2_stage2_k2.py"


def _load(path: Path, name: str) -> dict:
    src = path.read_text()
    src = src.replace("HARNESS_DIR = locate_harness()\nsys.path.insert(0, str(HARNESS_DIR))\n\nimport diar_harness as h  # noqa: E402\n",
                      "class _H:\n    @staticmethod\n    def environment_record(): return {}\n    @staticmethod\n    def gpu_snapshot(): return ''\n    @staticmethod\n    def emit_report(*a, **k): pass\nh = _H()\n")
    ns: dict = {"__name__": name}
    exec(compile(src, str(path), "exec"), ns)
    return ns


def test_k4_compiles_and_marked() -> None:
    assert K4.exists(), "k4 script missing"
    py_compile.compile(str(K4), doraise=True)
    text = K4.read_text()
    assert "m2_stage2_k4_probe" in text
    assert "teacher-forced" in text
    assert '"k4-measured"' in text or "'k4-measured'" in text


def test_k4_helpers_match_k2() -> None:
    import random
    k4, k2 = _load(K4, "k4test"), _load(K2, "k2test")
    rng = random.Random(11)
    T, C, NH, D, K = 3, 8, 2, 8, 5
    x = [[rng.uniform(-1, 1) for _ in range(C)] for _ in range(T)]
    assert k4["relpos_table"](T, C) == k2["relpos_table"](T, C)

    def mat(r, c):
        return [[rng.uniform(-0.3, 0.3) for _ in range(c)] for _ in range(r)]

    def vec(n):
        return [rng.uniform(-0.3, 0.3) for _ in range(n)]

    wt = {"q_w": mat(C, C), "qb": vec(C), "k_w": mat(C, C), "kb": vec(C),
          "v_w": mat(C, C), "vb": vec(C), "pos_w": mat(C, C),
          "bu": vec(C), "bv": vec(C), "o_w": mat(C, C), "ob": vec(C)}
    pe = k4["relpos_table"](T, C)
    a, b = k4["relpos_mha"](x, pe, wt, C, NH), k2["relpos_mha"](x, pe, wt, C, NH)
    for i in range(T):
        for j in range(C):
            assert abs(a[i][j] - b[i][j]) < 1e-9, f"mha fork [{i},{j}]"
    w1, bb1, w2, bb2 = mat(16, D), vec(16), mat(D, 16), vec(D)
    xd = [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
    a, b = k4["conformer_ff"](xd, w1, bb1, w2, bb2), k2["conformer_ff"](xd, w1, bb1, w2, bb2)
    for i in range(T):
        for j in range(D):
            assert abs(a[i][j] - b[i][j]) < 1e-9, f"ff fork [{i},{j}]"
    cw = {"pw1_w": mat(2 * D, D), "pw1_b": vec(2 * D),
          "dw_w": [vec(K) for _ in range(D)], "dw_b": vec(D),
          "bn_w": [1.0] * D, "bn_b": vec(D), "mean": vec(D),
          "var": [0.5 + rng.random() for _ in range(D)],
          "pw2_w": mat(D, D), "pw2_b": vec(D)}
    a, b = k4["conformer_conv"](xd, cw, D, K), k2["conformer_conv"](xd, cw, D, K)
    for i in range(T):
        for j in range(D):
            assert abs(a[i][j] - b[i][j]) < 1e-9, f"conv fork [{i},{j}]"


def test_k4_stages_present() -> None:
    text = K4.read_text()
    for stage in ("P0_xscaled_input", "P1_pos_table", "P2_norm_ff1",
                  "P3a_mha_full_local", "P3b_mha_nemo_qkvp_local_formula",
                  "P4_conv_on_nemo_input", "P5_full_layer_local", "P5_nemo_vs_dump",
                  "probe_store_rows", "pos_hook"):
        assert stage in text, f"K4 lost stage {stage}"


def test_k4_embed_registered() -> None:
    emb = (Path(__file__).parent.parent / "kaggle" / "m2_stage2" / "embed_stage2.py").read_text()
    assert "m2_stage2_k4.py" in emb
    assert "EMBEDDED_M2_STAGE2_K4" in emb
    assert "m2_stage2_k4_probe" in emb
