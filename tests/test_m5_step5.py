"""M3 Step 5 kernel embed round-trip test (local, no CUDA/Kaggle needed).

Pins the code_file-only discipline for kaggle/m5_step5: the runner ships as
ONE file (m5_step5_run.py, uploads carry only code_file), so all 23 C++
sources travel base64-embedded. Fails when:
- build_step5_kernel.py was not re-run after editing a C++ source
  (embedded bytes != on-disk files);
- the build script's file list drifts from the framework's FILES dict;
- the framework lost the placeholder or the runner is not ast-parseable.
"""
from __future__ import annotations

import ast
import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STEP4 = ROOT / "kaggle" / "m5_step5"
RUNNER = STEP4 / "m5_step5_run.py"
FRAMEWORK = STEP4 / "framework.py"
BUILD = ROOT / "tools" / "build_step5_kernel.py"


def _build_script_keys() -> list[str]:
    tree = ast.parse(BUILD.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == "FILES"):
            pairs = ast.literal_eval(node.value)
            return [key for _, key in pairs]
    raise AssertionError("FILES list missing in build script")


def _framework_keys() -> list[str]:
    # keys of the FILES dict literal in framework.py
    text = FRAMEWORK.read_text(encoding="utf-8")
    return re.findall(r'"([a-z]+_[a-z0-9_]+)":\s*WORK', text)


def test_build_script_and_framework_file_sets_match() -> None:
    build_keys = sorted(_build_script_keys())
    fw_keys = sorted(_framework_keys())
    assert len(set(build_keys)) == len(build_keys), "duplicate keys in build FILES"
    assert build_keys == fw_keys, (
        "build_step5_kernel.py FILES drifted from framework FILES dict: "
        f"{set(build_keys) ^ set(fw_keys)}")


def test_embed_fresh() -> None:
    # rebuild in-memory: run the builder, then verify every blob round-trips
    r = subprocess.run([sys.executable, str(BUILD)], capture_output=True, text=True)
    assert r.returncode == 0, f"build failed: {r.stdout[-800:]} {r.stderr[-800:]}"
    assert "blobs fresh" in r.stdout


def test_runner_blobs_match_sources() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    embeds: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            if (isinstance(target, ast.Subscript)
                    and getattr(target.value, "id", "") == "EMBED"):
                key = ast.literal_eval(target.slice)
                embeds[key] = ast.literal_eval(node.value)
    assert embeds, "no EMBED assignments in runner"
    # the build script ran in test_embed_fresh with cwd-independent ROOT;
    # spot-verify the critical artifacts against the on-disk bytes
    sources = {
        "cpp_bench_main": ROOT / "tools" / "step5_bench_main.cpp",
        "cpp_backend_cuda": ROOT / "src" / "backend_cuda.cpp",
        "hpp_backend_cuda": ROOT / "include" / "diar" / "backend_cuda.hpp",
        "cpp_conformer": ROOT / "src" / "conformer.cpp",
    }
    for key, path in sources.items():
        assert key in embeds, f"{key} missing from runner"
        blob = base64.b64decode(embeds[key])
        assert blob == path.read_bytes(), f"stale blob: {key} (re-run build)"
    assert len(embeds) == 23, f"expected 23 embedded files, got {len(embeds)}"


def test_framework_placeholder_and_gates_present() -> None:
    text = FRAMEWORK.read_text(encoding="utf-8")
    assert "#__EMBED_TABLE__" in text
    for gate in ("g_s4a_softmax", "g_s4a_glu", "g_s4a_dwconv", "g_s4b_mha",
                 "g_s4c_layer", "g_s4e_tf", "g_s4d_jit_bitidentical",
                 "g_s5a_cast", "g_s5b_linear_h", "g_s5b_linear_hq",
                 "g_s5c_layer_h", "g_s5d_tf_h", "g_s5e_gemm_p50",
                 "g_s5f_fp16_not_slower", "g_s5jit_linear_h",
                 "fatbin_sass_all3", "fatbin_ptx_compute60"):
        assert gate in text, f"gate {gate} missing from framework"
    # the parity-only rerun (PTX-JIT path) must be wired
    assert "--parity-only" in text


def test_scratch_formulas_carry_cast16_regions() -> None:
    """The three scratch formulas must reserve the fp16-route cast16 region
    ((max_elems + 1) // 2 floats) or the fp16 layer paths write OOB."""
    src = (ROOT / "src" / "backend_cuda.cpp").read_text(encoding="utf-8")
    assert "((static_cast<std::size_t>(p) * c + 1) / 2)" in src, "mha cast16"
    assert "((static_cast<std::size_t>(t) * inner + 1) / 2)" in src, "tf cast16"
    assert "((static_cast<std::size_t>(t) * d_ff + 1) / 2)" in src, "conf cast16"
    # mirror: python evaluation of the three formulas at engine geometry
    def mha(t, c, h):
        p = 2 * t - 1
        return 6 * t * c + p * c + h * t * (2 * t + p) + (p * c + 1) // 2
    def tf(t, h, inner, heads):
        return 7 * t * h + t * inner + 2 * heads * t * t + (t * inner + 1) // 2
    def conf(t, c, d_ff, h):
        return 7 * t * c + t * d_ff + mha(t, c, h) + (t * d_ff + 1) // 2
    assert mha(20, 512, 8) == 6*20*512 + 39*512 + 8*20*(40+39) + (39*512+1)//2
    assert tf(20, 192, 768, 8) == 7*20*192 + 20*768 + 2*8*400 + (20*768+1)//2
    assert conf(20, 512, 2048, 8) > mha(20, 512, 8)
