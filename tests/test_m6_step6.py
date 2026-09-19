"""M3 Step 6 kernel embed round-trip test (local, no CUDA/Kaggle needed).

Pins the code_file-only discipline for kaggle/m6_step6 (same shape as
test_m5_step5) plus the Step-6 verdict logic:
- build_step6_kernel.py re-run after any C++ edit (blobs == on-disk files);
- build FILES == framework FILES dict;
- judge(): fixture tiers (K6 verbatim), determinism, route_refused,
  G-D ms/chunk gate, fp16 advisory widening, G-B2 provisional route gate.
"""
from __future__ import annotations

import ast
import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STEP6 = ROOT / "kaggle" / "m6_step6"
RUNNER = STEP6 / "m6_step6_run.py"
FRAMEWORK = STEP6 / "framework.py"
BUILD = ROOT / "tools" / "build_step6_kernel.py"


def _build_script_keys() -> list[str]:
    tree = ast.parse(BUILD.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == "FILES"):
            pairs = ast.literal_eval(node.value)
            return [key for _, key in pairs]
    raise AssertionError("FILES list missing in build script")


def _framework_keys() -> list[str]:
    text = FRAMEWORK.read_text(encoding="utf-8")
    return re.findall(r'"([a-z]+_[a-z0-9_]+)":\s+WORK', text)


def test_build_script_and_framework_file_sets_match() -> None:
    build_keys = sorted(_build_script_keys())
    fw_keys = sorted(_framework_keys())
    assert len(set(build_keys)) == len(build_keys), "duplicate keys"
    assert build_keys == fw_keys, (
        "build_step6_kernel.py FILES drifted from framework FILES dict: "
        f"{set(build_keys) ^ set(fw_keys)}")


def test_embed_fresh() -> None:
    r = subprocess.run([sys.executable, str(BUILD)], capture_output=True, text=True)
    assert r.returncode == 0, f"build failed: {r.stdout[-800:]} {r.stderr[-800:]}"
    assert "blobs verified" in r.stdout


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
    sources = {
        "cpp_bench_main": ROOT / "tools" / "diar_bench_main.cpp",
        "cpp_encoder_cuda": ROOT / "src" / "encoder_cuda.cpp",
        "cpp_backend_cuda": ROOT / "src" / "backend_cuda.cpp",
        "hpp_encoder_cuda": ROOT / "include" / "diar" / "encoder_cuda.hpp",
        "cpp_engine": ROOT / "src" / "engine.cpp",
        "cpp_sortformer": ROOT / "src" / "sortformer.cpp",
    }
    for key, path in sources.items():
        assert key in embeds, f"{key} missing from runner"
        blob = base64.b64decode(embeds[key])
        assert blob == path.read_bytes(), f"stale blob: {key} (re-run build)"
    assert len(embeds) == 37, f"expected 37 embedded files, got {len(embeds)}"


def test_framework_placeholder_and_gates_present() -> None:
    text = FRAMEWORK.read_text(encoding="utf-8")
    assert "#__EMBED_TABLE__" in text
    for gate in ("cpu_selftest", "fatbin_3sass", "fatbin_ptx60",
                 "determinism_bit_identical", "route_refused",
                 "ms_per_chunk_best", "vs_cpu_max_abs", "vs_fixture"):
        assert gate in text, f"gate {gate} missing from framework"
    # routes per mode: full-offline is CPU-only (L unbounded on GPU by design)
    assert '[("cpu", 2)] if mode == "full-offline"' in text
    # G-E artifacts land in /kaggle/working for the user-card diff
    assert "/kaggle/working/ge_t4_" in text


def _judge_from_framework(cases: dict):
    """Exec the framework's judge() in isolation (no Kaggle imports at
    module scope beyond numpy — judge itself only uses plain data)."""
    text = FRAMEWORK.read_text(encoding="utf-8")
    m = re.search(r"def judge\(cases: dict\).*?(?=\nverdict, reasons)", text, re.S)
    assert m, "judge() not found in framework"
    HARD = {"max_abs": 0.05, "mean_abs": 0.005, "frame_agreement": 0.999}
    ADV = {"max_abs": 1.0, "mean_abs": 0.02, "frame_agreement": 0.99}
    ADVISORY = {"v13-mid-streaming-r0", "v13-mid-offline-preset-r0"}
    G_D_GATE_MS = 25.0
    ns = {"HARD": HARD, "ADV": ADV, "ADVISORY": ADVISORY,
          "G_D_GATE_MS": G_D_GATE_MS}
    exec(m.group(0), ns)  # noqa: S102 — test-scope isolation
    return ns["judge"](cases)


def _bench(det=True, refused=0, mpc=6.0) -> dict:
    return {"determinism_bit_identical": det, "route_refused": refused,
            "ms_per_chunk_best": mpc}


def _vf(mx=0.01, mean=0.001, fa=0.9999) -> dict:
    return {"max_abs": mx, "mean_abs": mean, "frame_agreement": fa}


def test_judge_green_path() -> None:
    cases = {
        "v12-short-streaming-r0": {
            "mode": "streaming", "face": "postgate",
            "routes": {
                "cpu": {"bench": _bench(), "vs_fixture": _vf()},
                "fp32": {"bench": _bench(), "vs_fixture": _vf(),
                         "vs_cpu_max_abs": 0.002},
                "fp16": {"bench": _bench(), "vs_fixture": _vf()},
            }},
        "v12-mid-offline-full-r0": {
            "mode": "full-offline", "face": "pregate",
            "routes": {"cpu": {"bench": _bench(), "vs_fixture": _vf()}},
        },
    }
    verdict, reasons = _judge_from_framework(cases)
    assert verdict == "s6-green", reasons


def test_judge_catches_each_failure_class() -> None:
    base = {
        "mode": "streaming", "face": "postgate",
        "routes": {"cpu": {"bench": _bench(), "vs_fixture": _vf()},
                   "fp32": {"bench": _bench(), "vs_fixture": _vf(),
                            "vs_cpu_max_abs": 0.001},
                   "fp16": {"bench": _bench(), "vs_fixture": _vf()}},
    }
    import copy
    # determinism red
    c = copy.deepcopy(base)
    c["routes"]["fp32"]["bench"]["determinism_bit_identical"] = False
    v, r = _judge_from_framework({"v12-short-streaming-r0": c})
    assert v == "s6-red" and any("determinism" in x for x in r), r
    # route fallback is loud (route_refused > 0)
    c = copy.deepcopy(base)
    c["routes"]["fp16"]["bench"]["route_refused"] = 2
    v, r = _judge_from_framework({"v12-short-streaming-r0": c})
    assert any("route_refused" in x for x in r), r
    # G-D over budget
    c = copy.deepcopy(base)
    c["routes"]["fp32"]["bench"]["ms_per_chunk_best"] = 30.0
    v, r = _judge_from_framework({"v12-short-streaming-r0": c})
    assert any("ms/chunk" in x for x in r), r
    # fixture tier breach on the cpu route (hard v12 fixture)
    c = copy.deepcopy(base)
    c["routes"]["cpu"]["vs_fixture"] = _vf(mx=0.2, mean=0.02, fa=0.99)
    v, r = _judge_from_framework({"v12-short-streaming-r0": c})
    assert any("vs_fixture" in x for x in r), r
    # advisory fixture gets the widened tier, fp16 widened further
    c = copy.deepcopy(base)
    c["routes"]["fp16"]["vs_fixture"] = _vf(mx=0.15, mean=0.01, fa=0.995)
    v, r = _judge_from_framework({"v13-mid-streaming-r0": c})
    assert v == "s6-green", r
    # G-B2 provisional: fp32 vs cpu beyond the hard tier
    c = copy.deepcopy(base)
    c["routes"]["fp32"]["vs_cpu_max_abs"] = 0.5
    v, r = _judge_from_framework({"v12-short-streaming-r0": c})
    assert any("vs_cpu" in x for x in r), r
    # case-level error propagates
    v, r = _judge_from_framework({"x": {"error": "boom"}})
    assert v == "s6-red" and any("boom" in x for x in r)


def test_ge_cross_diff_tool_gates() -> None:
    """scripts/ge_cross_diff.py: 1e-6 fp32 gate, wire loader discipline."""
    import struct
    import numpy as np
    script = ROOT / "scripts" / "ge_cross_diff.py"
    assert script.exists()
    H = struct.Struct("<qi")
    rng = np.random.RandomState(1)
    a = rng.rand(40, 4).astype(np.float32)
    b = a + np.float32(5e-7)
    c = a.copy()
    c[7, 1] += 0.5
    paths = []
    for name, arr in (("a", a), ("b", b), ("c", c)):
        p = Path("/tmp") / f"ge_test_{name}.f32"
        p.write_bytes(H.pack(40, 4) + arr.tobytes())
        paths.append(str(p))
    r = subprocess.run([sys.executable, str(script), paths[0], paths[1]],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r2 = subprocess.run([sys.executable, str(script), paths[0], paths[2]],
                        capture_output=True, text=True)
    assert r2.returncode == 1, r2.stdout + r2.stderr
