"""P100 compatibility probe for the pure diarization engine.

This job deliberately downloads no model. It tests both a native CUDA sm_60
kernel and a Pascal-compatible PyTorch wheel, then emits one compact JSON
report. It is safe to run as a short Kaggle experiment.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/kaggle/working/diar_gpu_engine_p100_compat")


def command(argv: list[str], timeout: int = 600) -> dict[str, object]:
    try:
        completed = subprocess.run(
            argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
        return {"argv": argv, "returncode": completed.returncode, "output": completed.stdout[-8000:]}
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "returncode": -1, "output": repr(exc)}


def main() -> int:
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "pure_speaker_diarization",
        "job": "p100_compat",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_query": command([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ], timeout=30),
    }

    # Native path: this is the decisive test for our future C++ engine.
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    source = Path("cuda_smoke.cu")
    native: dict[str, object]
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        native = {"status": "skipped", "reason": "nvcc not found"}
    else:
        compile_result = command([
            nvcc, "-O3", "-std=c++14", "-arch=sm_60", str(source),
            "-o", str(ROOT / "cuda_smoke"),
        ], timeout=600)
        run_result = command([str(ROOT / "cuda_smoke")], timeout=60) if compile_result["returncode"] == 0 else None
        native = {"nvcc": nvcc, "compile": compile_result, "run": run_result}
    result["native_cuda_sm60"] = native

    # Framework path: use a cu126 wheel only after removing the incompatible
    # default cu128 wheel. No model, audio, or ASR package is installed.
    install = command([
        sys.executable, "-m", "pip", "uninstall", "-y", "torch",
    ], timeout=300)
    install2 = command([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu126",
    ], timeout=900)
    framework: dict[str, object] = {"uninstall": install, "install": install2}
    try:
        import torch  # type: ignore

        framework["torch_version"] = torch.__version__
        framework["torch_cuda"] = torch.version.cuda
        framework["arch_list"] = torch.cuda.get_arch_list()
        framework["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            framework["device"] = torch.cuda.get_device_name(0)
            framework["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            x = torch.arange(1024, device="cuda", dtype=torch.float32)
            y = (x * 1.25 + 2.0).sum()
            framework["cuda_smoke_sum"] = float(y.cpu())
            framework["status"] = "pass"
        else:
            framework["status"] = "fail"
    except Exception as exc:  # noqa: BLE001
        framework["status"] = "fail"
        framework["error"] = repr(exc)
    result["torch_cu126"] = framework

    print(json.dumps(result, sort_keys=True, indent=2))
    # The report itself is the artifact; compatibility failures should not hide
    # the environment facts from the next engineering iteration.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
