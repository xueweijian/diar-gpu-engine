"""P100-compatible PyTorch check: uninstall cu128 wheels, install cu126 wheels.

Use with Kaggle GPU P100 because Kaggle's default torch 2.10.0+cu128 drops
sm_60 kernels. This script reports the resulting environment and repeats the
same tiny pure-diarization-free CUDA smoke operation, without downloading any
model or ASR dependency.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def main() -> int:
    uninstall = run([
        sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"
    ])
    install = run([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "torch==2.10.0", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    ])
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "pure_speaker_diarization",
        "job": "p100_torch126",
        "python": sys.version,
        "platform": platform.platform(),
        "uninstall": uninstall[-2000:],
        "install": install[-2000:],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch  # type: ignore

        result["torch_version"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        result["arch_list"] = torch.cuda.get_arch_list()
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["device"] = torch.cuda.get_device_name(0)
            result["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            x = torch.arange(1024, device="cuda", dtype=torch.float32)
            y = (x * 1.25 + 2.0).sum()
            result["cuda_smoke_sum"] = float(y.cpu())
    except Exception as exc:  # noqa: BLE001
        result["torch_error"] = repr(exc)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
