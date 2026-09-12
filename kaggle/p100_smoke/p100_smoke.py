"""Pure diarization P100 environment smoke test.

This intentionally does not download an ASR or diarization model. It proves
that the remote GPU is visible, records the environment, and performs only a
small CUDA tensor operation. A later model benchmark will replace the final
section with the pinned pure-diarization runtime.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001 - environment diagnostics must survive failures
        return f"ERROR: {exc}"


def main() -> int:
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "pure_speaker_diarization",
        "job": "p100_smoke",
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_query": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch  # type: ignore

        result["torch_version"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
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
    # The environment report is useful even when a default wheel lacks sm_60.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
