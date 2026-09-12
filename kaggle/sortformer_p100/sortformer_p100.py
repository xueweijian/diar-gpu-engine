"""Build NVIDIA's pure diarization C++ runtime and benchmark Sortformer v2 on P100.

The Kaggle runtime is disposable: model weights and source dependencies never
land on the phone or in Git. The output is a compact JSON report plus logs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working/diar_gpu_engine_sortformer_p100")
REPO = WORK / "NeMo-Speech.cpp"
MODEL = WORK / "diar_streaming_sortformer_4spk-v2.q8_0.gguf"
WAV = WORK / "synthetic_meeting_60s.wav"
RTTM = WORK / "synthetic_meeting_60s.rttm"
LOG = WORK / "run.log"
MODEL_URL = (
    "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/"
    "resolve/main/diar_streaming_sortformer_4spk-v2.q8_0.gguf?download=true"
)


def run(argv: list[str], cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str, float]:
    started = time.perf_counter()
    proc = subprocess.run(
        argv, cwd=str(cwd) if cwd else None, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    text = proc.stdout or ""
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(argv)}\n{text}\n")
    return proc.returncode, text, elapsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_wav(path: Path, seconds: int = 60, rate: int = 16000) -> None:
    # Deterministic, multi-speaker-like tonal fixture. This is a speed fixture,
    # not an accuracy corpus; no ASR or text is involved.
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        frames = bytearray()
        for i in range(seconds * rate):
            t = i / rate
            slot = int(t // 5) % 4
            base = (180.0, 235.0, 305.0, 410.0)[slot]
            modulation = 1.0 + 0.25 * math.sin(2.0 * math.pi * 3.0 * t)
            sample = 0.18 * math.sin(2.0 * math.pi * base * t) * modulation
            sample += 0.06 * math.sin(2.0 * math.pi * (base * 1.7) * t)
            value = max(-1.0, min(1.0, sample))
            integer = int(value * 32767.0)
            frames.extend(integer.to_bytes(2, "little", signed=True))
        out.writeframes(frames)


def find_binary() -> Path:
    candidates = [p for p in REPO.rglob("nemo-speech") if p.is_file() and os.access(p, os.X_OK)]
    if not candidates:
        raise RuntimeError("nemo-speech binary not found after build")
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def _main() -> int:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "pure_speaker_diarization",
        "job": "sortformer_v2_p100",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
    }

    # Record hardware before build/model work.
    _, gpu_text, _ = run([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ], timeout=30)
    result["gpu"] = gpu_text.strip()

    # Kaggle images normally include the toolchain. Install only the small
    # missing system package if necessary; never install an ASR framework.
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        raise RuntimeError("cmake/ninja missing from Kaggle image")
    if subprocess.run(["pkg-config", "--exists", "sentencepiece"], check=False).returncode != 0:
        code, _, _ = run(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq libsentencepiece-dev"], timeout=900)
        if code != 0:
            raise RuntimeError("could not install sentencepiece development files")

    code, output, build_clone_time = run([
        "git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules",
        "https://github.com/NVIDIA/NeMo-Speech.cpp.git", str(REPO),
    ], timeout=1800)
    if code != 0:
        raise RuntimeError("NeMo-Speech.cpp clone failed")
    result["runtime_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()

    code, _, configure_time = run([
        "bash", "scripts/configure.sh", "cuda-diar",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=60",
    ], cwd=REPO, timeout=1200)
    if code != 0:
        raise RuntimeError("NeMo-Speech.cpp CUDA diarization configure failed")
    code, _, build_time = run([
        "cmake", "--build", "--preset", "cuda-diar", "--parallel", "2",
    ], cwd=REPO, timeout=3600)
    if code != 0:
        raise RuntimeError("NeMo-Speech.cpp CUDA diarization build failed")
    binary = find_binary()
    result["binary"] = str(binary)
    result["build_seconds"] = build_time

    code, _, download_time = run([
        "curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
        "-o", str(MODEL), MODEL_URL,
    ], timeout=1800)
    if code != 0 or not MODEL.exists():
        raise RuntimeError("Sortformer GGUF download failed")
    result["model_bytes"] = MODEL.stat().st_size
    result["model_sha256"] = sha256(MODEL)
    result["download_seconds"] = download_time

    make_wav(WAV)
    result["fixture"] = {"path": str(WAV), "seconds": 60, "sample_rate": 16000}

    # Record runtime diagnostics, then run the standalone diarization CLI.
    run([str(binary), "doctor", "--json"], cwd=REPO, timeout=180)
    command = [
        str(binary), "diarize", str(WAV),
        "--diar-model", str(MODEL),
        "--device", "cuda:0",
        "--format", "rttm", "--output", str(RTTM),
    ]
    timings = []
    outputs = []
    for iteration in range(3):
        code, text, elapsed = run(command, cwd=REPO, timeout=900)
        timings.append(elapsed)
        outputs.append({"iteration": iteration, "returncode": code, "tail": text[-2000:]})
        if code != 0:
            break
    result["runs"] = outputs
    result["timing"] = {
        "iterations": len(timings),
        "seconds": timings,
        "audio_seconds": 60.0,
        "wall_seconds_last": timings[-1] if timings else None,
        "rtf_last": timings[-1] / 60.0 if timings else None,
        "realtime_x_last": 60.0 / timings[-1] if timings and timings[-1] > 0 else None,
        "note": "CLI wall time includes model load; this is an end-to-end runtime smoke, not neural-core-only timing.",
    }
    result["rttm_bytes"] = RTTM.stat().st_size if RTTM.exists() else None
    result["status"] = "pass" if timings and outputs[-1]["returncode"] == 0 else "fail"
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "pass" else 1


def main() -> int:
    try:
        return _main()
    except Exception as exc:  # noqa: BLE001 - preserve diagnostics in Kaggle output
        report = {
            "schema_version": 1,
            "scope": "pure_speaker_diarization",
            "job": "sortformer_v2_p100",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(json.dumps(report, sort_keys=True, indent=2))
        print("\n--- last runtime log ---\n", LOG.read_text(encoding="utf-8", errors="replace")[-12000:] if LOG.exists() else "<none>")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
