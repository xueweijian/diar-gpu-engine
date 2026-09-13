"""Benchmark NVIDIA's pure diarization runtime (Sortformer v2) on a Tesla P100.

Design notes
------------
* The build tree, model weights and converted audio live under ``/tmp`` so the
  Kaggle output directory only carries small, reviewable reports.
* Every fixture is diarized several times with fresh output paths; wall time is
  also fitted against audio length so the fixed cost (model load, process start)
  can be separated from the marginal real-time factor.
* Scope guard: pure speaker diarization only - no ASR model, no transcription.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
import wave
from pathlib import Path

TMP = Path("/tmp/diar-work")
REPO = TMP / "NeMo-Speech.cpp"
MODEL = TMP / "diar_streaming_sortformer_4spk-v2.q8_0.gguf"
OUT = Path("/kaggle/working")
LOG = TMP / "run.log"
REPORT = OUT / "sortformer_p100_report.json"
MODEL_URL = (
    "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/"
    "resolve/main/diar_streaming_sortformer_4spk-v2.q8_0.gguf?download=true"
)
RUNS_PER_FIXTURE = 3

# Kaggle GPU images expose the driver runtime but not a libcuda.so that
# FindCUDAToolkit recognises, so ggml-cuda's CUDA::cuda_driver target is never
# created. This shim defines that imported target deterministically.
DRIVER_SHIM = r'''
if(NOT TARGET CUDA::cuda_driver)
  set(_diar_candidates
      "$ENV{CUDA_HOME}/lib64/stubs/libcuda.so"
      "/usr/local/cuda/lib64/stubs/libcuda.so"
      "/usr/local/nvidia/lib64/libcuda.so"
      "/usr/lib/x86_64-linux-gnu/libcuda.so"
      "/usr/lib64/libcuda.so")
  set(_diar_driver "")
  foreach(_candidate IN LISTS _diar_candidates)
    if(EXISTS "${_candidate}")
      set(_diar_driver "${_candidate}")
      break()
    endif()
  endforeach()
  if(NOT _diar_driver)
    find_library(_diar_found NAMES cuda libcuda.so.1 libcuda)
    if(_diar_found)
      set(_diar_driver "${_diar_found}")
    endif()
  endif()
  if(NOT _diar_driver)
    message(FATAL_ERROR "[diar-shim] no libcuda driver library found")
  endif()
  add_library(CUDA::cuda_driver UNKNOWN IMPORTED)
  set_target_properties(CUDA::cuda_driver PROPERTIES IMPORTED_LOCATION "${_diar_driver}")
  message(STATUS "[diar-shim] CUDA::cuda_driver -> ${_diar_driver}")
endif()
'''

RESULT: dict[str, object] = {
    "schema_version": 2,
    "scope": "pure_speaker_diarization",
    "job": "sortformer_v2_p100",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "platform": platform.platform(),
}


def run(argv: list[str], cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str, float]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd) if cwd else None, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
        code, text = proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        code, text = 124, f"timeout after {timeout}s: {exc}"
    elapsed = time.perf_counter() - started
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(argv)}\n{text}\n")
    return code, text, elapsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> str:
    _, text, _ = run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ], timeout=60)
    return text.strip()


def wav_info(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        return {
            "channels": handle.getnchannels(),
            "sample_rate": rate,
            "sample_width": handle.getsampwidth(),
            "frames": handle.getnframes(),
            "seconds": handle.getnframes() / float(rate),
        }


def to_wav16k(source: Path, destination: Path) -> tuple[bool, str]:
    code, text, _ = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ], timeout=900)
    return code == 0 and destination.exists(), text.strip()


def make_wav(path: Path, seconds: int = 60, rate: int = 16000) -> None:
    """Deterministic tone fixture: a speed fixture, not an accuracy corpus."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        frames = bytearray()
        for index in range(seconds * rate):
            t = index / rate
            base = (180.0, 235.0, 305.0, 410.0)[int(t // 5) % 4]
            sample = 0.18 * math.sin(2.0 * math.pi * base * t) * (1.0 + 0.25 * math.sin(2.0 * math.pi * 3.0 * t))
            sample += 0.06 * math.sin(2.0 * math.pi * (base * 1.7) * t)
            value = max(-1.0, min(1.0, sample))
            frames.extend(int(value * 32767.0).to_bytes(2, "little", signed=True))
        out.writeframes(frames)


def parse_rttm(path: Path) -> dict[str, object]:
    speakers: dict[str, float] = {}
    lines = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith(";;"):
                continue
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                continue
            lines += 1
            speakers[fields[7]] = speakers.get(fields[7], 0.0) + float(fields[4])
    return {
        "segments": lines,
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "speech_seconds": {k: round(v, 3) for k, v in sorted(speakers.items())},
    }


def find_binary() -> Path:
    candidates = [p for p in REPO.rglob("nemo-speech") if p.is_file() and os.access(p, os.X_OK)]
    if not candidates:
        raise RuntimeError("nemo-speech binary not found after build")
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def prepare_driver() -> Path:
    notes: dict[str, object] = {}
    _, ldconfig_text, _ = run(["bash", "-lc", "ldconfig -p | grep -i libcuda || true"], timeout=60)
    notes["ldconfig"] = ldconfig_text.strip()
    shim_path = TMP / "diar_cuda_driver_shim.cmake"
    shim_path.write_text(DRIVER_SHIM, encoding="utf-8")
    notes["shim"] = str(shim_path)
    RESULT["cuda_driver_preparation"] = notes
    return shim_path


def build_repo() -> Path:
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        raise RuntimeError("cmake/ninja missing from the Kaggle image")
    if subprocess.run(["pkg-config", "--exists", "sentencepiece"], check=False).returncode != 0:
        code, _, _ = run([
            "bash", "-lc", "apt-get update -qq && apt-get install -y -qq libsentencepiece-dev",
        ], timeout=900)
        if code != 0:
            raise RuntimeError("could not install sentencepiece development files")

    code, _, _ = run([
        "git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules",
        "https://github.com/NVIDIA/NeMo-Speech.cpp.git", str(REPO),
    ], timeout=1800)
    if code != 0:
        raise RuntimeError("NeMo-Speech.cpp clone failed")
    RESULT["runtime_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()

    shim = prepare_driver()
    code, configure_text, configure_seconds = run([
        "bash", "scripts/configure.sh", "cuda-diar",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=60",
        f"-DCMAKE_PROJECT_INCLUDE_BEFORE={shim}",
    ], cwd=REPO, timeout=1200)
    RESULT["configure_seconds"] = configure_seconds
    if code != 0:
        raise RuntimeError(f"configure failed (rc={code}): {configure_text[-2000:]}")
    code, build_text, build_seconds = run([
        "cmake", "--build", "--preset", "cuda-diar", "--parallel", "2",
    ], cwd=REPO, timeout=5400)
    RESULT["build_seconds"] = build_seconds
    if code != 0:
        raise RuntimeError(f"build failed (rc={code}): {build_text[-2000:]}")
    return find_binary()


def diarize(binary: Path, audio: Path, label: str) -> dict[str, object]:
    _, help_text, _ = run([str(binary), "diarize", "--help"], cwd=REPO, timeout=120)
    runs: list[dict[str, object]] = []
    for iteration in range(RUNS_PER_FIXTURE):
        rttm = TMP / f"{label}.run{iteration}.rttm"
        rttm.unlink(missing_ok=True)
        before = gpu_snapshot()
        code, text, elapsed = run([
            str(binary), "diarize", str(audio),
            "--diar-model", str(MODEL),
            "--device", "cuda:0",
            "--format", "rttm", "--output", str(rttm),
        ], cwd=REPO, timeout=1800)
        runs.append({
            "iteration": iteration, "returncode": code, "seconds": elapsed,
            "gpu_before": before, "rttm": str(rttm),
            "tail": text[-800:],
        })
    ok = [r for r in runs if r["returncode"] == 0]
    times = [float(r["seconds"]) for r in ok]
    info = wav_info(audio)
    seconds = float(info["seconds"])
    return {
        "label": label,
        "audio": str(audio),
        "audio_seconds": seconds,
        "audio_info": info,
        "runs": runs,
        "rttm_sha256": sha256(TMP / f"{label}.run0.rttm") if (TMP / f"{label}.run0.rttm").exists() else None,
        "rttm_first_run": parse_rttm(TMP / f"{label}.run0.rttm"),
        "cli_help": help_text.strip()[:1200],
        "timing": {
            "iterations_ok": len(ok),
            "seconds": times,
            "wall_seconds_median": statistics.median(times) if times else None,
            "wall_seconds_min": min(times) if times else None,
            "rtf_median": (statistics.median(times) / seconds) if times else None,
            "realtime_x_median": (seconds / statistics.median(times)) if times else None,
            "note": "CLI wall time includes process start, CUDA init and model load.",
        },
        "status": "pass" if len(ok) == RUNS_PER_FIXTURE else "fail",
    }


def fit_runtime(entries: list[dict[str, object]]) -> dict[str, object]:
    """Least-squares fit wall_time = fixed + rtf * audio_seconds."""
    points = [
        (float(e["audio_seconds"]), float(e["timing"]["wall_seconds_median"]))
        for e in entries if e.get("timing", {}).get("wall_seconds_median")
    ]
    if len(points) < 2:
        return {"points": points, "note": "need >=2 distinct fixture lengths"}
    n = float(len(points))
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xx = sum(x * x for x, _ in points)
    sum_xy = sum(x * y for x, y in points)
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-9:
        return {"points": points, "note": "degenerate fit"}
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return {
        "points": [{"audio_seconds": x, "wall_seconds": y} for x, y in points],
        "fixed_seconds": intercept,
        "marginal_rtf": slope,
        "marginal_realtime_x": (1.0 / slope) if slope > 0 else None,
        "note": "fixed = model load + CUDA init; marginal rtf = compute per audio second",
    }


def collect_fixtures() -> list[tuple[str, Path]]:
    fixtures: list[tuple[str, Path]] = []
    conversions: list[dict[str, object]] = []
    for candidate in sorted(Path("/kaggle/input").glob("diar-*/**/*.wav")):
        fixtures.append((re.sub(r"[^A-Za-z0-9_.-]", "_", candidate.stem)[:40], candidate))
    for candidate in sorted(Path("/kaggle/input").glob("diar-*/**/*.mp3")):
        target = TMP / (re.sub(r"[^A-Za-z0-9_.-]", "_", candidate.stem)[:40] + ".wav")
        ok, message = to_wav16k(candidate, target)
        conversions.append({"source": str(candidate), "target": str(target), "ok": ok, "message": message})
        if ok:
            fixtures.append((target.stem, target))
    make_wav(TMP / "synthetic_tone_60s.wav")
    fixtures.append(("synthetic_tone_60s", TMP / "synthetic_tone_60s.wav"))
    RESULT["audio_conversions"] = conversions
    return fixtures


def _main() -> int:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    RESULT["gpu_before"] = gpu_snapshot()

    binary = build_repo()
    RESULT["binary"] = str(binary)
    _, version_text, _ = run([str(binary), "--version"], cwd=REPO, timeout=120)
    RESULT["runtime_version_output"] = version_text.strip()[:400]

    code, _, download_seconds = run([
        "curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error", "-o", str(MODEL), MODEL_URL,
    ], timeout=1800)
    if code != 0 or not MODEL.exists():
        raise RuntimeError("Sortformer GGUF download failed")
    RESULT["model_bytes"] = MODEL.stat().st_size
    RESULT["model_sha256"] = sha256(MODEL)
    RESULT["download_seconds"] = download_seconds

    entries = [diarize(binary, audio, label) for label, audio in collect_fixtures()]
    RESULT["fixtures"] = entries
    RESULT["runtime_fit"] = fit_runtime(entries)
    RESULT["gpu_after"] = gpu_snapshot()
    RESULT["status"] = "pass" if entries and all(e["status"] == "pass" for e in entries) else "partial"
    RESULT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return 0


def _emit(status: str) -> None:
    RESULT["status"] = status
    RESULT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = json.dumps(RESULT, sort_keys=True, indent=2, ensure_ascii=False)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(payload, encoding="utf-8")
    print(payload)
    if LOG.exists():
        tail = LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
        (OUT / "run_tail.log").write_text(tail, encoding="utf-8")
        print("\n--- runtime log tail ---\n", tail[-2000:])


def main() -> int:
    exit_code = 0
    try:
        _main()
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive on Kaggle
        RESULT["error_type"] = type(exc).__name__
        RESULT["error"] = str(exc)
        exit_code = 1
    _emit(str(RESULT.get("status", "partial")) if exit_code == 0 else "error")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
