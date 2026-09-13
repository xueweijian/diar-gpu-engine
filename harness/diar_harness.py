"""Shared benchmark harness for disposable Kaggle diarization runs.

This module is published as the Kaggle dataset ``diar-gpu-engine-harness`` so
each kernel script only describes *what* to measure. Everything here is
side-effect free at import time.

Scope guard: pure speaker diarization only. No ASR model, tokenizer,
Parakeet, Whisper or Nemotron dependency is referenced anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import threading
import time
import wave
from pathlib import Path

WORK_ROOT = Path(os.environ.get("DIAR_WORK_ROOT", "/tmp/diar-work"))
REPO_DIR = WORK_ROOT / "NeMo-Speech.cpp"
MODEL_PATH = WORK_ROOT / "diar_streaming_sortformer_4spk-v2.q8_0.gguf"
LOG_PATH = WORK_ROOT / "run.log"
OUT_DIR = Path(os.environ.get("DIAR_OUT_DIR", "/kaggle/working"))
MODEL_URL = (
    "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/"
    "resolve/main/diar_streaming_sortformer_4spk-v2.q8_0.gguf?download=true"
)
REPO_URL = "https://github.com/NVIDIA/NeMo-Speech.cpp.git"

# Kaggle GPU images expose the driver runtime but not a libcuda.so that CMake's
# FindCUDAToolkit recognises, so ggml-cuda's CUDA::cuda_driver imported target
# is never created and configure aborts. This shim defines it deterministically.
CUDA_DRIVER_SHIM = r'''
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


# --------------------------------------------------------------------------
# process helpers
# --------------------------------------------------------------------------
def reset_workspace() -> None:
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")


def run(argv: list[str], cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str, float]:
    """Run a command, append its output to the runtime log, return (rc, text, seconds)."""
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd) if cwd else None, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
        code, text = proc.returncode, proc.stdout or ""
    except FileNotFoundError as exc:
        code, text = 127, f"missing executable: {exc}"
    except subprocess.TimeoutExpired as exc:
        code, text = 124, f"timeout after {timeout}s: {exc}"
    elapsed = time.perf_counter() - started
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(argv)}\n{text}\n")
    return code, text, elapsed


def child_cpu_seconds() -> float:
    """Cumulative CPU time of all reaped children (user + system)."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slugify(name: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:limit]


AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def discover_audio(root: Path | str = "/kaggle/input") -> list[Path]:
    """Find every audio file under a mount root, layout-independent.

    Kaggle has used both ``/kaggle/input/<slug>/`` and the nested
    ``/kaggle/input/datasets/<owner>/<slug>/`` layout, so never assume a
    prefix: search recursively and sort for deterministic ordering.
    """
    base = Path(root)
    if not base.exists():
        return []
    found: list[Path] = []
    for suffix in AUDIO_SUFFIXES:
        found.extend(p for p in base.rglob(f"*{suffix}") if p.is_file())
    return sorted(set(found))


def input_layout(root: Path | str = "/kaggle/input", depth: int = 3) -> list[str]:
    """Compact snapshot of the mount tree, for diagnosing missing datasets."""
    base = Path(root)
    if not base.exists():
        return [f"MISSING {base}"]
    lines: list[str] = []

    def walk(directory: Path, level: int) -> None:
        if level > depth:
            return
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            lines.append(f"{'  ' * level}<unreadable {directory}: {exc}>")
            return
        for entry in entries:
            if entry.is_dir():
                lines.append(f"{'  ' * level}{entry.name}/")
                walk(entry, level + 1)
            else:
                lines.append(f"{'  ' * level}{entry.name}")

    walk(base, 0)
    return lines


# --------------------------------------------------------------------------
# GPU sampling
# --------------------------------------------------------------------------
GPU_QUERY = "clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used,temperature.gpu"


def gpu_snapshot() -> str:
    _, text, _ = run([
        "nvidia-smi", f"--query-gpu=name,driver_version,memory.total,{GPU_QUERY}",
        "--format=csv,noheader",
    ], timeout=60)
    return text.strip()


class GpuSampler:
    """Background 1 Hz sampler of clocks/power/utilisation/VRAM."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            code, text, _ = run([
                "nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
            ], timeout=30)
            if code == 0 and text.strip():
                values = [part.strip() for part in text.strip().splitlines()[0].split(",")]
                if len(values) == 6:
                    self.samples.append({
                        "t": time.time(),
                        "sm_clock_mhz": _as_float(values[0]),
                        "mem_clock_mhz": _as_float(values[1]),
                        "power_w": _as_float(values[2]),
                        "util_percent": _as_float(values[3]),
                        "vram_used_mib": _as_float(values[4]),
                        "temperature_c": _as_float(values[5]),
                    })
            self._stop.wait(self.interval)

    def start(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "GpuSampler":
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        return self

    def summary(self) -> dict[str, object]:
        def column(key: str) -> list[float]:
            return [float(s[key]) for s in self.samples if s.get(key) is not None]

        def peak(key: str) -> float | None:
            values = column(key)
            return round(max(values), 2) if values else None

        def median(key: str) -> float | None:
            values = column(key)
            return round(statistics.median(values), 2) if values else None

        return {
            "sample_count": len(self.samples),
            "peak_vram_mib": peak("vram_used_mib"),
            "peak_power_w": peak("power_w"),
            "peak_temperature_c": peak("temperature_c"),
            "median_sm_clock_mhz": median("sm_clock_mhz"),
            "median_mem_clock_mhz": median("mem_clock_mhz"),
            "median_util_percent": median("util_percent"),
            "first_sm_clock_mhz": (column("sm_clock_mhz") or [None])[0],
            "last_sm_clock_mhz": (column("sm_clock_mhz") or [None])[-1],
        }


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# WAV fixtures
# --------------------------------------------------------------------------
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


def write_tone_wav(path: Path, seconds: int = 60, rate: int = 16000) -> Path:
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
            frames.extend(int(max(-1.0, min(1.0, sample)) * 32767.0).to_bytes(2, "little", signed=True))
        out.writeframes(frames)
    return path


def concat_wav(sources: list[Path], destination: Path) -> Path:
    """Concatenate same-format WAV files into one fixture (length scaling)."""
    if not sources:
        raise ValueError("concat_wav needs at least one source")
    with wave.open(str(sources[0]), "rb") as first:
        params = first.getparams()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setparams(params)
        for source in sources:
            with wave.open(str(source), "rb") as handle:
                if handle.getnchannels() != params.nchannels or handle.getframerate() != params.framerate:
                    raise ValueError(f"format mismatch in {source}")
                out.writeframes(handle.readframes(handle.getnframes()))
    return destination


def to_wav16k(source: Path, destination: Path) -> tuple[bool, str]:
    code, text, _ = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ], timeout=1800)
    return code == 0 and destination.exists(), text.strip()


# --------------------------------------------------------------------------
# runtime build
# --------------------------------------------------------------------------
def prepare_cuda_driver_shim() -> Path:
    shim = WORK_ROOT / "diar_cuda_driver_shim.cmake"
    shim.write_text(CUDA_DRIVER_SHIM, encoding="utf-8")
    return shim


def install_build_dependencies() -> None:
    if subprocess.run(["pkg-config", "--exists", "sentencepiece"], check=False).returncode == 0:
        return
    code, _, _ = run([
        "bash", "-lc", "apt-get update -qq && apt-get install -y -qq libsentencepiece-dev",
    ], timeout=1800)
    if code != 0:
        raise RuntimeError("could not install sentencepiece development files")


def clone_runtime() -> str:
    code, _, _ = run([
        "git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules",
        REPO_URL, str(REPO_DIR),
    ], timeout=3600)
    if code != 0:
        raise RuntimeError("NeMo-Speech.cpp clone failed")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), text=True).strip()


def build_runtime(preset: str = "cuda-diar", cuda_architectures: str = "60") -> dict[str, object]:
    """Configure and build. Returns a record of the build with timings."""
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        raise RuntimeError("cmake/ninja missing from the image")
    install_build_dependencies()
    commit = clone_runtime()
    shim = prepare_cuda_driver_shim()
    shutil.rmtree(REPO_DIR / "build", ignore_errors=True)
    code, configure_text, configure_seconds = run([
        "bash", "scripts/configure.sh", preset,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
        f"-DCMAKE_PROJECT_INCLUDE_BEFORE={shim}",
    ], cwd=REPO_DIR, timeout=1800)
    if code != 0:
        raise RuntimeError(f"configure failed (rc={code}): {configure_text[-2000:]}")
    code, build_text, build_seconds = run([
        "cmake", "--build", "--preset", preset, "--parallel", "2",
    ], cwd=REPO_DIR, timeout=7200)
    if code != 0:
        raise RuntimeError(f"build failed (rc={code}): {build_text[-2000:]}")
    binary = find_binary()
    return {
        "runtime_commit": commit, "preset": preset, "binary": str(binary),
        "cuda_architectures": cuda_architectures,
        "configure_seconds": round(configure_seconds, 3),
        "build_seconds": round(build_seconds, 3),
    }


def find_binary() -> Path:
    candidates = [p for p in REPO_DIR.rglob("nemo-speech") if p.is_file() and os.access(p, os.X_OK)]
    if not candidates:
        raise RuntimeError("nemo-speech binary not found after build")
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def download_model() -> dict[str, object]:
    code, _, seconds = run([
        "curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
        "-o", str(MODEL_PATH), MODEL_URL,
    ], timeout=3600)
    if code != 0 or not MODEL_PATH.exists():
        raise RuntimeError("Sortformer GGUF download failed")
    return {
        "model": MODEL_PATH.name,
        "model_bytes": MODEL_PATH.stat().st_size,
        "model_sha256": sha256(MODEL_PATH),
        "download_seconds": round(seconds, 3),
    }


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def parse_rttm(path: Path) -> dict[str, object]:
    speakers: dict[str, float] = {}
    segments: list[dict[str, object]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                continue
            start, duration = float(fields[3]), float(fields[4])
            speakers[fields[7]] = speakers.get(fields[7], 0.0) + duration
            segments.append({"start": start, "duration": duration, "speaker": fields[7]})
    return {
        "segments": len(segments),
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "speech_seconds": {k: round(v, 3) for k, v in sorted(speakers.items())},
        "last_end_seconds": round(max((s["start"] + s["duration"] for s in segments), default=0.0), 3),
    }


def diarize_once(
    binary: Path,
    audio: Path,
    output: Path,
    device: str = "cuda:0",
    preset: str | None = None,
    extra_args: list[str] | None = None,
    timeout: int = 3600,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    argv = [
        str(binary), "diarize", str(audio),
        "--diar-model", str(MODEL_PATH),
        "--device", device,
        "--format", "rttm",
        "--output", str(output),
        "--recording-id", output.stem,
    ]
    if preset:
        argv += ["--preset", preset]
    if extra_args:
        argv += extra_args
    cpu_before = child_cpu_seconds()
    code, text, seconds = run(argv, cwd=REPO_DIR, timeout=timeout)
    cpu_after = child_cpu_seconds()
    return {
        "returncode": code,
        "wall_seconds": round(seconds, 4),
        "cpu_seconds": round(cpu_after - cpu_before, 4),
        "output": str(output),
        "output_sha256": sha256(output) if output.exists() else None,
        "tail": text[-1200:],
    }


def measure(
    binary: Path,
    audio: Path,
    label: str,
    device: str = "cuda:0",
    preset: str | None = None,
    warmup: int = 1,
    iterations: int = 3,
    sample_gpu: bool = True,
) -> dict[str, object]:
    """Warm up, then time repeated runs; report median plus GPU sampling."""
    warmup_runs = []
    for index in range(warmup):
        result = diarize_once(binary, audio, WORK_ROOT / "out" / f"{label}.warm{index}.rttm", device, preset)
        warmup_runs.append({"returncode": result["returncode"], "wall_seconds": result["wall_seconds"]})

    sampler = GpuSampler().start() if sample_gpu else None
    runs = [diarize_once(binary, audio, WORK_ROOT / "out" / f"{label}.run{i}.rttm", device, preset)
            for i in range(iterations)]
    gpu = sampler.stop().summary() if sampler else {}

    ok = [r for r in runs if r["returncode"] == 0]
    wall = [float(r["wall_seconds"]) for r in ok]
    cpu = [float(r["cpu_seconds"]) for r in ok]
    info = wav_info(audio)
    audio_seconds = float(info["seconds"])
    median_wall = statistics.median(wall) if wall else None
    return {
        "label": label,
        "audio": str(audio),
        "audio_seconds": audio_seconds,
        "audio_info": info,
        "device": device,
        "preset": preset,
        "warmup": warmup_runs,
        "runs": runs,
        "timing": {
            "iterations_ok": len(ok),
            "iterations_requested": iterations,
            "wall_seconds_all": wall,
            "wall_seconds_median": median_wall,
            "wall_seconds_min": min(wall) if wall else None,
            "wall_seconds_max": max(wall) if wall else None,
            "wall_spread_percent": (round(100.0 * (max(wall) - min(wall)) / median_wall, 3)
                                    if wall and median_wall else None),
            "cpu_seconds_median": statistics.median(cpu) if cpu else None,
            "cpu_fraction_of_wall": (round(statistics.median(cpu) / median_wall, 4)
                                     if cpu and median_wall else None),
            "rtf_median": (median_wall / audio_seconds) if median_wall else None,
            "realtime_x_median": (audio_seconds / median_wall) if median_wall else None,
            "note": "end_to_end CLI wall time: process start, CUDA init, model load, decode, inference, RTTM write.",
        },
        "gpu": gpu,
        "rttm": parse_rttm(WORK_ROOT / "out" / f"{label}.run0.rttm"),
        "status": "pass" if len(ok) == iterations else "fail",
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def environment_record() -> dict[str, object]:
    _, cuda_text, _ = run(["bash", "-lc", "nvcc --version 2>/dev/null | tail -2 || true"], timeout=60)
    _, cc_text, _ = run([
        "bash", "-lc",
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true",
    ], timeout=60)
    _, driver_text, _ = run([
        "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
    ], timeout=60)
    gpu_name, _, driver = (driver_text.strip().partition(","))
    return {
        "os": platform.platform(),
        "compiler": (cuda_text.strip().splitlines()[0] if cuda_text.strip() else "unknown"),
        "cuda_toolkit": cuda_text.strip() or None,
        "driver": driver.strip() or None,
        "gpu": gpu_name.strip() or "none",
        "compute_capability": cc_text.strip() or None,
    }


def benchmark_record(
    commit: str,
    stage: str,
    environment: dict[str, object],
    workload: dict[str, object],
    timing: dict[str, object],
    accuracy: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one schema-conformant benchmark-v1 record."""
    record = {
        "schema_version": 1,
        "commit": commit,
        "scope": "pure_speaker_diarization",
        "stage": stage,
        "environment": environment,
        "workload": workload,
        "timing": timing,
        "accuracy": accuracy or {"status": "not_scored"},
    }
    if extra:
        record.update(extra)
    return record


def append_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def emit_report(report: dict[str, object], name: str = "report.json") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False))
    if LOG_PATH.exists():
        tail = LOG_PATH.read_text(encoding="utf-8", errors="replace")[-4000:]
        (OUT_DIR / "run_tail.log").write_text(tail, encoding="utf-8")
    return path
