"""Sortformer v2 on Tesla P100: a measurement matrix, not a single number.

The previous run established that `nemo-speech diarize` completes on sm_60. This
run answers the questions that actually drive engine design:

1. What is the *marginal* compute cost per audio second, separated from the
   fixed process/model-load cost? (length scaling fit)
2. Does throughput hold on long inputs, or does state growth slow it down?
3. How fast is the CPU path on the same binary, for the fallback claim?
4. Is streaming vs offline geometry materially different in cost?
5. Does the same input produce a byte-identical RTTM across runs (a hard
   prerequisite before any parity claim)?
6. Does `--concurrency` batch directory work usefully on one P100?

Everything is pure diarization: no ASR, tokenizer, or transcription path.
Artifacts are small reports; build trees and weights stay in /tmp.
"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, "/kaggle/input/diar-gpu-engine-harness")

import diar_harness as h  # noqa: E402

STAGE = "m1-sortformer-p100-matrix"
OUT = Path("/kaggle/working")
FIXTURE_ROOT = Path("/kaggle/input")
COMMIT = "d00a769"

REPORT: dict[str, object] = {
    "schema_version": 2,
    "scope": "pure_speaker_diarization",
    "job": "sortformer_v2_p100_matrix",
    "harness_source": "/kaggle/input/diar-gpu-engine-harness/diar_harness.py",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def cut_wav(source: Path, destination: Path, start_seconds: float, seconds: float) -> Path:
    """Take a contiguous slice, preserving the source format."""
    with wave.open(str(source), "rb") as handle:
        params = handle.getparams()
        handle.setpos(int(start_seconds * handle.getframerate()))
        frames = handle.readframes(int(seconds * handle.getframerate()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return destination


def find_fixture(pattern: str) -> Path | None:
    matches = sorted(FIXTURE_ROOT.glob(pattern))
    return matches[0] if matches else None


def build_fixtures() -> tuple[list[tuple[str, Path]], dict[str, str]]:
    """Returns (fixtures, provenance). Synthetic lengths make scaling measurable."""
    fixtures: list[tuple[str, Path]] = []
    provenance: dict[str, str] = {}

    real_short = find_fixture("diar-smoke-audio/**/*.wav")
    real_mid = find_fixture("diar-real-audio-5/**/*.wav")
    real_long = find_fixture("diar-real-audio-1/**/*.mp3")

    for label, path in (("real_short", real_short), ("real_mid", real_mid)):
        if path:
            fixtures.append((f"{label}_{h.slugify(path.stem, 18)}", path))
            provenance[f"{label}_{h.slugify(path.stem, 18)}"] = f"real audio: {path}"

    if real_long:
        converted = h.WORK_ROOT / "real_long.wav"
        ok, message = h.to_wav16k(real_long, converted)
        provenance["real_long"] = f"real audio converted to 16k mono ({ok}): {message}"
        if ok:
            fixtures.append(("real_long", converted))

    unit = h.write_tone_wav(h.WORK_ROOT / "tone_unit_60s.wav", seconds=60)
    for seconds in (300, 900, 1800):
        repeats = seconds // 60
        target = h.WORK_ROOT / f"synth_{seconds}s.wav"
        h.concat_wav([unit] * repeats, target)
        fixtures.append((f"synth_{seconds}s", target))
        provenance[f"synth_{seconds}s"] = f"{repeats}x concatenated 60s deterministic tone unit"

    return fixtures, provenance


def run_matrix(binary: Path, fixtures: list[tuple[str, Path]]) -> list[dict[str, object]]:
    results = []
    for label, audio in fixtures:
        entry = h.measure(binary, audio, label, device="cuda:0", preset=None, warmup=1, iterations=3)
        results.append(entry)
        print(f"[gpu] {label}: {json.dumps(entry['timing'], sort_keys=True)}", flush=True)
    return results


def run_cpu(binary: Path, fixtures: list[tuple[str, Path]]) -> list[dict[str, object]]:
    """CPU fallback cost - short fixtures only, the point is the ratio."""
    results = []
    for label, audio in fixtures[:2]:
        entry = h.measure(
            binary, audio, f"cpu_{label}", device="cpu", preset=None,
            warmup=0, iterations=1, sample_gpu=False,
        )
        results.append(entry)
        print(f"[cpu] {label}: rc={entry['runs'][0]['returncode']} "
              f"wall={entry['timing']['wall_seconds_median']}", flush=True)
    return results


def run_offline(binary: Path, fixtures: list[tuple[str, Path]]) -> list[dict[str, object]]:
    """Offline geometry is documented for short audio; compare cost on the same file."""
    results = []
    for label, audio in fixtures[:1]:
        entry = h.measure(
            binary, audio, f"offline_{label}", device="cuda:0", preset="offline",
            warmup=1, iterations=2,
        )
        results.append(entry)
        print(f"[offline] {label}: {json.dumps(entry['timing'], sort_keys=True)}", flush=True)
    return results


def run_concurrency(binary: Path, source: Path) -> dict[str, object]:
    """Batch throughput: same total audio as one long file, split into 4 chunks.

    ``-c`` lets directory work share a model and batch compatible GPU steps, so
    this measures whether that batching is worth anything on a single P100.
    """
    directory = h.WORK_ROOT / "concurrency"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.wav"):
        stale.unlink()
    total = h.wav_info(source)["seconds"]
    chunk = total / 4.0
    for index in range(4):
        cut_wav(source, directory / f"chunk{index}.wav", index * chunk, chunk)

    outcomes = []
    for concurrency in (1, 4):
        out_dir = h.WORK_ROOT / f"conc_out_{concurrency}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("*.rttm"):
            stale.unlink()
        audio_seconds = sum(h.wav_info(p)["seconds"] for p in sorted(directory.glob("*.wav")))
        code, text, seconds = h.run([
            str(binary), "diarize", str(directory),
            "--diar-model", str(h.MODEL_PATH),
            "--device", "cuda:0", "--format", "rttm",
            "--output-dir", str(out_dir),
            "--concurrency", str(concurrency),
        ], cwd=h.REPO_DIR, timeout=3600)
        produced = sorted(p.name for p in out_dir.glob("*.rttm"))
        outcomes.append({
            "concurrency": concurrency,
            "returncode": code,
            "wall_seconds": round(seconds, 4),
            "audio_seconds": audio_seconds,
            "rtf": round(seconds / audio_seconds, 6) if audio_seconds else None,
            "realtime_x": round(audio_seconds / seconds, 3) if seconds else None,
            "files_produced": produced,
            "tail": text[-600:],
        })
        print(f"[concurrency={concurrency}] wall={seconds:.3f}s files={len(produced)}", flush=True)
    return {
        "chunks": [str(p) for p in sorted(directory.glob("*.wav"))],
        "chunk_seconds": chunk,
        "outcomes": outcomes,
        "speedup_4_vs_1": (
            round(outcomes[0]["wall_seconds"] / outcomes[1]["wall_seconds"], 4)
            if len(outcomes) == 2 and outcomes[1]["wall_seconds"] else None
        ),
    }


def fit_scaling(entries: list[dict[str, object]]) -> dict[str, object]:
    """Least squares wall_seconds = fixed + marginal_rtf * audio_seconds."""
    points = [
        (float(e["audio_seconds"]), float(e["timing"]["wall_seconds_median"]))
        for e in entries
        if e.get("timing", {}).get("wall_seconds_median")
    ]
    if len(points) < 2:
        return {"points": points, "note": "need >= 2 fixtures"}
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
    predicted = [intercept + slope * x for x, _ in points]
    residuals = [y - p for (_, y), p in zip(points, predicted)]
    return {
        "points": [{"audio_seconds": round(x, 3), "wall_seconds": round(y, 3)} for x, y in points],
        "fixed_seconds": round(intercept, 4),
        "marginal_rtf": round(slope, 6),
        "marginal_realtime_x": round(1.0 / slope, 3) if slope > 0 else None,
        "max_abs_residual_seconds": round(max(abs(r) for r in residuals), 4),
        "note": "fixed = process start + CUDA init + model load; marginal = compute per audio second",
    }


def determinism(entries: list[dict[str, object]]) -> dict[str, object]:
    """Same input, fresh output path, byte-identical RTTM?"""
    checks = []
    for entry in entries:
        hashes = [r["output_sha256"] for r in entry["runs"] if r["returncode"] == 0]
        checks.append({
            "label": entry["label"],
            "runs": len(hashes),
            "unique_hashes": len(set(hashes)),
            "identical": len(set(hashes)) == 1,
            "sha256": hashes[0] if hashes else None,
        })
    return {
        "checks": checks,
        "all_identical": all(c["identical"] for c in checks) if checks else False,
        "note": "byte-level RTTM equality across repeated runs on the same fixture",
    }


def to_records(environment: dict[str, object], entries: list[dict[str, object]], model: dict[str, object]) -> list[dict[str, object]]:
    records = []
    for entry in entries:
        timing = entry["timing"]
        device = entry["device"]
        records.append(h.benchmark_record(
            commit=COMMIT,
            stage=STAGE,
            environment=environment,
            workload={
                "model": "diar_streaming_sortformer_4spk-v2 (q8_0 GGUF)",
                "checkpoint_hash": model.get("model_sha256"),
                "fixture": entry["label"],
                "precision": "Q8_0",
                "quantization": "q8_0",
                "batch": 1,
                "chunk_frames": None,
                "right_context_frames": None,
                "fifo_frames": None,
                "spkcache_frames": None,
            },
            timing={
                "audio_seconds": entry["audio_seconds"],
                "warmup": len(entry["warmup"]),
                "iterations": timing["iterations_ok"],
                "frontend_seconds": 0.0,
                "neural_core_seconds": 0.0,
                "postprocess_seconds": 0.0,
                "wall_seconds": timing["wall_seconds_median"] or 0.0,
                "rtf": timing["rtf_median"] or 0.0,
                "realtime_x": timing["realtime_x_median"] or 0.0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
            },
            accuracy={
                "status": "not_scored",
                "max_abs_probability_error": None,
                "frame_agreement": None,
                "der": None,
                "jer": None,
                "notes": "timing only; no reference labels scored in this stage",
            },
            extra={
                "device": device,
                "preset": entry["preset"] or "streaming",
                "level": "end_to_end",
                "gpu_sampling": entry["gpu"],
                "rttm": entry["rttm"],
                "wall_spread_percent": timing["wall_spread_percent"],
            },
        ))
    return records


def _main() -> None:
    h.reset_workspace()
    OUT.mkdir(parents=True, exist_ok=True)
    environment = h.environment_record()
    REPORT["environment"] = environment
    REPORT["gpu_before"] = h.gpu_snapshot()

    build = h.build_runtime(preset="cuda-diar", cuda_architectures="60")
    REPORT["build"] = build
    binary = Path(build["binary"])
    REPORT["runtime_version_output"] = h.run([str(binary), "--version"], cwd=h.REPO_DIR, timeout=120)[1].strip()[:200]

    model = h.download_model()
    REPORT["model"] = model

    fixtures, provenance = build_fixtures()
    REPORT["fixtures"] = {label: str(path) for label, path in fixtures}
    REPORT["fixture_provenance"] = provenance

    gpu_entries = run_matrix(binary, fixtures)
    REPORT["gpu_streaming"] = gpu_entries
    REPORT["length_scaling_fit"] = fit_scaling(gpu_entries)
    REPORT["determinism"] = determinism(gpu_entries)

    REPORT["gpu_offline_preset"] = run_offline(binary, fixtures)
    REPORT["cpu_fallback"] = run_cpu(binary, fixtures)

    if fixtures:
        REPORT["directory_concurrency"] = run_concurrency(binary, fixtures[0][1])

    REPORT["gpu_after"] = h.gpu_snapshot()
    REPORT["records"] = to_records(environment, gpu_entries, model)
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    exit_code = 0
    try:
        _main()
        REPORT["status"] = "pass"
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive on Kaggle
        REPORT["status"] = "error"
        REPORT["error_type"] = type(exc).__name__
        REPORT["error"] = str(exc)
        exit_code = 1
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="sortformer_matrix_report.json")
    records = REPORT.get("records") or []
    if records:
        h.append_jsonl(OUT / "benchmarks.jsonl", records)
        print(f"\nwrote {len(records)} schema-conformant benchmark records")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
