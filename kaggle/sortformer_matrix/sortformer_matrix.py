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
7. (v6) Which layer drifts? Same-audio env-knob sweep (CUDA_LAUNCH_BLOCKING,
   CUBLAS_WORKSPACE_CONFIG, GGML_SKINNY_Q8_CUBLAS_F16) on short-streaming,
   mid-offline and mid-streaming isolates the nondeterminism source without
   touching the engine.

Everything is pure diarization: no ASR, tokenizer, or transcription path.
Artifacts are small reports; build trees and weights stay in /tmp.
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path


def locate_harness() -> Path:
    """Find the published harness dataset wherever Kaggle mounted it.

    Dataset mount paths depend on the owner ref and can nest, so resolve the
    module's directory by searching rather than hard-coding one path.
    """
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    listing: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for entry in sorted(root.iterdir()):
                listing.append(str(entry))
        except OSError as exc:
            listing.append(f"{root}: {exc}")
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("diar_harness.py"):
            return candidate.parent
    raise RuntimeError(
        "diar_harness.py not found. Kaggle input listing: " + " | ".join(listing)
    )


HARNESS_DIR = locate_harness()
sys.path.insert(0, str(HARNESS_DIR))

import diar_harness as h  # noqa: E402

STAGE = "m1-sortformer-p100-matrix"
OUT = Path("/kaggle/working")
FIXTURE_ROOT = Path("/kaggle/input")
COMMIT = "d00a769"

REPORT: dict[str, object] = {
    "schema_version": 3,
    "scope": "pure_speaker_diarization",
    "job": "sortformer_v2_p100_matrix",
    "harness_source": "diar_harness.py resolved at runtime from /kaggle/input",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

# v8 pipeline-knob sweep: same binary, same audio, only the CLI flags/env
# differ between sets. v7 excluded the CUDA/cuBLAS layer (async launch,
# workspace algorithm choice, F16 accumulation); v8 moves the probes up to
# the diarization pipeline itself, all via flags the pinned CLI accepts
# (NeMo-Speech.cpp a5b6953 app/diarize.cpp):
#   baseline    — no knobs (control; set[0] must stay knob-free, the
#                 cross-set comparison keys on it);
#   no_batching — --no-batching isolates GPU dynamic-batching nondeterminism.
#                 NOTE: batching only engages when workers > 1 (directory
#                 input, diarize.cpp:222); single-file runs never batch, so
#                 this set is expected to match baseline on single-file cases
#                 and only informative on directory/concurrency probes;
#   fixed_chunk — explicit --diar-chunk 20 --diar-fifo 80 --diar-spkcache 160
#                 (= streaming preset defaults in aosc_state.h) isolates
#                 chunk-boundary vs AOSC-state drift: if preset resolution
#                 itself drifts, explicit geometry diverges from baseline;
#   offline_full— --offline = full-attention single pass with NO streaming
#                 state, vs --preset offline (= larger AOSC streaming chunks).
#                 If offline_full is identical across reps while streaming
#                 drifts, the drift lives in AOSC state, not the model.
# Each set is (name, extra_args, env): extra_args reach nemo-speech via
# h.diarize_once(extra_args=...), env reaches the child via h.run(env=...).
#
# v9 prob-stagger sweep (DET_SETS_V9): asks WHERE the drift lives, not which
# knob collapses it. Every set dumps frame probabilities (DIAR_DUMP_PROBS via
# h.diarize_once(dump_probs_path=...), binary needs the harness probdump
# patch) so the verdict compares probs before hysteresis, not just RTTM
# bodies after it:
#   short_streaming x3  — known-good control, probs must be bit-identical;
#   mid_streaming x5    — v5/v7/v8's drifter, 5 reps resolve the distribution;
#   mid_offline_full x3 — full-attention control, probs must be bit-identical;
#   mid_offline_preset x3 — larger-chunk AOSC streaming; v8 showed it drifts
#                 too, so this set rules out "small-chunk-only" drift.
# N_DET_REPEAT stays 3 (env-overridable via DIAR_DET_REPEAT); the mid_streaming
# case runs 5 reps via DET_REPEAT_OVERRIDE below. Analysis per set: prob
# bit-identical? max/mean abs diff? argmax frame agreement? body hash equal?
# Verdict rule: probs drift + bodies drift = neural (GPU kernel) source;
# probs identical + bodies drift = host-side race (P0 bug: host is oracle-
# pinned, this outcome contradicts the AOSC/BirthGate differential proofs).
DET_SETS_V8: list[tuple[str, list[str], dict[str, str]]] = [
    ("baseline", [], {}),
    ("no_batching", ["--no-batching"], {}),
    ("fixed_chunk", ["--diar-chunk", "20", "--diar-fifo", "80",
                     "--diar-spkcache", "160"], {}),
    ("offline_full", ["--offline"], {}),
]
N_DET_REPEAT = int(os.environ.get("DIAR_DET_REPEAT", "3"))

# v9 prob-stagger sweep: (case_label, preset, reps, dump_probs). The case
# label doubles as the audio selector in _main (real_short_*/real_mid_* from
# the matrix entries). dump_probs=True routes every rep's probabilities to
# det_sweep/<label>/probs.<rep>.f32 via dump_probs_path.
DET_SETS_V9: list[tuple[str, str | None, int, bool]] = [
    ("short_streaming", None, 3, True),
    ("mid_streaming", None, 5, True),
    ("mid_offline_full", "unused-offline-flag", 3, True),
    ("mid_offline_preset", "offline", 3, True),
]
# --offline is a CLI flag, not a preset: mid_offline_full runs WITHOUT any
# preset and WITH the --offline flag. The sentinel above keeps the tuple
# shape uniform; run_prob_sweep translates it (see the flag branch there).
OFFLINE_FLAG_SENTINEL = "unused-offline-flag"


def _export_parity_candidate(
    label: str,
    audio: Path,
    preset: str | None,
    sweep: dict[str, object],
) -> dict[str, object] | None:
    """v12: copy rep0's probs dump + RTTM + raw provenance into OUT.

    The dumps are written under WORK_ROOT (harness) which dies with the
    session — parity fixtures may only be backfilled from kernel OUTPUT
    (truth discipline), so rep0's payloads must physically land in
    /kaggle/working. Only rep0 is exported: a fixture pins ONE observation,
    not the sweep. The local scripts/fill_parity_fixture.py turns the
    candidate directory into a schema-v1 parity fixture.
    """
    import hashlib
    import shutil

    cases = sweep.get("cases") or []
    if not cases:
        return None
    case = cases[0]
    reps = case.get("reps") or []
    if not reps or not reps[0].get("probs_available"):
        print(f"[parity-candidate] {label}: rep0 has no probs dump, skipped",
              flush=True)
        return None
    case_dir = h.WORK_ROOT / "det_sweep" / label
    src_probs = case_dir / "probs.0.f32"
    src_rttm = case_dir / "rep0.rttm"
    if not src_probs.exists():
        print(f"[parity-candidate] {label}: {src_probs} missing, skipped",
              flush=True)
        return None
    dst = OUT / "parity_candidates" / label
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_probs, dst / "probs.f32")
    if src_rttm.exists():
        shutil.copy2(src_rttm, dst / "rep0.rttm")
    audio_sha = None
    if Path(audio).exists():
        audio_sha = hashlib.sha256(Path(audio).read_bytes()).hexdigest()
    prov: dict[str, object] = {
        "label": label,
        "audio_path": str(audio),
        "audio_sha256": audio_sha,
        "preset": None if preset == OFFLINE_FLAG_SENTINEL else preset,
        "offline_flag": preset == OFFLINE_FLAG_SENTINEL,
        "env_pins": dict(V12_ENV_PINS),
        "cross_session_stable": label in V11_SESSION_STABLE,
        "reps": [
            {k: r.get(k) for k in ("rep", "body_sha256", "probs_sha256",
                                   "probs_shape", "probs_available")}
            for r in reps
        ],
        "model_path": str(h.MODEL_PATH),
        "runtime_version": REPORT.get("runtime_version_output"),
        "gpu_before": REPORT.get("gpu_before"),
        "upstream_commit": "a5b6953",
        "harness_commit": COMMIT,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (dst / "candidate.json").write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n")
    print(f"[parity-candidate] {label}: exported rep0 -> {dst}", flush=True)
    return {"label": label, "dir": str(dst), "stable": prov["cross_session_stable"]}

# v13 (attribution experiment): v12 minus the env pins. v11 (unpinned) and
# v12 (pinned) produced IDENTICAL hashes on all four cases — including mid
# ddf2312b, which had never repeated across sessions before. Two candidate
# explanations: (a) the environment/build became deterministic at v11 and the
# pins are no-ops; (b) the pins select the same algo autotune already picked.
# v13 flips ONLY the pin variable against v12:
#   mid == ddf2312b unpinned  -> environment converged at v11; pins no-op;
#   mid != ddf2312b unpinned  -> pins are causal; mid fixtures gate on pins.
# Either way a third session hash locks fixture legitimacy for the cases
# that reproduce.
V12_ENV_PINS: dict[str, str] = {}
# Cross-session-stable cases per the v11 hash lineage (short == v5/v7/v8
# baseline; offline_full == v8's stable hash). These are the only legitimate
# parity L1 fixture candidates; the mid cases churn across sessions.
V11_SESSION_STABLE = ("short_streaming", "mid_offline_full")


def run_prob_sweep(
    binary: Path,
    cases: list[tuple[str, Path, str | None]],
) -> dict[str, object]:
    """v9: same audio x prob-stagger sets, every rep dumps probabilities.

    Each case is (label, audio, preset). Inside one case the binary, audio,
    preset and working directory are all fixed; reps differ only by process
    invocation. Every rep writes ``probs.<rep>.f32`` next to its RTTM, then
    the analysis compares rep0 against each later rep at BOTH levels:

    * probs level (pre-hysteresis): bit-identical? max/mean abs diff?
      argmax frame agreement? shape stable?
    * body level (post-hysteresis): RTTM body hash equal to rep0?

    The verdict rule is structural: probs drift + bodies drift = neural
    (GPU kernel) source; probs identical + bodies drift = host-side race.
    A rep whose binary lacks the patch (or whose dump is missing/corrupt)
    fails loudly with probs_available=False — a silent no-dump run must
    never read as "identical".
    """
    import hashlib

    sweep: dict[str, object] = {"sets": DET_SETS_V9, "cases": []}
    for label, audio, preset in cases:
        case_dir = h.WORK_ROOT / "det_sweep" / label
        case_dir.mkdir(parents=True, exist_ok=True)
        case_entry: dict[str, object] = {"label": label, "preset": preset, "reps": []}
        # --offline is a CLI flag, not a --preset value: the sentinel preset
        # maps to preset=None + the flag in extra_args.
        use_offline_flag = preset == OFFLINE_FLAG_SENTINEL
        child_preset = None if use_offline_flag else preset
        child_extra = ["--offline"] if use_offline_flag else []
        rep_dumps: list[tuple[int, int, list[float]] | None] = []
        rep_hashes: list[str] = []
        rep_bodies: list[str] = []
        ok = True
        for rep in range(N_DET_REPEAT):
            out = case_dir / f"rep{rep}.rttm"
            dump_path = case_dir / f"probs.{rep}.f32"
            result = h.diarize_once(
                binary, audio, out, device="cuda:0", preset=child_preset,
                extra_args=list(child_extra),
                timeout=3600, dump_probs_path=dump_path,
                env=dict(V12_ENV_PINS),
            )
            rep_record: dict[str, object] = {
                "rep": rep,
                "returncode": result["returncode"],
                "wall_seconds": float(result["wall_seconds"]),
                "probs_path": str(dump_path),
                "probs_sha256": result["probs_sha256"],
            }
            if result["returncode"] != 0:
                ok = False
                rep_record["probs_available"] = False
                rep_record["error"] = (result.get("tail") or "")[-300:]
                case_entry["reps"].append(rep_record)
                break
            body = "\n".join(
                f"{s:.3f} {d:.3f} {spk}" for s, d, spk in h.read_segments(out))
            rep_bodies.append(body)
            rep_hashes.append(hashlib.sha256(body.encode()).hexdigest())
            rep_record["body_sha256"] = rep_hashes[-1]
            try:
                rep_dumps.append(h.read_probdump(dump_path))
                rep_record["probs_available"] = True
                rep_record["probs_shape"] = list(rep_dumps[-1][:2])
            except (FileNotFoundError, ValueError) as exc:
                rep_dumps.append(None)
                rep_record["probs_available"] = False
                rep_record["probs_error"] = f"{type(exc).__name__}: {exc}"
                ok = False
                case_entry["reps"].append(rep_record)
                break
            case_entry["reps"].append(rep_record)
        # pairwise rep0-vs-repN at both levels (rep0 is the reference)
        prob_pairs: list[dict[str, object]] = []
        body_pairs: list[bool] = []
        if len(rep_dumps) >= 2 and all(d is not None for d in rep_dumps):
            first = rep_dumps[0]
            assert first is not None
            for other in rep_dumps[1:]:
                assert other is not None
                prob_pairs.append(h.probdiff(first, other))
        if len(rep_hashes) >= 2:
            body_pairs = [x == rep_hashes[0] for x in rep_hashes[1:]]
        unique_bodies = sorted(set(rep_hashes))
        probs_ok = bool(rep_dumps) and all(d is not None for d in rep_dumps)
        probs_bit = (
            all(bool(p.get("bit_identical")) for p in prob_pairs)
            if prob_pairs else (len(rep_dumps) == 1 and probs_ok)
        )
        max_abs = max((float(p.get("max_abs_diff", 0.0)) for p in prob_pairs), default=0.0)
        mean_abs = max((float(p.get("mean_abs_diff", 0.0)) for p in prob_pairs), default=0.0)
        min_agree = min(
            (float(p.get("frame_agreement", 1.0)) for p in prob_pairs), default=1.0)
        if not probs_ok:
            verdict = "no_dump"
        elif probs_bit and len(unique_bodies) == 1:
            verdict = "identical"
        elif not probs_bit and len(unique_bodies) > 1:
            verdict = "neural_drift"
        elif probs_bit and len(unique_bodies) > 1:
            verdict = "host_race"
        elif not probs_bit and len(unique_bodies) == 1:
            verdict = "sub_hysteresis_drift"
        else:
            verdict = "unreachable"
        case_entry["returncode_ok"] = ok
        case_entry["probs_available"] = probs_ok
        case_entry["prob_pairs_vs_rep0"] = prob_pairs
        case_entry["body_pairs_vs_rep0"] = body_pairs
        case_entry["unique_body_hashes"] = len(unique_bodies)
        case_entry["body_identical"] = ok and len(unique_bodies) == 1
        case_entry["probs_bit_identical"] = probs_ok and probs_bit
        case_entry["probs_max_abs_diff"] = max_abs if probs_ok else None
        case_entry["probs_mean_abs_diff"] = mean_abs if probs_ok else None
        case_entry["probs_min_frame_agreement"] = min_agree if probs_ok else None
        case_entry["verdict"] = verdict if ok else "run_failed"
        print(f"[prob] {label}: verdict={case_entry['verdict']} "
              f"bodies_unique={len(unique_bodies)} probs_bit={probs_ok and probs_bit} "
              f"max_abs={max_abs:.3g} agree={min_agree:.4f}", flush=True)
        sweep["cases"].append(case_entry)
    return sweep


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


AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def discover_audio(root: Path) -> list[Path]:
    """Find audio under a mount root without assuming a mount layout.

    Kaggle has used both ``/kaggle/input/<slug>/`` and the nested
    ``/kaggle/input/datasets/<owner>/<slug>/`` layout, so search recursively.
    Implemented locally (rather than relying on the harness dataset version) so
    this kernel keeps working whichever harness revision gets mounted.
    """
    if not root.exists():
        return []
    found: list[Path] = []
    for suffix in AUDIO_SUFFIXES:
        found.extend(p for p in root.rglob(f"*{suffix}") if p.is_file())
    return sorted(set(found))


def layout_snapshot(root: Path, depth: int = 3) -> list[str]:
    if not root.exists():
        return [f"MISSING {root}"]
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

    walk(root, 0)
    return lines


def build_fixtures() -> tuple[list[tuple[str, Path]], dict[str, str]]:
    """Returns (fixtures, provenance).

    Real audio is discovered by recursive search rather than a fixed mount
    prefix. Synthetic lengths (concatenated deterministic tone) exist purely to
    make the length-scaling fit well conditioned; they carry no speech and are
    speed fixtures, not accuracy corpora.
    """
    fixtures: list[tuple[str, Path]] = []
    provenance: dict[str, str] = {}

    audio = discover_audio(FIXTURE_ROOT)
    REPORT["input_layout"] = layout_snapshot(FIXTURE_ROOT)
    REPORT["discovered_audio"] = [str(p) for p in audio]

    def pick(*needles: str) -> Path | None:
        for path in audio:
            text = str(path).lower()
            if all(needle in text for needle in needles):
                return path
        return None

    real_short = pick("smoke-audio") or pick("four-speakers")
    real_mid = pick("real-audio-5")
    real_long = pick("real-audio-1")

    for label, path in (("real_short", real_short), ("real_mid", real_mid)):
        if not path:
            continue
        if path.suffix.lower() == ".wav":
            fixtures.append((f"{label}_{h.slugify(path.stem, 16)}", path))
            provenance[label] = f"real audio: {path}"
        else:
            converted = h.WORK_ROOT / f"{label}.wav"
            ok, message = h.to_wav16k(path, converted)
            provenance[label] = f"real audio converted to 16k mono ({ok}): {message}"
            if ok:
                fixtures.append((label, converted))

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
        provenance[f"synth_{seconds}s"] = f"{repeats}x concatenated 60s deterministic tone unit (speed fixture, no speech)"

    REPORT["fixture_provenance"] = provenance
    return fixtures, provenance


def run_matrix(binary: Path, fixtures: list[tuple[str, Path]]) -> list[dict[str, object]]:
    results = []
    for label, audio in fixtures:
        entry = h.measure(binary, audio, label, device="cuda:0", preset=None, warmup=1, iterations=3)
        results.append(entry)
        print(f"[gpu] {label}: {json.dumps(entry['timing'], sort_keys=True)}", flush=True)
    return results


def select_entries(entries: list[dict[str, object]], prefix: str) -> list[dict[str, object]]:
    """Filter measurement entries by label prefix.

    (A previous revision wrongly called the tuple-based ``select`` on entry
    dicts, which raised ``KeyError: 0`` and killed the whole run after the GPU
    matrix had already completed. Entry filtering is string-based by design.)
    """
    return [entry for entry in entries if str(entry.get("label", "")).startswith(prefix)]


def run_det_sweep(
    binary: Path,
    cases: list[tuple[str, Path, str | None]],
) -> dict[str, object]:
    """v8: same audio x pipeline-knob sets, repeated sequential single-file runs.

    Each case is (label, audio, preset). Inside one case the binary, audio,
    preset, recording-id and working directory are all fixed; the only thing
    that varies between sets is extra CLI flags / child env (see DET_SETS_V8).
    Every set runs N_DET_REPEAT sequential `diarize` invocations so the verdict
    compares like with like: does this knob collapse the run-to-run drift that
    v5/v7 measured, or leave it untouched?
    """
    import hashlib

    sweep: dict[str, object] = {"sets": DET_SETS_V8, "repeat": N_DET_REPEAT, "cases": []}
    for label, audio, preset in cases:
        case_dir = h.WORK_ROOT / "det_sweep" / label
        case_dir.mkdir(parents=True, exist_ok=True)
        case_entry: dict[str, object] = {"label": label, "preset": preset, "sets": []}
        for set_name, extra_args, env in DET_SETS_V8:
            bodies: list[str] = []
            hashes: list[str] = []
            walls: list[float] = []
            ok = True
            for rep in range(N_DET_REPEAT):
                out = case_dir / f"{set_name}.rep{rep}.rttm"
                result = h.diarize_once(
                    binary, audio, out, device="cuda:0", preset=preset,
                    extra_args=list(extra_args), timeout=3600, env=dict(env),
                )
                if result["returncode"] != 0:
                    ok = False
                    break
                body = "\n".join(
                    f"{s:.3f} {d:.3f} {spk}" for s, d, spk in h.read_segments(out))
                bodies.append(body)
                hashes.append(hashlib.sha256(body.encode()).hexdigest())
                walls.append(float(result["wall_seconds"]))
            unique = sorted(set(hashes))
            case_entry["sets"].append({
                "name": set_name,
                "extra_args": list(extra_args),
                "env": dict(env),
                "returncode_ok": ok,
                "runs": N_DET_REPEAT if ok else len(hashes),
                "unique_body_hashes": len(unique),
                "body_identical": ok and len(unique) == 1,
                "pairwise_with_rep0": (
                    round(sum(1 for x in hashes if x == hashes[0]) / len(hashes), 4)
                    if hashes else None
                ),
                "cross_set_matches_baseline_rep0": (
                    hashes[0] == case_entry["sets"][0].get("rep0_hash")
                    if case_entry["sets"] and hashes else None
                ),
                "rep0_hash": hashes[0] if hashes else None,
                "wall_seconds": walls,
                "wall_median": (
                    round(sorted(walls)[len(walls) // 2], 4) if walls else None),
            })
            print(f"[det] {label}/{set_name}: unique={len(unique)} "
                  f"identical={ok and len(unique) == 1} walls={walls}", flush=True)
        sweep["cases"].append(case_entry)
    return sweep


def run_cpu(binary: Path, fixtures: list[tuple[str, Path]]) -> list[dict[str, object]]:
    """CPU fallback cost. One short real fixture: the point is the GPU ratio."""
    results = []
    for label, audio in fixtures[:1]:
        entry = h.measure(
            binary, audio, f"cpu_{label}", device="cpu", preset=None,
            warmup=0, iterations=1, sample_gpu=False,
        )
        results.append(entry)
        print(f"[cpu] {label}: rc={entry['runs'][0]['returncode']} "
              f"wall={entry['timing']['wall_seconds_median']}", flush=True)
    return results


def run_offline(binary: Path, fixtures: list[tuple[str, Path]]) -> list[dict[str, object]]:
    """Offline geometry is documented for short audio; compare cost on same files."""
    results = []
    for label, audio in fixtures[:2]:
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

    Three variants, all covering the same audio, in wall-clock order:
    ``file/chunks`` (glued RTTM vs whole-file RTTM body equality tells whether
    chunking changes content), ``dir-c1`` and ``dir-c4`` (directory batching).
    The directory CLI has no ``--recording-id`` (rejected for directory input),
    so per-file recording ids are the input stems; the single-file baseline is
    run with ``--recording-id`` fixed to the first chunk stem so bodies are
    comparable byte-for-byte.
    """
    import hashlib

    directory = h.WORK_ROOT / "concurrency"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.wav"):
        stale.unlink()
    total = h.wav_info(source)["seconds"]
    chunk = total / 4.0
    chunk_paths = []
    for index in range(4):
        chunk_paths.append(cut_wav(source, directory / f"chunk{index}.wav", index * chunk, chunk))

    def body_of(rttm: Path) -> str:
        return "\n".join(f"{s:.3f} {d:.3f} {spk}" for s, d, spk in h.read_segments(rttm))

    # Variant A: single-file baseline over the same chunks (fixed recording id).
    file_out = h.WORK_ROOT / "conc_file_out"
    file_out.mkdir(parents=True, exist_ok=True)
    for stale in file_out.glob("*.rttm"):
        stale.unlink()
    file_seconds = 0.0
    file_run_ok = True
    for chunk_path in chunk_paths:
        code, text, seconds = h.run([
            str(binary), "diarize", str(chunk_path),
            "--diar-model", str(h.MODEL_PATH),
            "--device", "cuda:0", "--format", "rttm",
            "--output", str(file_out / f"{chunk_path.stem}.rttm"),
            "--recording-id", "chunk0",
        ], cwd=h.REPO_DIR, timeout=3600)
        file_seconds += seconds
        if code != 0:
            file_run_ok = False
    file_bodies = {p.stem: body_of(p) for p in sorted(file_out.glob("*.rttm"))}

    outcomes = [{
        "variant": "file_sequential_same_chunks",
        "concurrency": 1,
        "returncode": 0 if file_run_ok else 1,
        "wall_seconds": round(file_seconds, 4),
        "audio_seconds": sum(h.wav_info(p)["seconds"] for p in chunk_paths),
        "rtf": round(file_seconds / total, 6) if total else None,
        "realtime_x": round(total / file_seconds, 3) if file_seconds else None,
        "files_produced": sorted(p.name for p in file_out.glob("*.rttm")),
        "note": "sum of 4 sequential single-file runs (own process each); baseline for chunk-vs-whole",
    }]
    print(f"[file/chunks] wall={file_seconds:.3f}s files={len(file_bodies)}", flush=True)

    glued_bodies: dict[str, str] = {}
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
        dir_bodies = {p.stem: body_of(p) for p in sorted(out_dir.glob("*.rttm"))}
        if concurrency == 1:
            glued_bodies = dir_bodies
        body_match = (
            set(dir_bodies) == set(file_bodies)
            and all(dir_bodies[k] == file_bodies[k] for k in file_bodies)
        ) if file_bodies and dir_bodies else False
        outcomes.append({
            "variant": f"dir_c{concurrency}",
            "concurrency": concurrency,
            "returncode": code,
            "wall_seconds": round(seconds, 4),
            "audio_seconds": audio_seconds,
            "rtf": round(seconds / audio_seconds, 6) if audio_seconds else None,
            "realtime_x": round(audio_seconds / seconds, 3) if seconds else None,
            "files_produced": produced,
            "bodies_match_file_baseline": body_match,
            "tail": text[-600:],
        })
        print(f"[concurrency={concurrency}] wall={seconds:.3f}s files={len(produced)}", flush=True)

    # Whole-file rerun of the same source (fixed recording id chunk0) so a
    # glued chunk timeline can be compared against uninterrupted streaming.
    whole_out = h.WORK_ROOT / "conc_whole.rttm"
    whole_out.unlink(missing_ok=True)
    code, text, seconds = h.run([
        str(binary), "diarize", str(source),
        "--diar-model", str(h.MODEL_PATH),
        "--device", "cuda:0", "--format", "rttm",
        "--output", str(whole_out),
        "--recording-id", "chunk0",
    ], cwd=h.REPO_DIR, timeout=3600)
    whole_body = body_of(whole_out) if whole_out.exists() else ""
    glued: list[str] = []
    for index in range(4):
        chunk_body = glued_bodies.get(f"chunk{index}", "")
        offset = index * chunk
        for line in chunk_body.splitlines():
            s, d, spk = line.split()
            glued.append(f"{float(s) + offset:.3f} {d} {spk}")
    glued_text = "\n".join(glued)
    glue_hash = hashlib.sha256(glued_text.encode()).hexdigest() if glued_text else None
    whole_hash = hashlib.sha256(whole_body.encode()).hexdigest() if whole_body else None
    chunk_vs_whole = {
        "whole_wall_seconds": round(seconds, 4),
        "whole_returncode": code,
        "glued_segments": len(glued),
        "whole_segments": len(whole_body.splitlines()) if whole_body else 0,
        "glued_body_sha256": glue_hash,
        "whole_body_sha256": whole_hash,
        "identical": bool(glue_hash) and glue_hash == whole_hash,
        "note": ("chunk timelines shifted by chunk offsets and concatenated, "
                 "compared against one uninterrupted run of the same source"),
    }
    print(f"[chunk-vs-whole] glued={len(glued)} whole={chunk_vs_whole['whole_segments']} "
          f"identical={chunk_vs_whole['identical']}", flush=True)

    dir_c1 = outcomes[1]["wall_seconds"] if len(outcomes) > 1 else None
    dir_c4 = outcomes[2]["wall_seconds"] if len(outcomes) > 2 else None
    return {
        "chunks": [str(p) for p in sorted(directory.glob("*.wav"))],
        "chunk_seconds": chunk,
        "source": str(source),
        "outcomes": outcomes,
        "chunk_vs_whole": chunk_vs_whole,
        "speedup_4_vs_1": (
            round(dir_c1 / dir_c4, 4) if dir_c1 and dir_c4 else None
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
    """Same input, repeated runs: are the segment bodies identical?

    Compares RTTM segment bodies (start/duration/speaker) with the RTTM
    recording-id field excluded: the harness passes a per-output recording id
    (``--recording-id <output.stem>``), so whole-file sha256 differs across
    runs by construction and can never answer the determinism question.
    Both views are reported; the body view is the verdict.
    """
    checks = []
    for entry in entries:
        hashes = [r["output_sha256"] for r in entry["runs"] if r["returncode"] == 0]
        body = entry.get("body_determinism") or {}
        segments = int(entry.get("rttm", {}).get("segments") or 0)
        informative = segments > 0 and len(hashes) > 1
        checks.append({
            "label": entry["label"],
            "runs": len(hashes),
            "segments": segments,
            "unique_hashes": len(set(hashes)),
            "identical": len(set(hashes)) == 1,
            "informative": informative,
            "sha256": hashes[0] if hashes else None,
            "body_unique_hashes": body.get("unique_body_hashes"),
            "body_identical": body.get("body_identical"),
            "body_segments_consistent": body.get("body_segments_consistent"),
            "body_sha256": body.get("body_sha256"),
        })
    informative = [c for c in checks if c["informative"]]
    return {
        "checks": checks,
        "informative_fixtures": len(informative),
        "all_identical": bool(informative) and all(c["identical"] for c in informative),
        "all_body_identical": bool(informative) and all(c.get("body_identical") for c in informative),
        "empty_output_fixtures": [c["label"] for c in checks if c["segments"] == 0],
        "note": ("whole-file RTTM equality (recording-id included) vs segment-body "
                 "equality (recording-id excluded); body view is the verdict; "
                 "empty-output fixtures are excluded"),
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
    real_entries = select_entries(gpu_entries, "real_")
    real_fixtures = [(e["label"], Path(str(e["audio"]))) for e in real_entries]
    if len(real_entries) >= 2:
        REPORT["length_scaling_fit_real_only"] = fit_scaling(real_entries)
    REPORT["determinism"] = determinism(gpu_entries)

    REPORT["gpu_offline_preset"] = run_offline(binary, real_fixtures or fixtures)
    REPORT["cpu_fallback"] = run_cpu(binary, real_fixtures or fixtures)

    if real_fixtures:
        REPORT["directory_concurrency"] = run_concurrency(binary, real_fixtures[0][1])

    # v6 determinism sweep: short-streaming (known-good control), mid-offline
    # (single forward, no AOSC loop) and mid-streaming (v5's drifter). Runs
    # AFTER the matrix so the timing numbers above stay comparable to v4/v5.
    det_cases: list[tuple[str, Path, str | None]] = []
    by_label = {e["label"]: e for e in gpu_entries}
    short_entries = [e for e in gpu_entries if e["label"].startswith("real_short")]
    mid_entries = [e for e in gpu_entries if e["label"].startswith("real_mid")]
    _ = by_label
    if short_entries:
        det_cases.append((
            "short_streaming",
            Path(str(short_entries[0]["audio"])), None))
    if mid_entries:
        det_cases.append((
            "mid_streaming",
            Path(str(mid_entries[0]["audio"])), None))
        det_cases.append((
            "mid_offline",
            Path(str(mid_entries[0]["audio"])), "offline"))
    if det_cases:
        REPORT["determinism_sweep"] = run_det_sweep(binary, det_cases)

    # v9 prob-stagger sweep: same audios, every rep dumps frame probabilities
    # so the verdict sees pre-hysteresis drift. Per-set rep counts come from
    # DET_SETS_V9 (mid_streaming runs 5); N_DET_REPEAT is only the fallback.
    # --offline is a CLI flag, not a preset: mid_offline_full carries the
    # sentinel preset and gets the flag here via extra_args-free direct call
    # (run_prob_sweep passes preset through; handle the flag set inline).
    if short_entries or mid_entries:
        prob_cases: list[tuple[str, Path, str | None]] = []
        audio_by_set = {
            "short_streaming": (short_entries[0] if short_entries else None, None),
            "mid_streaming": (mid_entries[0] if mid_entries else None, None),
            "mid_offline_full": (mid_entries[0] if mid_entries else None,
                                 OFFLINE_FLAG_SENTINEL),
            "mid_offline_preset": (mid_entries[0] if mid_entries else None, "offline"),
        }
        wanted = [(set_label, reps) for set_label, _, reps, _ in DET_SETS_V9]
        saved_repeat = N_DET_REPEAT
        prob_sweeps: list[dict[str, object]] = []
        for set_label, reps in wanted:
            slot = audio_by_set.get(set_label)
            if not slot or slot[0] is None:
                continue
            entry, set_preset = slot
            assert entry is not None
            globals()["N_DET_REPEAT"] = reps
            sweep = run_prob_sweep(
                binary, [(set_label, Path(str(entry["audio"])), set_preset)])
            prob_sweeps.append(sweep)
            exported = _export_parity_candidate(
                set_label, Path(str(entry["audio"])), set_preset, sweep)
            if exported is not None:
                REPORT.setdefault("parity_candidates", []).append(exported)
        globals()["N_DET_REPEAT"] = saved_repeat
        REPORT["prob_sweep"] = prob_sweeps

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
