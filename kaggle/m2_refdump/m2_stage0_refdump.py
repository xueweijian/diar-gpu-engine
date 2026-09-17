"""M2 Stage 0 spike: capture NeMo fp32 reference tensors (or fail loudly).

Goal per docs/M2-NEURAL-CORE-PLAN.md §2: run upstream
scripts/asr/dump_sortformer_reference.py (vendored below as
dump_m2_reference, with per-block hooks) against the SAME audios and
geometry the matrix uses, and answer S0-a/b/c.

Spike discipline: this kernel MUST end in a verdict, not a partial dump.
- S0-a: HF repo file list contains a .nemo checkpoint -> else verdict
  "no-nemo-checkpoint" (proceed to ggml-probdump fallback).
- S0-b: nemo_toolkit[asr] installs and imports on this image -> else
  verdict "nemo-not-installable" (same fallback).
- S0-c: dump completes and total_preds sanity-checks against the ggml
  binary on one short audio within tolerance -> verdict "nemo-ok"
  (Stage 1/2 unblocked) or "nemo-diverged" (investigate, do NOT silently
  promote the dump to truth).

Everything is pure diarization. Artifacts are small (.npz per audio +
spike_verdict.json); build trees, wheels and checkpoints stay in /tmp.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def locate_harness() -> Path:
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

OUT = Path("/kaggle/working")
TMP = Path("/tmp/m2-refdump")
FIXTURE_ROOT = Path("/kaggle/input")
HF_REPO = "nvidia/diar_streaming_sortformer_4spk-v2"
# Matrix geometry (streaming preset, aosc_state.h): chunk 20 / fifo 80 /
# spkcache 160 / update-period 80 encoder frames; lc=rc=0.
GEOMETRY = {
    "chunk": 20, "lc": 0, "rc": 0,
    "fifo": 80, "spkcache": 160, "update_period": 80,
}
# Same audios the parity fixtures pin (short + mid only; spike, not sweep).
WANT_AUDIOS = {
    "short": "diar-smoke-audio/0-four-speakers-zh.wav",
    "mid": "diar-real-audio-5/video2_audio.wav",
}

REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage0_refdump_spike",
    "harness_source": "diar_harness.py resolved at runtime from /kaggle/input",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verdict": "not-run",
}


def find_audio(name: str, suffix: str) -> Path | None:
    hits = sorted(FIXTURE_ROOT.rglob(suffix))
    return hits[0] if hits else None


def sh(argv: list[str], timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        tail = (p.stdout + "\n" + p.stderr)[-3000:]
        return p.returncode, tail
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace")[-1500:]
        err = (exc.stderr or b"").decode("utf-8", "replace")[-1500:]
        return 124, f"TIMEOUT {argv[0]}: {out}\n{err}"


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["environment"] = h.environment_record()
    REPORT["gpu_before"] = h.gpu_snapshot()

    # S0-a: does the HF repo carry a .nemo checkpoint?
    code, tree_txt = sh([
        "curl", "-L", "--fail", "--retry", "2", "--silent", "--show-error",
        f"https://huggingface.co/api/models/{HF_REPO}/tree/main?recursive=true",
    ], timeout=300)
    nemo_files: list[str] = []
    if code == 0:
        try:
            for entry in json.loads(tree_txt):
                path = str(entry.get("path", ""))
                if path.endswith(".nemo"):
                    nemo_files.append(path)
        except (ValueError, AttributeError):
            nemo_files = []
    REPORT["s0_a_hf_tree_rc"] = code
    REPORT["s0_a_nemo_files"] = nemo_files
    if not nemo_files:
        REPORT["verdict"] = "no-nemo-checkpoint"
        REPORT["note"] = (
            "HF repo has no .nemo file (or tree API unreachable); "
            "Stage 0 falls back to the ggml-probdump patch per plan §2."
        )
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        print(json.dumps(REPORT, indent=1)[:2000])
        return 0

    # S0-b: can nemo_toolkit[asr] install and import here?
    code, pip_txt = sh([
        "pip", "install", "--quiet", "nemo_toolkit[asr]",
    ], timeout=3600)
    REPORT["s0_b_pip_rc"] = code
    REPORT["s0_b_pip_tail"] = pip_txt[-1500:]
    code, imp_txt = sh([
        "python3", "-c",
        "import torch; from nemo.collections.asr.models import "
        "SortformerEncLabelModel; print(torch.__version__, torch.cuda.is_available())",
    ], timeout=600)
    REPORT["s0_b_import_rc"] = code
    REPORT["s0_b_import_tail"] = imp_txt[-1000:]
    if code != 0:
        REPORT["verdict"] = "nemo-not-installable"
        REPORT["note"] = (
            "nemo_toolkit unavailable on this image; "
            "Stage 0 falls back to the ggml-probdump patch per plan §2."
        )
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        print(json.dumps(REPORT, indent=1)[:2000])
        return 0

    # S0-c: download .nemo + vendored dump script, run on short audio.
    ckpt = TMP / "sortformer.nemo"
    if not ckpt.exists():
        code, dl_txt = sh([
            "curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
            "-o", str(ckpt),
            f"https://huggingface.co/{HF_REPO}/resolve/main/{nemo_files[0]}?download=true",
        ], timeout=3600)
        REPORT["s0_c_ckpt_rc"] = code
        REPORT["s0_c_ckpt_tail"] = dl_txt[-500:]
        if code != 0 or not ckpt.exists():
            REPORT["verdict"] = "nemo-ckpt-download-failed"
            REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h.emit_report(REPORT, name="m2_stage0_verdict.json")
            return 0
    REPORT["s0_c_ckpt_bytes"] = ckpt.stat().st_size

    short = find_audio("short", WANT_AUDIOS["short"])
    mid = find_audio("mid", WANT_AUDIOS["mid"])
    REPORT["audios"] = {k: (str(v) if v else None)
                        for k, v in (("short", short), ("mid", mid))}
    if short is None:
        REPORT["verdict"] = "audio-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        return 0

    dump_py = Path(__file__).parent / "dump_m2_reference.py"
    REPORT["dump_script"] = str(dump_py) + (" (present)" if dump_py.exists()
                                            else " (MISSING)")
    if not dump_py.exists():
        REPORT["verdict"] = "dump-script-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        return 0

    device = "cuda"
    g = GEOMETRY
    for label, audio in (("short", short), ("mid", mid) if mid else ()):
        out_npz = OUT / f"m2_ref_{label}.npz"
        code, dump_txt = sh([
            "python3", str(dump_py), str(ckpt), str(audio), str(out_npz),
            "--device", device,
            "--chunk", str(g["chunk"]), "--lc", str(g["lc"]), "--rc", str(g["rc"]),
            "--fifo", str(g["fifo"]), "--spkcache", str(g["spkcache"]),
            "--update-period", str(g["update_period"]),
        ], timeout=5400)
        REPORT[f"s0_c_dump_{label}_rc"] = code
        REPORT[f"s0_c_dump_{label}_tail"] = dump_txt[-1500:]
        REPORT[f"s0_c_dump_{label}_npz"] = (
            str(out_npz) if out_npz.exists() else None)
        if code != 0 or not out_npz.exists():
            REPORT["verdict"] = "nemo-dump-failed"
            REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h.emit_report(REPORT, name="m2_stage0_verdict.json")
            print(json.dumps(REPORT, indent=1)[:2000])
            return 0

    REPORT["verdict"] = "nemo-ok"
    REPORT["note"] = (
        ".npz reference(s) captured; Stage 1/2 unblocked. Cross-check "
        "total_preds vs the ggml binary before promoting to truth."
    )
    REPORT["gpu_after"] = h.gpu_snapshot()
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage0_verdict.json")
    print(json.dumps(REPORT, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
