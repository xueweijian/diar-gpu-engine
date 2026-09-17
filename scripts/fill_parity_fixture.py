#!/usr/bin/env python3
"""Backfill parity L1 fixtures from kernel prob-stagger output (v12+).

Truth discipline (parity/README.md): fixtures are generated EXCLUSIVELY from
kernel output — never from engine output (anti-self-certification). This
script consumes a parity_candidates/<label>/ directory exported by the v12+
kernel into /kaggle/working and writes a schema-v1 fixture:

    <fixtures-dir>/<name>/manifest.json   (parity/manifest.py schema v1)
    <fixtures-dir>/<name>/probs.f32       (12B <qi header + f32 payload)

Validations before anything is written:
  * candidate.json exists, rep0 probs_available=True;
  * probs.f32 parses as probdump wire format AND its whole-file sha256 equals
    rep0's pinned probs_sha256 (the exact observation, not "some" dump);
  * only cross-session-stable labels, unless --allow-unstable is passed
    (mid cases churn across sessions — they are NOT legitimate truth).

frame_grid_ms: 80 is pinned from upstream geometry — FE hop 160 samples @16k
(10 ms) x conformer 8x subsampling; cross-checked against the streaming
presets (spkcache 160 = 12.8 s, fifo 80 = 6.4 s, chunk 20 = 1.6 s) and the
mid fixture audio (4467 frames x 80 ms = 357.4 s = the known 357 s clip).

Usage:
  python3 scripts/fill_parity_fixture.py <candidate_dir> \
      --name v12-short-streaming-r0 --kernel-version 12 \
      --src-commit $(git rev-parse --short HEAD) \
      [--fixtures-dir parity/fixtures] [--allow-unstable] [--frame-grid-ms 80]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parity.loader import load_case  # noqa: E402
from parity.manifest import load_manifest  # noqa: E402

KERNEL_NAME = "kaggle://weijianxue/diar-gpu-engine-sortformer-matrix"
PROBDUMP_HEADER = struct.Struct("<qi")


def parse_probdump(raw: bytes) -> tuple[int, int]:
    """Validate wire format, return (n_frames, n_spk). No float unpack here."""
    if len(raw) < PROBDUMP_HEADER.size:
        raise SystemExit(f"probs.f32: short header ({len(raw)} bytes)")
    n_frames, n_spk = PROBDUMP_HEADER.unpack(raw[: PROBDUMP_HEADER.size])
    if n_frames <= 0 or n_spk <= 0:
        raise SystemExit(f"probs.f32: bad shape frames={n_frames} spk={n_spk}")
    expect = PROBDUMP_HEADER.size + n_frames * n_spk * 4
    if len(raw) < expect:
        raise SystemExit(f"probs.f32: truncated ({len(raw)} < {expect} bytes)")
    if len(raw) != expect:
        raise SystemExit(f"probs.f32: trailing bytes ({len(raw)} != {expect})")
    return n_frames, n_spk


def gpu_label(snapshot: object) -> str:
    """Best-effort human GPU string from the kernel's gpu_snapshot dict."""
    if isinstance(snapshot, dict):
        for key in ("name", "gpu", "device", "gpu_name", "model"):
            value = snapshot.get(key)
            if isinstance(value, str) and value.strip():
                return value
        text = json.dumps(snapshot, sort_keys=True)
        if "NVIDIA" in text:
            for token in text.replace('"', " ").split():
                if "NVIDIA" in token:
                    return token
        return text[:120]
    return str(snapshot)[:120]


def fill(
    candidate_dir: Path,
    name: str,
    kernel_version: int,
    src_commit: str,
    fixtures_dir: Path,
    allow_unstable: bool = False,
    frame_grid_ms: int = 80,
) -> Path:
    prov_path = candidate_dir / "candidate.json"
    try:
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"{prov_path}: not found (is this a v12+ candidate dir?)")

    label = prov.get("label", candidate_dir.name)
    reps = prov.get("reps") or []
    if not reps or not reps[0].get("probs_available"):
        raise SystemExit(f"{label}: rep0 has no probs dump — refusing to fabricate truth")

    if not prov.get("cross_session_stable") and not allow_unstable:
        raise SystemExit(
            f"{label}: cross_session_stable is False (mid cases churn across "
            "sessions and are NOT legitimate truth). Pass --allow-unstable only "
            "with an explicit cross-session convergence story.")

    probs_path = candidate_dir / "probs.f32"
    raw = probs_path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    pinned_sha = reps[0].get("probs_sha256")
    if pinned_sha and pinned_sha != file_sha:
        raise SystemExit(
            f"{label}: probs.f32 sha {file_sha} != rep0 pin {pinned_sha} — "
            "candidate bytes are not the pinned observation")
    n_frames, n_spk = parse_probdump(raw)

    body_groups: dict[str, list[int]] = {}
    for rep in reps:
        body_groups.setdefault(str(rep.get("body_sha256")), []).append(int(rep.get("rep", 0)))
    hash_groups = {sha: idxs for sha, idxs in body_groups.items()}
    deterministic = len(hash_groups) == 1

    manifest = {
        "schema_version": 1,
        "name": name,
        "case": {
            "audio": prov.get("audio_path"),
            "preset": prov.get("preset"),
            "offline": bool(prov.get("offline_flag")),
            "geometry": None,
        },
        "source": {
            "kernel": KERNEL_NAME,
            "kernel_version": int(kernel_version),
            "src_commit": src_commit,
            "upstream_commit": str(prov.get("upstream_commit")),
            "captured_at": str(prov.get("captured_at")),
            "gpu": gpu_label(prov.get("gpu_before")),
            "repeats": len(reps),
            "env_pins": prov.get("env_pins"),
        },
        "observation": {
            "frame_grid_ms": int(frame_grid_ms),
            "n_frames": n_frames,
            "n_spk": n_spk,
            "deterministic": deterministic,
            "hash_groups": hash_groups,
        },
        "tensors": [{
            "name": "probs",
            "layer": "frame_probs",
            "dtype": "f32",
            "shape": [n_frames, n_spk],
            "nbytes": n_frames * n_spk * 4,
            "sha256": file_sha,
            "path": "probs.f32",
        }],
    }

    fixture_dir = fixtures_dir / name
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "probs.f32").write_bytes(raw)
    (fixture_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Immediate self-check: the manifest must parse (schema v1) and the payload
    # must survive the loader's integrity verification before we claim success.
    load_manifest(fixture_dir / "manifest.json")
    _manifest, tensors = load_case(fixture_dir)
    if "probs" not in tensors:
        raise SystemExit(f"{fixture_dir}: loader returned no probs tensor")

    print(f"[fill] {fixture_dir}: fixture written and loader-verified "
          f"(n_frames={n_frames}, n_spk={n_spk}, "
          f"deterministic={deterministic}, stable={prov.get('cross_session_stable')})")
    return fixture_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    parser.add_argument("--src-commit", required=True)
    parser.add_argument("--fixtures-dir", type=Path, default=REPO_ROOT / "parity" / "fixtures")
    parser.add_argument("--allow-unstable", action="store_true")
    parser.add_argument("--frame-grid-ms", type=int, default=80)
    args = parser.parse_args()
    fill(args.candidate_dir, args.name, args.kernel_version, args.src_commit,
         args.fixtures_dir, allow_unstable=args.allow_unstable,
         frame_grid_ms=args.frame_grid_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
