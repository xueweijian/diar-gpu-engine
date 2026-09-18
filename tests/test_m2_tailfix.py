"""Mechanics for the tail-fixture kernel (kaggle/m2_tailfix).

No Kaggle, no torch, no weights: pins the payload's static contract —
the vendored NeMo dump script is embedded byte-equal to its single source
of truth (m2_refdump/dump_m2_reference.py), the slice constants stay
inside the mid audio, and the kernel metadata references real datasets.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TAILFIX = REPO / "kaggle" / "m2_tailfix"
KERNEL = TAILFIX / "m2_tail_fixture.py"
DUMP_SRC = REPO / "kaggle" / "m2_refdump" / "dump_m2_reference.py"

# mid audio (video2_audio.wav) duration, from the m2-ref npz (357.286875 s)
MID_AUDIO_SECONDS = 357.286875


def _module_assign(name: str):
    text = KERNEL.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == name):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {KERNEL.name}")


def test_embedded_dump_matches_single_source():
    """One dump script, no fork drift: embedded == refdump's source file."""
    blob = _module_assign("EMBEDDED_DUMP_M2_REFERENCE")
    assert blob == DUMP_SRC.read_text(encoding="utf-8")
    assert "_capture_block_outputs" in blob


def test_slice_constants_are_inside_mid_audio():
    start = _module_assign("SLICE_START_SEC")
    length = _module_assign("SLICE_LENGTH_SEC")
    assert 0 < start < MID_AUDIO_SECONDS
    assert 20.0 <= length <= 120.0
    # the whole slice must exist in the source audio
    assert start + length <= MID_AUDIO_SECONDS
    # 60 s @ 16 kHz mono int16 = 1 920 000 bytes (the fixture audio size)
    assert length == 60.0


def test_geometry_matches_the_streaming_preset():
    g = _module_assign("GEOMETRY")
    assert g == {"chunk": 20, "lc": 0, "rc": 0, "fifo": 80, "spkcache": 160,
                 "update_period": 80}


def test_kernel_metadata_consistent():
    meta = json.loads((TAILFIX / "kernel-metadata.json").read_text())
    assert meta["id"] == "weijianxue/diar-m2-tail-fixture"
    assert (TAILFIX / meta["code_file"]).exists()
    assert "weijianxue/diar-gpu-engine-harness" in meta["dataset_sources"]
    assert "weijianxue/diar-real-audio-5" in meta["dataset_sources"]
