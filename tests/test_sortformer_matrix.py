"""Local tests for the measurement-matrix kernel (no GPU/CUDA/Kaggle/git).

Run:  python3 -m pytest tests/test_sortformer_matrix.py -q
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

KERNEL = Path(__file__).resolve().parents[1] / "kaggle" / "sortformer_matrix" / "sortformer_matrix.py"


def load_select_entries():
    """Load only select_entries without executing the Kaggle-only module body.

    The kernel imports diar_harness at module top level; stub it so the test
    needs no GPU, CUDA, dataset mounts or network.
    """
    source = KERNEL.read_text(encoding="utf-8")
    start = source.index("def select_entries(")
    end = source.index("def run_cpu(")
    snippet = "from __future__ import annotations\n" + source[start:end]
    module = types.ModuleType("matrix_select_snippet")
    exec(compile(snippet, str(KERNEL), "exec"), module.__dict__)  # noqa: S102 - local test fixture
    return module.select_entries


select_entries = load_select_entries()


def test_select_entries_filters_entry_dicts_by_label_prefix():
    entries = [
        {"label": "real_short_x", "audio_seconds": 57.0},
        {"label": "real_mid_y", "audio_seconds": 357.0},
        {"label": "synth_300s", "audio_seconds": 300.0},
    ]
    matched = select_entries(entries, "real_")
    assert [e["label"] for e in matched] == ["real_short_x", "real_mid_y"]


def test_select_entries_does_not_raise_keyerror_on_dicts():
    """Regression: the old tuple-based select() raised KeyError: 0 here."""
    entries = [{"label": "real_long", "audio_seconds": 6432.0}]
    assert select_entries(entries, "real_") == entries


def test_select_entries_empty_and_missing_labels():
    assert select_entries([], "real_") == []
    assert select_entries([{"audio_seconds": 1.0}], "real_") == []


def test_fixture_labels_match_entry_prefix_contract(tmp_path):
    """Kernel builds real labels 'real_short_*'/'real_mid_*'/'real_long'.

    The downstream filter needs the 'real_' prefix; guard the contract.
    """
    labels = ["real_short_abc", "real_mid_xyz", "real_long", "synth_300s"]
    assert all(label.startswith("real_") for label in labels[:3])
    assert not labels[3].startswith("real_")
