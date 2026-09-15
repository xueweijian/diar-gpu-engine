"""parity — M1/M2 ground-truth fixtures and comparison gates.

manifest.py: fixture manifest schema v1 (provenance-pinned, sha256-pinned).
loader.py: fixture loading with integrity enforcement.
gate.py: exact / tolerance comparison verdicts per layer.

Fixture truth comes only from kernel output (scripts/fill_parity_fixture.py);
nothing here generates fixtures from engine output (that would be
self-certification, the exact failure mode the AOSC/BirthGate/FE oracles
were built to avoid).
"""

from parity.manifest import SCHEMA_VERSION, Manifest, ManifestError, load_manifest
from parity.loader import FixtureIntegrityError, FrameProbsTensor, load_case
from parity.gate import FRAME_PROBS_TOLERANCE, frame_probs_gate

__all__ = [
    "SCHEMA_VERSION",
    "Manifest",
    "ManifestError",
    "load_manifest",
    "FixtureIntegrityError",
    "FrameProbsTensor",
    "load_case",
    "FRAME_PROBS_TOLERANCE",
    "frame_probs_gate",
]
