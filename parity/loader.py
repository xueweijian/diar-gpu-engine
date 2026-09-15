"""Load and integrity-check parity fixture payloads.

``load_case`` reads a fixture directory, re-derives every tensor's sha256
and refuses to return data whose hash or byte count disagrees with the
manifest. A fixture that fails here is corrupt or hand-edited — never
run the gate on it.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

from parity.manifest import Manifest, ManifestError, load_manifest


@dataclass(frozen=True)
class FrameProbsTensor:
    n_frames: int
    n_spk: int
    values: tuple[float, ...]

    def row(self, frame: int) -> tuple[float, ...]:
        return self.values[frame * self.n_spk:(frame + 1) * self.n_spk]


class FixtureIntegrityError(Exception):
    """Payload bytes disagree with the manifest pins."""


def _pinned_bytes(fixture_dir: Path, manifest: Manifest, tensor_name: str,
                  file_nbytes: int) -> bytes:
    pin = manifest.tensor(tensor_name)
    path = fixture_dir / pin.path
    if path.resolve().parent != fixture_dir.resolve():
        raise FixtureIntegrityError(f"{path}: tensor path escapes fixture directory")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise FixtureIntegrityError(f"{path}: missing payload") from None
    if len(raw) != file_nbytes:
        raise FixtureIntegrityError(
            f"{path}: {len(raw)} bytes, manifest pins {file_nbytes}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin.sha256:
        raise FixtureIntegrityError(f"{path}: sha256 {digest} != pinned {pin.sha256}")
    return raw


def load_case(fixture_dir: Path) -> tuple[Manifest, dict[str, FrameProbsTensor]]:
    """Verify and load all frame_probs tensors of a fixture directory.

    Returns (manifest, {tensor_name: FrameProbsTensor}). Raises
    ManifestError / FixtureIntegrityError; no partial results. Payloads are
    stored exactly as the kernel wrote them: frame_probs files keep the
    probdump wire format (12-byte <qi> header + f32 payload), so the file is
    pin.nbytes + 12 bytes on disk while the manifest's nbytes describes the
    logical tensor.
    """
    fixture_dir = Path(fixture_dir)
    manifest = load_manifest(fixture_dir / "manifest.json")
    tensors: dict[str, FrameProbsTensor] = {}
    for pin in manifest.tensors:
        if pin.layer == "frame_probs":
            raw = _pinned_bytes(fixture_dir, manifest, pin.name, pin.nbytes + 12)
            # Same wire format as the kernel probdump: <qi> header + f32
            # row-major payload. The manifest shape must agree with it.
            n_frames, n_spk = struct.unpack("<qi", raw[:12])
            if (n_frames, n_spk) != pin.shape[:2] or len(pin.shape) != 2:
                raise FixtureIntegrityError(
                    f"{fixture_dir / pin.path}: header ({n_frames}, {n_spk}) "
                    f"disagrees with manifest shape {list(pin.shape)}")
            obs_frames = manifest.observation["n_frames"]
            obs_spk = manifest.observation["n_spk"]
            if n_frames != obs_frames or n_spk != obs_spk:
                raise FixtureIntegrityError(
                    f"{fixture_dir / pin.path}: header ({n_frames}, {n_spk}) "
                    f"disagrees with observation ({obs_frames}, {obs_spk})")
            values = struct.unpack(f"<{n_frames * n_spk}f", raw[12:])
            tensors[pin.name] = FrameProbsTensor(
                n_frames=n_frames, n_spk=n_spk, values=values)
        else:  # pragma: no cover - v1 has only frame_probs
            raise ManifestError(f"unhandled layer {pin.layer!r}")
    return manifest, tensors
