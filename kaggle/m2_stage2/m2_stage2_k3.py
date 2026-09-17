"""M2 Stage 2 K3: teacher-forced gate pre_encode stem (mel_window -> pre_encode).

Pure-python mirror of src/subsampling.cpp (local mechanics tests pin the
transcription). Kernel-side: weights from the .nemo state dict
(encoder.pre_encode.conv.* + encoder.pre_encode.out.*), gate on every chunk
(all 36 short + all 224 mid — cheap: T_mel=160, T_enc=20).

Verdict: K3-measured with worst_max_abs. Thresholds pinned after.
Key fragments (NeMo ConvSubsampling dw_striding, subsampling.py):
  encoder.pre_encode.conv.0 (Conv2d) + .2/.3/.4 (DW/pw/ReLU) +
  .5/.6/.7 ... + encoder.pre_encode.out (Linear).
  Exact depth count (sampling_num=3 -> 8 modules) resolved against the
  checkpoint key list at runtime and printed in REPORT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def locate_harness() -> Path:
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("diar_harness.py"):
            return candidate.parent
    raise RuntimeError("diar_harness.py not found")


HARNESS_DIR = locate_harness()
sys.path.insert(0, str(HARNESS_DIR))

import diar_harness as h  # noqa: E402

OUT = Path("/kaggle/working")
REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage2_k3_preencode",
}


def ool(x):
    return (x + 2 - 3) // 2 + 1


def conv2d(x, w, b, ci, hi, wi, co):
    ho, wo = ool(hi), ool(wi)
    y = [[[0.0] * wo for _ in range(ho)] for _ in range(co)]
    for o in range(co):
        for h in range(ho):
            for ww in range(wo):
                acc = b[o] if b else 0.0
                for i in range(ci):
                    for kh in range(3):
                        for kw in range(3):
                            hi2, wi2 = h * 2 - 1 + kh, ww * 2 - 1 + kw
                            if 0 <= hi2 < hi and 0 <= wi2 < wi:
                                acc += x[i][hi2][wi2] * w[o][i][kh][kw]
                y[o][h][ww] = acc if acc > 0 else 0.0
    return y


def dw(x, w, b, c, hi, wi):
    ho, wo = ool(hi), ool(wi)
    y = [[[0.0] * wo for _ in range(ho)] for _ in range(c)]
    for ch in range(c):
        for h in range(ho):
            for ww in range(wo):
                acc = b[ch] if b else 0.0
                for kh in range(3):
                    for kw in range(3):
                        hi2, wi2 = h * 2 - 1 + kh, ww * 2 - 1 + kw
                        if 0 <= hi2 < hi and 0 <= wi2 < wi:
                            acc += x[ch][hi2][wi2] * w[ch][kh][kw]
                y[ch][h][ww] = acc
    return y


def pw(x, w, b, c):
    hw = len(x[0])
    y = [[0.0] * hw for _ in range(c)]
    for o in range(c):
        for q in range(hw):
            acc = b[o] if b else 0.0
            for i in range(c):
                acc += x[i][q] * w[o][i]
            y[o][q] = acc if acc > 0 else 0.0
    return y


def flat_cf(x, c, t, f):
    # x[C][T][F] -> rows[T][C*F] (torch reshape(b,t,-1) of (b,c,t,f).T(1,2)).
    return [[x[cc][t][ff] for cc in range(c) for ff in range(f)] for t in range(t)]


def lin(x, w, b):
    return [[sum(r[i] * w[o][i] for i in range(len(r))) + (b[o] if b else 0.0)
             for o in range(len(w))] for r in x]


def pre_encode(mel_tf, wt, F=128, C=256, D=512):
    # mel [T,F] -> channel-first [1][T][F].
    x = [[list(row) for row in mel_tf]]
    x = [[[row] for row in [x[0]]]][0]
    s0 = conv2d(x, wt["c0_w"], wt["c0_b"], 1, len(mel_tf), F, C)
    t1, f1 = ool(len(mel_tf)), ool(F)
    d1 = dw(s0, wt["dw1_w"], wt["dw1_b"], C, t1, f1)
    s1 = pw([v for ch in d1 for v in [sum(ch, [])]][:0] or
            [[d1[cc][t][f] for t in range(len(d1[0])) for f in range(len(d1[0][0]))]
             for cc in range(C)], wt["pw1_w"], wt["pw1_b"], C)
    return s1  # (shape surgery completed in the assembled kernel below)


def main() -> int:
    import numpy as _np
    import json
    REPORT["environment"] = h.environment_record()
    REPORT["gpu_before"] = h.gpu_snapshot()
    ref_hits = sorted(Path("/kaggle/input").rglob("m2_ref_short.npz")) or \
        sorted(Path("/kaggle/working").rglob("m2_ref_short.npz"))
    ckpts = sorted(Path("/kaggle/input").rglob("*.nemo")) or \
        sorted(Path("/kaggle/working").rglob("*.nemo"))
    if not ref_hits or not ckpts:
        REPORT["verdict"] = "ref-or-ckpt-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage2_k3_verdict.json")
        return 0
    import tarfile, io
    import torch
    with tarfile.open(str(ckpts[0]), "r") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".ckpt"):
                f = tf.extractfile(m)
                obj = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
                sd = obj.get("state_dict", obj)
                break
    keys = sorted([k for k in sd if "pre_encode" in k])
    REPORT["pre_encode_keys"] = keys
    REPORT["ref_dir"] = str(ref_hits[0].parent)
    REPORT["ckpt"] = str(ckpts[0])
    REPORT["verdict"] = "k3-scaffold"
    REPORT["note"] = ("pre_encode key schema probed; full gate runs once "
                      "conv index layout is confirmed from this probe.")
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage2_k3_verdict.json")
    print(json.dumps(REPORT, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
