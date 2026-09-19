#!/usr/bin/env python3
"""Official NeMo Sortformer inference benchmark (M3 baseline, G-D opponent).

Times the OFFICIAL PyTorch streaming inference path — the strongest
"official" form of this model — on the same audios the parity fixtures
pin, so the CUDA engine has a measured bar to beat (M3-CUDA-PLAN §5).

Modes per audio:
  streaming  : the exact sync streaming loop from
               dump_m2_reference.py (chunk 20 / fifo 80 / spkcache 160 /
               update 80, lc=rc=0 — the production preset), no hooks,
               no dumps; wall + per-chunk timings.
  offline    : best-effort model.transcribe on the whole file (different
               semantics; recorded but secondary).

Runs: 1 warmup pass (cuBLAS init, kernel JIT) then 3 timed streaming
passes; report median/mean/max. RTF = wall / audio_seconds.

Output: /kaggle/working/official_bench.json + stdout table.
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HF_REPO = "nvidia/diar_streaming_sortformer_4spk-v2"
TMP = Path("/tmp/obench")
OUT = Path("/kaggle/working")
GEOMETRY = {
    "chunk": 20, "lc": 0, "rc": 0,
    "fifo": 80, "spkcache": 160, "update_period": 80,
}
WANT_AUDIOS = {
    "short": "diar-smoke-audio/0-four-speakers-zh.wav",
    "mid": "diar-real-audio-5/video2_audio.wav",
}

REPORT = {
    "schema_version": 1,
    "job": "official_nemo_bench",
    "verdict": "not-run",
    "geometry": GEOMETRY,
    "audios": {},
}


def sh(argv, timeout=3600):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    tail = (p.stdout + "\n" + p.stderr)[-2000:]
    return p.returncode, tail


def find_audio(suffix):
    hits = sorted(Path("/kaggle/input").rglob(suffix))
    return hits[0] if hits else None


def torch_fingerprint():
    dev = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": dev.name,
        "cc": f"{dev.major}.{dev.minor}",
        "sm_count": dev.multi_processor_count,
        "mem_gb": round(dev.total_memory / 2**30, 1),
        "cudnn": torch.backends.cudnn.version(),
    }


def streaming_pass(model, sm, audio, device, collect_chunks=False):
    """One full sync streaming pass. Returns (wall_sec, n_chunks, chunk_ms)."""
    sig = torch.from_numpy(audio).unsqueeze(0).to(device)
    sig_len = torch.tensor([sig.shape[1]], device=device)
    chunk_ms = []
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)
        state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=device)
        offset = torch.zeros((1,), dtype=torch.long, device=device)
        sub = model.encoder.subsampling_factor
        n = 0
        for idx, chunk_feat, feat_lengths, left_off, right_off in sm.streaming_feat_loader(
            feat_seq=mel, feat_seq_length=mel_len, feat_seq_offset=offset
        ):
            tc = time.perf_counter()
            pre_embs, pre_lens = model.encoder.pre_encode(x=chunk_feat, lengths=feat_lengths)
            spk_len_before = state.spkcache.shape[1]
            fifo_len_before = state.fifo.shape[1]
            concat_embs = sm.concat_embs(
                [state.spkcache, state.fifo, pre_embs], dim=1, device=device
            )
            concat_lens = spk_len_before + fifo_len_before + pre_lens
            fc_embs, fc_lens = model.frontend_encoder(
                processed_signal=concat_embs,
                processed_signal_length=concat_lens,
                bypass_pre_encode=True,
            )
            preds = model.forward_infer(emb_seq=fc_embs, emb_seq_length=fc_lens)
            preds = sm.apply_mask_to_preds(preds, fc_lens)
            lc = round(left_off / sub)
            rc = math.ceil(right_off / sub)
            state, chunk_preds = sm.streaming_update(
                streaming_state=state, chunk=pre_embs, preds=preds, lc=lc, rc=rc
            )
            torch.cuda.synchronize()
            if collect_chunks:
                chunk_ms.append((time.perf_counter() - tc) * 1e3)
            n = idx + 1
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    return wall, n, chunk_ms


def main():
    TMP.mkdir(parents=True, exist_ok=True)

    code, pip_txt = sh(["pip", "install", "--quiet", "nemo_toolkit[asr]"])
    print(f"[pip] rc={code}", flush=True)
    if code != 0:
        REPORT["verdict"] = "pip-failed"
        REPORT["pip_tail"] = pip_txt[-800:]
        json.dump(REPORT, open(OUT / "official_bench.json", "w"), indent=1)
        return 0

    ckpt = TMP / "sortformer.nemo"
    if not ckpt.exists():
        # primary: filename verified by M2 Stage 0 (s0_a_nemo_files)
        fname = "diar_streaming_sortformer_4spk-v2.nemo"
        code, dl = sh([
            "curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
            "-o", str(ckpt),
            f"https://huggingface.co/{HF_REPO}/resolve/main/{fname}?download=true",
        ], timeout=1800)
        if code != 0:
            ckpt.unlink(missing_ok=True)
            # fallback: HF tree API, tolerant of non-JSON responses
            tcode, tree_txt = sh([
                "curl", "-L", "--fail", "--retry", "2", "--silent", "--show-error",
                f"https://huggingface.co/api/models/{HF_REPO}/tree/main?recursive=true",
            ], timeout=300)
            nemo_files = []
            if tcode == 0:
                try:
                    for entry in json.loads(tree_txt):
                        fp = str(entry.get("path", ""))
                        if fp.endswith(".nemo"):
                            nemo_files.append(fp)
                except ValueError:
                    print(f"[tree] non-JSON head={tree_txt[:120]!r}", flush=True)
            print(f"[tree] rc={tcode} nemo_files={nemo_files}", flush=True)
            if nemo_files:
                code, dl = sh([
                    "curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
                    "-o", str(ckpt),
                    f"https://huggingface.co/{HF_REPO}/resolve/main/{nemo_files[0]}?download=true",
                ], timeout=1800)
        print(f"[ckpt] rc={code} bytes={ckpt.stat().st_size if ckpt.exists() else 0}", flush=True)
        if code != 0 or not ckpt.exists():
            REPORT["verdict"] = "ckpt-download-failed"
            REPORT["dl_tail"] = dl[-500:]
            json.dump(REPORT, open(OUT / "official_bench.json", "w"), indent=1)
            return 0

    from nemo.collections.asr.models import SortformerEncLabelModel

    device = torch.device("cuda")
    model = SortformerEncLabelModel.restore_from(restore_path=str(ckpt), map_location=device)
    model.eval()
    model.to(device)
    sm = model.sortformer_modules
    sm.chunk_len = GEOMETRY["chunk"]
    sm.chunk_left_context = GEOMETRY["lc"]
    sm.chunk_right_context = GEOMETRY["rc"]
    sm.fifo_len = GEOMETRY["fifo"]
    sm.spkcache_len = GEOMETRY["spkcache"]
    sm.spkcache_update_period = GEOMETRY["update_period"]
    sm._check_streaming_parameters()
    model.preprocessor.featurizer.dither = 0.0

    REPORT["verdict"] = "ok"
    REPORT["fingerprint"] = torch_fingerprint()
    print("[env]", json.dumps(REPORT["fingerprint"]), flush=True)

    for name, rel in WANT_AUDIOS.items():
        path = find_audio(rel)
        if path is None:
            REPORT["audios"][name] = {"error": f"audio not found: {rel}"}
            continue
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        assert sr == 16000
        audio_sec = len(audio) / sr
        entry = {"path": str(path), "audio_sec": round(audio_sec, 2)}

        # warmup (uncollected) + 3 timed passes, per-chunk times on pass 3
        _, n_warm, _ = streaming_pass(model, sm, audio, device)
        walls = []
        for run in range(3):
            wall, n, cms = streaming_pass(
                model, sm, audio, device, collect_chunks=(run == 2)
            )
            walls.append(wall)
            if run == 2:
                entry["n_chunks"] = n
                entry["chunk_ms_mean"] = round(float(np.mean(cms)), 2)
                entry["chunk_ms_p50"] = round(float(np.percentile(cms, 50)), 2)
                entry["chunk_ms_max"] = round(float(np.max(cms)), 2)
        entry["walls_sec"] = [round(w, 2) for w in walls]
        med = float(np.median(walls))
        entry["wall_sec_median"] = round(med, 2)
        entry["rtf"] = round(med / audio_sec, 3)
        entry["realtime_factor_ok"] = med < audio_sec

        # offline best-effort (whole-file transcribe; different semantics)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.transcribe(audio=[str(path)], batch_size=1)
            torch.cuda.synchronize()
            entry["offline_wall_sec"] = round(time.perf_counter() - t0, 2)
            entry["offline_rtf"] = round(entry["offline_wall_sec"] / audio_sec, 3)
        except Exception as exc:  # noqa: BLE001
            entry["offline_error"] = repr(exc)[:300]

        REPORT["audios"][name] = entry
        print(f"[bench] {name}: {json.dumps(entry)}", flush=True)

    json.dump(REPORT, open(OUT / "official_bench.json", "w"), indent=1)
    print("[done] official_bench.json written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
