# M3 Step 6 kernel — engine-level integration + diar-bench acceptance
# (docs/M3-STEP6-ACCEPTANCE-PLAN.md). Kaggle GPU (T4, CUDA 12.8).
#
# What this kernel proves, for the first time, with the REAL q8 GGUF through
# the GPU (steps 3-5 gated operators in harness geometry only):
#   G-B   fixtures x routes {cpu, fp32, fp16} vs the M1 fixture probs,
#         K6 tiers verbatim (hard for v12 fixtures, advisory for v13-mid)
#   G-B2  route-vs-cpu same-engine diff (fp32 route vs cpu route on the
#         same fixture + face): the sharpest integration gate — v1 measures
#         the band, provisional gate at K6 tier, v2 calibrates (step-5
#         sqrt-walk discipline)
#   G-C   determinism: reps bit-identical per route (G-C gate in diar-bench)
#   G-D   wall ms/chunk per route vs official T4 45.5 ms (hard <= 25)
#   G-E   probs dumps emitted for the user-card three-way diff (G-E proper
#         runs when P100/V100 rows arrive; this kernel emits the T4 row)
# Fatbin check: diar-bench built with all 4 gencode; cuobjdump verifies
# 3 SASS + PTX on the exact binary that ran (the §6 deliverable shape).
import base64, json, os, subprocess, sys, time, wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

EMBED = {}
#__EMBED_TABLE__

WORK = "/tmp/step6"
os.makedirs(WORK + "/include/diar", exist_ok=True)
os.makedirs(WORK + "/src", exist_ok=True)
os.makedirs(WORK + "/tools", exist_ok=True)

FILES = {
    "hpp_backend":        WORK + "/include/diar/backend.hpp",
    "hpp_backend_cuda":   WORK + "/include/diar/backend_cuda.hpp",
    "hpp_encoder_cuda":   WORK + "/include/diar/encoder_cuda.hpp",
    "hpp_cublas_layout":  WORK + "/include/diar/cublas_layout.hpp",
    "hpp_nn":             WORK + "/include/diar/nn.hpp",
    "hpp_gguf":           WORK + "/include/diar/gguf.hpp",
    "hpp_sortformer":     WORK + "/include/diar/sortformer.hpp",
    "hpp_engine":         WORK + "/include/diar/engine.hpp",
    "hpp_diar":           WORK + "/include/diar/diar.hpp",
    "hpp_tailfix":        WORK + "/include/diar/tailfix.hpp",
    "hpp_mha":            WORK + "/include/diar/mha.hpp",
    "hpp_conv":           WORK + "/include/diar/conv.hpp",
    "hpp_conformer":      WORK + "/include/diar/conformer.hpp",
    "hpp_layers":         WORK + "/include/diar/layers.hpp",
    "hpp_posenc":         WORK + "/include/diar/posenc.hpp",
    "hpp_subsampling":    WORK + "/include/diar/subsampling.hpp",
    "hpp_profile":        WORK + "/include/diar/profile.hpp",
    "cpp_backend":        WORK + "/src/backend.cpp",
    "cpp_cublas_layout":  WORK + "/src/cublas_layout.cpp",
    "cpp_backend_cuda":   WORK + "/src/backend_cuda.cpp",
    "cpp_encoder_cuda":   WORK + "/src/encoder_cuda.cpp",
    "cpp_nn":             WORK + "/src/nn.cpp",
    "cpp_gguf":           WORK + "/src/gguf.cpp",
    "cpp_sortformer":     WORK + "/src/sortformer.cpp",
    "cpp_engine":         WORK + "/src/engine.cpp",
    "cpp_fe":             WORK + "/src/fe.cpp",
    "cpp_diar":           WORK + "/src/diar.cpp",
    "cpp_aosc":           WORK + "/src/aosc.cpp",
    "cpp_birth_gate":     WORK + "/src/birth_gate.cpp",
    "cpp_tailfix":        WORK + "/src/tailfix.cpp",
    "cpp_profile":        WORK + "/src/profile.cpp",
    "cpp_mha":            WORK + "/src/mha.cpp",
    "cpp_conv":           WORK + "/src/conv.cpp",
    "cpp_conformer":      WORK + "/src/conformer.cpp",
    "cpp_layers":         WORK + "/src/layers.cpp",
    "cpp_posenc":         WORK + "/src/posenc.cpp",
    "cpp_subsampling":    WORK + "/src/subsampling.cpp",
    "cpp_bench_main":     WORK + "/tools/diar_bench_main.cpp",
}
for key, path in FILES.items():
    with open(path, "wb") as f:
        f.write(base64.b64decode(EMBED[key]))
print("[s6k] sources materialized:", len(FILES))

REPORT = {"gates": {}, "notes": [], "cases": {}}
T0 = time.time()

GENCODE = ["-gencode", "arch=compute_60,code=sm_60",
           "-gencode", "arch=compute_70,code=sm_70",
           "-gencode", "arch=compute_75,code=sm_75",
           "-gencode", "arch=compute_60,code=compute_60"]
SRC_GPU = [FILES[k] for k in ("cpp_backend_cuda", "cpp_encoder_cuda")]
SRC_CPU = [FILES[k] for k in (
    "cpp_backend", "cpp_cublas_layout", "cpp_nn", "cpp_gguf", "cpp_sortformer",
    "cpp_engine", "cpp_fe", "cpp_diar", "cpp_aosc", "cpp_birth_gate",
    "cpp_tailfix", "cpp_profile", "cpp_mha", "cpp_conv", "cpp_conformer",
    "cpp_layers", "cpp_posenc", "cpp_subsampling")]

def sh(cmd, timeout=3600, check=True):
    print("[s6k] $", " ".join(cmd[:8]), "...", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    print(f"[s6k] rc={p.returncode} ({time.time()-t0:.1f}s)", flush=True)
    if p.stdout.strip():
        print(p.stdout[-2500:], flush=True)
    if p.returncode != 0 and check:
        if p.stderr.strip():
            print(p.stderr[-3000:], flush=True)
        REPORT["notes"].append(f"FAILED: {' '.join(cmd[:6])} rc={p.returncode}")
        raise SystemExit(3)
    return p

# 1) CPU selftest of diar-bench (wire round-trip + metrics oracles)
sh(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", WORK + "/include", "-O2",
    FILES["cpp_bench_main"], *SRC_CPU, "-o", WORK + "/diar_bench_cpu"])
st = sh([WORK + "/diar_bench_cpu", "--selftest"])
REPORT["gates"]["cpu_selftest"] = "OK" in st.stdout

# 2) nvcc build: the deliverable fatbin shape (3 SASS + PTX), profiled build
BENCH = WORK + "/diar_bench"
sh(["nvcc", "-O2", "-std=c++17", "-DDIAR_WITH_CUDA", "-DDIAR_PROFILE_STAGE",
    "-I", WORK + "/include", *GENCODE,
    "-x", "cu", FILES["cpp_backend_cuda"],
    "-x", "cu", FILES["cpp_encoder_cuda"],
    "-x", "cu", FILES["cpp_bench_main"],
    *SRC_CPU, "-lcublas", "-o", BENCH])

# 3) fatbin verification on the exact binary that will run
elf = sh(["cuobjdump", "--list-elf", BENCH]).stdout
ptx = sh(["cuobjdump", "--list-ptx", BENCH]).stdout
REPORT["gates"]["fatbin_3sass"] = all(a in elf for a in ("sm_60", "sm_70", "sm_75"))
REPORT["gates"]["fatbin_ptx60"] = ("sm_60" in ptx) or ("compute_60" in ptx)

# 4) weights + fixtures + audio
HF_REPO = "nvidia/diar_streaming_sortformer_4spk-v2"
GGUF_FILE = "diar_streaming_sortformer_4spk-v2.q8_0.gguf"

def ensure_gguf() -> Path:
    hits = sorted(Path("/kaggle/input").rglob(GGUF_FILE))
    if hits:
        return hits[0]
    dest = Path(WORK) / GGUF_FILE
    if dest.exists():
        return dest
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{GGUF_FILE}?download=true"
    print("[s6k] downloading q8 GGUF ...", flush=True)
    p = subprocess.run(["curl", "-sSL", "--retry", "5", "-o", str(dest), url],
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0 or not dest.exists() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"gguf download failed rc={p.returncode}")
    print(f"[s6k] gguf ready: {dest.stat().st_size} bytes", flush=True)
    return dest

def find_fixtures_dir() -> Path:
    for cand in sorted(Path("/kaggle/input").rglob("manifest.json")):
        d = cand.parent
        if (d / "probs.f32").exists() or any(d.glob("*/probs.f32")):
            return d.parent if (d / "probs.f32").exists() else d
    raise RuntimeError("fixtures dir not found under /kaggle/input")

def case_dirs(root: Path) -> dict:
    out = {}
    for m in sorted(root.rglob("manifest.json")):
        out[m.parent.name] = m.parent
    return out

def decode_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if rate != 16000 or nch != 1 or sw != 2:
        raise RuntimeError(f"{path.name}: expected 16k mono i16, got {rate}/{nch}/{sw}")
    return np.ascontiguousarray(
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)

def find_audio(basename: str, roots: list) -> Path:
    for root in roots:
        hits = sorted(Path(root).rglob(basename))
        if hits:
            return hits[0]
    raise RuntimeError(f"audio {basename} not found")

GGUF = ensure_gguf()
FIXROOT = find_fixtures_dir()
CASES = case_dirs(FIXROOT)
print("[s6k] fixtures:", sorted(CASES), flush=True)
AUDIO_ROOTS = ["/kaggle/input"]

# fixture face/mode mapping (K6 CASE_PLAN discipline)
def classify(manifest: dict):
    case = manifest["case"]
    if case.get("offline"):
        return ("full-offline", "pregate")
    if case.get("preset"):
        return ("offline-preset", "postgate")
    return ("streaming", "postgate")

# K6 tiers verbatim (M2 Stage 3 closure): v12 hard, v13-mid advisory
HARD = {"max_abs": 0.05, "mean_abs": 0.005, "frame_agreement": 0.999}
ADV = {"max_abs": 1.0, "mean_abs": 0.02, "frame_agreement": 0.99}
ADVISORY = {"v13-mid-streaming-r0", "v13-mid-offline-preset-r0"}
G_D_GATE_MS = 25.0
OFFICIAL_T4_MS = 45.5

def run_bench(tag: str, mode: str, route: str, audio_f32: Path,
              ref: Path | None, ref_face: str, reps: int) -> dict:
    prefix = str(Path(WORK) / tag)
    cmd = [BENCH, "--weights", str(GGUF), "--audio", str(audio_f32),
           "--mode", mode, "--route", route, "--reps", str(reps),
           "--label", tag, "--out", prefix]
    if ref is not None:
        cmd += ["--ref", str(ref), "--ref-face", ref_face]
    sh(cmd, timeout=6 * 3600)
    with open(prefix + ".bench.json") as f:
        return json.load(f)

def case_job(label: str, fdir: Path) -> dict:
    """One fixture: cpu route (baseline, slow) then routed runs vs it."""
    manifest = json.loads((fdir / "manifest.json").read_text())
    mode, face = classify(manifest)
    pin = next(t for t in manifest["tensors"] if t["layer"] == "frame_probs")
    ref_fixture = fdir / pin["path"]
    pcm = decode_wav(find_audio(Path(manifest["case"]["audio"]).name, AUDIO_ROOTS))
    audio_f32 = Path(WORK) / f"{label}.f32"
    audio_f32.write_bytes(pcm.tobytes())

    out = {"mode": mode, "face": face, "routes": {}}
    routes = [("cpu", 2)] if mode == "full-offline" else \
             [("cpu", 2), ("fp32", 3), ("fp16", 3)]
    cpu_probs = None
    for route, reps in routes:
        ref = ref_fixture  # every route compares against the FIXTURE
        bj = run_bench(f"{label}.{route}", mode, route, audio_f32, ref, face, reps)
        entry = {"bench": bj, "vs_fixture": bj.get("vs_ref")}
        if route == "cpu":
            cpu_probs = Path(WORK) / f"{label}.cpu.postgate.f32"
            if face == "pregate":
                cpu_probs = Path(WORK) / f"{label}.cpu.pregate.f32"
        else:
            # route-vs-cpu: compare the routed dump against the cpu dump
            dump = Path(WORK) / f"{label}.{route}.{face}.f32"
            entry["vs_cpu_max_abs"] = wire_diff(dump, cpu_probs)
        out["routes"][route] = entry
    # G-E artifact: the fp32 streaming postgate dump goes to /kaggle/working
    if mode != "full-offline":
        src = Path(WORK) / f"{label}.fp32.{face}.f32"
        if src.exists():
            import shutil
            shutil.copy(src, f"/kaggle/working/ge_t4_{label}_{face}.f32")
    return out

def wire_diff(a: Path, b: Path) -> float:
    import struct as _s
    def load(p: Path):
        raw = p.read_bytes()
        n, spk = _s.unpack("<qi", raw[:12])
        return np.frombuffer(raw, dtype="<f4", count=n * spk, offset=12)
    ra, rb = load(a), load(b)
    n = min(len(ra), len(rb))
    return float(np.abs(ra[:n].astype(np.float64) - rb[:n].astype(np.float64)).max())

# 5) run the four fixtures in parallel (CPU baselines dominate; K6 shape)
JOBS = 4
with ThreadPoolExecutor(max_workers=JOBS) as ex:
    futs = {ex.submit(case_job, label, fdir): label
            for label, fdir in CASES.items()}
    for fut in as_completed(futs):
        label = futs[fut]
        try:
            REPORT["cases"][label] = fut.result()
            print(f"[s6k] case {label} done", flush=True)
        except Exception as e:  # noqa: BLE001 — record, keep verdict loud
            REPORT["cases"][label] = {"error": repr(e)}
            print(f"[s6k] case {label} ERROR {e!r}", flush=True)

# 6) verdict assembly (pure, test-visible logic below in judge())
def judge(cases: dict) -> tuple[str, list[str]]:
    reasons = []
    for label, c in sorted(cases.items()):
        if "error" in c:
            reasons.append(f"{label}: {c['error']}")
            continue
        tier = ADV if label in ADVISORY else HARD
        for route, entry in sorted(c["routes"].items()):
            b = entry["bench"]
            if not b.get("determinism_bit_identical", False):
                reasons.append(f"{label}/{route}: determinism red")
            vf = entry.get("vs_fixture")
            if vf is None:
                reasons.append(f"{label}/{route}: no vs_fixture metrics")
                continue
            # fixture gates: cpu + fp32 at the K6 tier; fp16 advisory in v1
            # (promotion to default is the K6-hard gate — judged separately)
            t = tier if route != "fp16" else \
                {"max_abs": max(tier["max_abs"], 0.2),
                 "mean_abs": max(tier["mean_abs"], 0.02),
                 "frame_agreement": min(tier["frame_agreement"], 0.99)}
            if vf["max_abs"] > t["max_abs"]:
                reasons.append(f"{label}/{route}: vs_fixture max_abs="
                               f"{vf['max_abs']:.4g} > {t['max_abs']}")
            if vf["mean_abs"] > t["mean_abs"]:
                reasons.append(f"{label}/{route}: vs_fixture mean_abs="
                               f"{vf['mean_abs']:.4g} > {t['mean_abs']}")
            if vf["frame_agreement"] < t["frame_agreement"]:
                reasons.append(f"{label}/{route}: frame_agreement="
                               f"{vf['frame_agreement']:.4f}")
            # route hygiene: a routed run must never fall back to CPU
            if route != "cpu" and b.get("route_refused", 0) != 0:
                reasons.append(f"{label}/{route}: route_refused="
                               f"{b.get('route_refused')}")
            # G-D: routed wall ms/chunk (skip full-offline, always cpu)
            if route != "cpu":
                mpc = b.get("ms_per_chunk_best", 1e9)
                if mpc > G_D_GATE_MS:
                    reasons.append(f"{label}/{route}: {mpc:.2f} ms/chunk "
                                   f"> {G_D_GATE_MS}")
            # G-B2 provisional: fp32 route vs cpu route (K6 tier in v1)
            if route == "fp32" and "vs_cpu_max_abs" in entry:
                if entry["vs_cpu_max_abs"] > HARD["max_abs"]:
                    reasons.append(f"{label}/fp32: vs_cpu max_abs="
                                   f"{entry['vs_cpu_max_abs']:.4g}")
    return ("s6-green" if not reasons else "s6-red"), reasons

verdict, reasons = judge(REPORT["cases"])
REPORT["verdict"] = verdict
REPORT["reasons"] = reasons
REPORT["gates"]["g_d_official_ms"] = OFFICIAL_T4_MS

with open("/kaggle/working/step6_verdict.json", "w") as f:
    json.dump(REPORT, f, indent=1)
print("[s6k] VERDICT:", verdict, flush=True)
for r in reasons:
    print("[s6k] reason:", r, flush=True)
print(f"[s6k] wall {time.time()-T0:.0f}s", flush=True)
