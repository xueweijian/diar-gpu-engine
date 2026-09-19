# M3 Step 3 kernel — real CUDA compile + parity + launch + SGEMM timing.
# Kaggle GPU (T4, CUDA 12.8). Embedded sources: 5 headers + 5 impls + harness.
# Gates: G-S3a SASS parity <=1e-5 (5 shapes x3 runs, via backend::Context),
# G-S3b compute_60 PTX JIT parity <=1e-5, G-S3c launch us recorded,
# G-S3d SGEMM medians recorded, G-S3e fuse parity <=1e-5.
import base64, json, os, subprocess, sys, time

EMBED = {}
#__EMBED_TABLE__

WORK = "/tmp/step3"
os.makedirs(WORK + "/include/diar", exist_ok=True)
os.makedirs(WORK + "/src", exist_ok=True)
os.makedirs(WORK + "/tools", exist_ok=True)

FILES = {
    "hpp_backend":       WORK + "/include/diar/backend.hpp",
    "hpp_cublas_layout": WORK + "/include/diar/cublas_layout.hpp",
    "hpp_backend_cuda":  WORK + "/include/diar/backend_cuda.hpp",
    "hpp_nn":            WORK + "/include/diar/nn.hpp",
    "hpp_gguf":          WORK + "/include/diar/gguf.hpp",
    "cpp_cublas_layout": WORK + "/src/cublas_layout.cpp",
    "cpp_backend_cuda":  WORK + "/src/backend_cuda.cpp",
    "cpp_nn":            WORK + "/src/nn.cpp",
    "cpp_gguf":          WORK + "/src/gguf.cpp",
    "cpp_bench_main":    WORK + "/tools/step3_bench_main.cpp",
}
for key, path in FILES.items():
    with open(path, "wb") as f:
        f.write(base64.b64decode(EMBED[key]))
print("[s3k] sources materialized:", len(FILES))

REPORT = {"gates": {}, "notes": []}
T0 = time.time()

def sh(cmd, timeout=1200, check=True):
    print("[s3k] $", " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    print(f"[s3k] rc={p.returncode} ({dt:.1f}s)", flush=True)
    if p.stdout.strip():
        print(p.stdout[-3000:], flush=True)
    if p.returncode != 0 and check:
        if p.stderr.strip():
            print(p.stderr[-3000:], flush=True)
        REPORT["notes"].append(f"FAILED: {' '.join(cmd[:6])}... rc={p.returncode}")
        raise SystemExit(3)
    return p

# 1) CPU selftest (harness logic must be green before burning GPU time)
sh(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", WORK + "/include", "-O2",
    FILES["cpp_bench_main"], FILES["cpp_cublas_layout"], FILES["cpp_nn"],
    FILES["cpp_gguf"], "-o", WORK + "/selftest_cpu"])
st = sh([WORK + "/selftest_cpu"])
cpu_lines = [l for l in st.stdout.splitlines() if l.startswith("[s3]")]
REPORT["gates"]["cpu_selftest"] = all('"pass":true' in l for l in cpu_lines if '"parity_cpu"' in l)
print("[s3k] cpu selftest gates:", REPORT["gates"]["cpu_selftest"])

# 2) nvcc SASS fatbin: sm_60/70/75 + compute_60 PTX
GENCODE = ["-gencode", "arch=compute_60,code=sm_60",
           "-gencode", "arch=compute_70,code=sm_70",
           "-gencode", "arch=compute_75,code=sm_75",
           "-gencode", "arch=compute_60,code=compute_60"]
sh(["nvcc", "-O3", "-std=c++17", "-DDIAR_WITH_CUDA", "-I", WORK + "/include",
    *GENCODE,
    FILES["cpp_bench_main"], FILES["cpp_cublas_layout"],
    FILES["cpp_backend_cuda"], FILES["cpp_nn"], FILES["cpp_gguf"],
    "-lcublas", "-o", WORK + "/bench_sass"], timeout=1200)

# verify the fatbin really carries 3 SASS + 1 PTX
elf = sh(["cuobjdump", "--list-elf", WORK + "/bench_sass"], check=False).stdout
ptx = sh(["cuobjdump", "--list-ptx", WORK + "/bench_sass"], check=False).stdout
def _archs(text, prefix):
    toks = {t.strip(",;:") for t in text.split()}
    return sorted(t for t in toks if t.startswith(prefix))
sass_archs = _archs(elf, "sm_")
ptx_archs = _archs(ptx, "compute_")
print("[s3k] SASS archs in fatbin:", sass_archs)
print("[s3k] PTX archs in fatbin:", ptx_archs)
REPORT["gates"]["fatbin_sass_all3"] = all(a in sass_archs for a in ("sm_60", "sm_70", "sm_75"))
REPORT["gates"]["fatbin_ptx_compute60"] = "compute_60" in ptx_archs

# 3) GPU bench (SASS path): parity + launch + sgemm + fuse
b1 = sh([WORK + "/bench_sass", "sass"])
parsed = []
for l in b1.stdout.splitlines():
    if l.startswith("[s3] "):
        try:
            parsed.append(json.loads(l[5:]))
        except Exception:
            pass
REPORT["bench_sass"] = parsed
par = [r for r in parsed if r.get("k") == "parity_sass"]
REPORT["gates"]["g_s3a_parity"] = bool(par) and all(r.get("pass") for r in par)
fuse = [r for r in parsed if r.get("k") == "fuse_residual"]
REPORT["gates"]["g_s3e_fuse"] = bool(fuse) and all(r.get("pass") for r in fuse)

# 4) PTX-only binary (compute_60 PTX; JIT-compiles to sm_75 at load)
sh(["nvcc", "-O3", "-std=c++17", "-DDIAR_WITH_CUDA", "-I", WORK + "/include",
    "-gencode", "arch=compute_60,code=compute_60",
    FILES["cpp_bench_main"], FILES["cpp_cublas_layout"],
    FILES["cpp_backend_cuda"], FILES["cpp_nn"], FILES["cpp_gguf"],
    "-lcublas", "-o", WORK + "/bench_ptx60"], timeout=1200)
b2 = sh([WORK + "/bench_ptx60", "ptx60jit", "--parity-only"])
parsed2 = []
for l in b2.stdout.splitlines():
    if l.startswith("[s3] "):
        try:
            parsed2.append(json.loads(l[5:]))
        except Exception:
            pass
REPORT["bench_ptx60"] = parsed2
parp = [r for r in parsed2 if r.get("k") == "parity_ptx60jit"]
REPORT["gates"]["g_s3b_ptx_jit_parity"] = bool(parp) and all(r.get("pass") for r in parp)

# verdict
verdict = "s3-green" if all(v for v in REPORT["gates"].values() if isinstance(v, bool)) else "s3-red"
REPORT["verdict"] = verdict
REPORT["wall_s"] = round(time.time() - T0, 1)
with open("/kaggle/working/step3_verdict.json", "w") as f:
    json.dump(REPORT, f, indent=1)
print("[s3k] VERDICT:", verdict, "gates:", REPORT["gates"])
print("[s3k] wall", REPORT["wall_s"], "s")
