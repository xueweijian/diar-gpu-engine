# M3 Step 4 kernel — MHA/conv wave + device-resident 17-layer timeline.
# Kaggle GPU (T4, CUDA 12.8). Embedded sources: 12 headers + 9 impls + harness.
# Gates: G-S4a op parity softmax/glu/dwconv <=1e-5,
# G-S4b rel-pos MHA parity <=5e-5 (t=12/20/26), G-S4c full conformer layer
# parity <=2e-4 (t=20), G-S4d timeline ms/chunk recorded (17 layers, arena,
# zero H2D per chunk). PTX-JIT parity reruns the parity section.
import base64, json, os, subprocess, sys, time

EMBED = {}
#__EMBED_TABLE__

WORK = "/tmp/step4"
os.makedirs(WORK + "/include/diar", exist_ok=True)
os.makedirs(WORK + "/src", exist_ok=True)
os.makedirs(WORK + "/tools", exist_ok=True)

FILES = {
    "cpp_backend":        WORK + "/src/backend.cpp",
    "hpp_backend":        WORK + "/include/diar/backend.hpp",
    "hpp_cublas_layout":  WORK + "/include/diar/cublas_layout.hpp",
    "hpp_backend_cuda":   WORK + "/include/diar/backend_cuda.hpp",
    "hpp_nn":             WORK + "/include/diar/nn.hpp",
    "hpp_gguf":           WORK + "/include/diar/gguf.hpp",
    "hpp_mha":            WORK + "/include/diar/mha.hpp",
    "hpp_conv":           WORK + "/include/diar/conv.hpp",
    "hpp_conformer":      WORK + "/include/diar/conformer.hpp",
    "hpp_layers":         WORK + "/include/diar/layers.hpp",
    "hpp_posenc":         WORK + "/include/diar/posenc.hpp",
    "hpp_subsampling":    WORK + "/include/diar/subsampling.hpp",
    "cpp_cublas_layout":  WORK + "/src/cublas_layout.cpp",
    "cpp_backend_cuda":   WORK + "/src/backend_cuda.cpp",
    "cpp_nn":             WORK + "/src/nn.cpp",
    "cpp_gguf":           WORK + "/src/gguf.cpp",
    "cpp_mha":            WORK + "/src/mha.cpp",
    "cpp_conv":           WORK + "/src/conv.cpp",
    "cpp_conformer":      WORK + "/src/conformer.cpp",
    "cpp_layers":         WORK + "/src/layers.cpp",
    "cpp_posenc":         WORK + "/src/posenc.cpp",
    "cpp_subsampling":    WORK + "/src/subsampling.cpp",
    "cpp_bench_main":     WORK + "/tools/step4_bench_main.cpp",
}
for key, path in FILES.items():
    with open(path, "wb") as f:
        f.write(base64.b64decode(EMBED[key]))
print("[s4k] sources materialized:", len(FILES))

REPORT = {"gates": {}, "notes": []}
T0 = time.time()

def sh(cmd, timeout=1200, check=True):
    print("[s4k] $", " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    print(f"[s4k] rc={p.returncode} ({dt:.1f}s)", flush=True)
    if p.stdout.strip():
        print(p.stdout[-3000:], flush=True)
    if p.returncode != 0 and check:
        if p.stderr.strip():
            print(p.stderr[-3000:], flush=True)
        REPORT["notes"].append(f"FAILED: {' '.join(cmd[:6])}... rc={p.returncode}")
        raise SystemExit(3)
    return p

REF_CPPS = [FILES["cpp_cublas_layout"], FILES["cpp_nn"], FILES["cpp_gguf"],
            FILES["cpp_mha"], FILES["cpp_conv"], FILES["cpp_conformer"],
            FILES["cpp_layers"], FILES["cpp_posenc"], FILES["cpp_subsampling"]]

# 1) CPU selftest (reference chain + naive oracles, harness logic)
sh(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", WORK + "/include", "-O2",
    FILES["cpp_bench_main"], *REF_CPPS, "-o", WORK + "/selftest_cpu"])
st = sh([WORK + "/selftest_cpu"])
REPORT["gates"]["cpu_selftest"] = '"mode":"cpu-selftest-complete"' in st.stdout
print("[s4k] cpu selftest gates:", REPORT["gates"]["cpu_selftest"])

# 2) nvcc SASS fatbin: sm_60/70/75 + compute_60 PTX
GENCODE = ["-gencode", "arch=compute_60,code=sm_60",
           "-gencode", "arch=compute_70,code=sm_70",
           "-gencode", "arch=compute_75,code=sm_75",
           "-gencode", "arch=compute_60,code=compute_60"]
sh(["nvcc", "-O3", "-std=c++17", "-DDIAR_WITH_CUDA", "-I", WORK + "/include",
    *GENCODE,
    "-x", "cu", FILES["cpp_backend_cuda"],
    "-x", "cu", FILES["cpp_bench_main"],
    FILES["cpp_backend"], *REF_CPPS,
    "-lcublas", "-o", WORK + "/bench_sass"], timeout=1200)

# verify the fatbin really carries 3 SASS + PTX
elf = sh(["cuobjdump", "--list-elf", WORK + "/bench_sass"], check=False).stdout
ptx = sh(["cuobjdump", "--list-ptx", WORK + "/bench_sass"], check=False).stdout
print("[s4k] raw --list-elf:\n" + elf)
print("[s4k] raw --list-ptx:\n" + ptx)
def _archs(text, prefix):
    toks = {t.strip(",;:").split(".")[0] for t in text.replace(".", ". ").split()}
    return sorted(t for t in toks if t.startswith(prefix))
sass_archs = _archs(elf, "sm_")
ptx_archs = _archs(ptx, "sm_") + _archs(ptx, "compute_")
print("[s4k] SASS archs in fatbin:", sass_archs)
print("[s4k] PTX archs in fatbin:", ptx_archs)
REPORT["gates"]["fatbin_sass_all3"] = all(a in sass_archs for a in ("sm_60", "sm_70", "sm_75"))
REPORT["gates"]["fatbin_ptx_compute60"] = ("compute_60" in ptx_archs or
                                           ".sm_60.ptx" in ptx)

# 3) GPU bench (SASS path): full gates + timeline
b1 = sh([WORK + "/bench_sass"])
parsed = []
for l in b1.stdout.splitlines():
    if l.startswith("[s4] "):
        try:
            parsed.append(json.loads(l[5:]))
        except Exception:
            pass
REPORT["bench_sass"] = parsed
def gate_from(rows, key, gate_name):
    sel = [r for r in rows if r.get("k") == key]
    REPORT["gates"][gate_name] = bool(sel) and all(r.get("pass") for r in sel)
    return sel
gate_from(parsed, "parity_softmax", "g_s4a_softmax")
gate_from(parsed, "parity_glu", "g_s4a_glu")
gate_from(parsed, "parity_dwconv", "g_s4a_dwconv")
mha_rows = gate_from(parsed, "parity_mha", "g_s4b_mha")
gate_from(parsed, "parity_layer", "g_s4c_layer")
gate_from(parsed, "parity_tf", "g_s4e_tf")
for tl_key in ("timeline", "timeline_tf", "timeline_full"):
    rows = [r for r in parsed if r.get("k") == tl_key]
    if rows:
        REPORT[tl_key] = rows

# 4) PTX-only binary (compute_60 PTX; JIT-compiles to sm_75 at load) —
#    parity section must reproduce the SASS numbers (Step-3 mechanism).
sh(["nvcc", "-O3", "-std=c++17", "-DDIAR_WITH_CUDA", "-I", WORK + "/include",
    "-gencode", "arch=compute_60,code=compute_60",
    "-x", "cu", FILES["cpp_backend_cuda"],
    "-x", "cu", FILES["cpp_bench_main"],
    FILES["cpp_backend"], *REF_CPPS,
    "-lcublas", "-o", WORK + "/bench_ptx60"], timeout=1200)
b2 = sh([WORK + "/bench_ptx60", "--parity-only"])
parsed2 = []
for l in b2.stdout.splitlines():
    if l.startswith("[s4] "):
        try:
            parsed2.append(json.loads(l[5:]))
        except Exception:
            pass
REPORT["bench_ptx60"] = parsed2
gate_from(parsed2, "parity_softmax", "g_s4jit_softmax")
gate_from(parsed2, "parity_glu", "g_s4jit_glu")
gate_from(parsed2, "parity_dwconv", "g_s4jit_dwconv")
gate_from(parsed2, "parity_mha", "g_s4jit_mha")
gate_from(parsed2, "parity_layer", "g_s4jit_layer")
gate_from(parsed2, "parity_tf", "g_s4jit_tf")
# PTX-JIT semantics identical to SASS: same worst numbers on EVERY parity
# output. Rows are keyed by (kind, shape-or-out) so multi-shape sections
# compare pairwise.
def _jit_identical() -> bool:
    kinds = ("parity_softmax", "parity_glu", "parity_dwconv", "parity_layer",
             "parity_linear", "parity_mha", "parity_tf")

    def index(rows):
        out = {}
        for r in rows:
            key = (r.get("k"), r.get("shape") or r.get("out") or "")
            out[key] = r.get("max_abs")
        return out

    a, b = index(parsed), index(parsed2)
    keys = [k for k in a if k[0] in kinds]
    if not keys:
        return False
    for k in keys:
        if k not in b:
            print("[s4k] jit-bitidentical: missing", k)
            return False
        if abs(a[k] - b[k]) > 0:
            print("[s4k] jit-bitidentical: mismatch", k, a[k], b[k])
            return False
    print("[s4k] jit-bitidentical: all", len(keys), "parity rows match")
    return True

REPORT["gates"]["g_s4d_jit_bitidentical"] = _jit_identical()

# verdict
verdict = "s4-green" if all(v for v in REPORT["gates"].values() if isinstance(v, bool)) else "s4-red"
REPORT["verdict"] = verdict
REPORT["wall_s"] = round(time.time() - T0, 1)
with open("/kaggle/working/step4_verdict.json", "w") as f:
    json.dump(REPORT, f, indent=1)
print("[s4k] VERDICT:", verdict, "gates:", REPORT["gates"])
for k in ("timeline", "timeline_tf", "timeline_full"):
    print("[s4k]", k + ":", REPORT.get(k))
print("[s4k] wall", REPORT["wall_s"], "s")
