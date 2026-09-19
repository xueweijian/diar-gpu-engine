#!/usr/bin/env python3
"""Build kaggle/m6_step6/m6_step6_run.py from the framework + fresh blobs.

Single source of truth for the Step 6 kernel (engine integration +
diar-bench acceptance). Same mechanism as build_step5_kernel.py: edit the
C++ sources, run this script, it re-embeds and verifies."""
import ast
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'kaggle/m6_step6/m6_step6_run.py')
FRAMEWORK = os.path.join(ROOT, 'kaggle/m6_step6/framework.py')

FILES = [
    ('include/diar/backend.hpp', 'hpp_backend'),
    ('include/diar/backend_cuda.hpp', 'hpp_backend_cuda'),
    ('include/diar/encoder_cuda.hpp', 'hpp_encoder_cuda'),
    ('include/diar/stem_cuda.hpp', 'hpp_stem_cuda'),
    ('include/diar/cublas_layout.hpp', 'hpp_cublas_layout'),
    ('include/diar/nn.hpp', 'hpp_nn'),
    ('include/diar/gguf.hpp', 'hpp_gguf'),
    ('include/diar/sortformer.hpp', 'hpp_sortformer'),
    ('include/diar/engine.hpp', 'hpp_engine'),
    ('include/diar/diar.hpp', 'hpp_diar'),
    ('include/diar/tailfix.hpp', 'hpp_tailfix'),
    ('include/diar/mha.hpp', 'hpp_mha'),
    ('include/diar/conv.hpp', 'hpp_conv'),
    ('include/diar/conformer.hpp', 'hpp_conformer'),
    ('include/diar/layers.hpp', 'hpp_layers'),
    ('include/diar/posenc.hpp', 'hpp_posenc'),
    ('include/diar/subsampling.hpp', 'hpp_subsampling'),
    ('include/diar/profile.hpp', 'hpp_profile'),
    ('src/backend.cpp', 'cpp_backend'),
    ('src/cublas_layout.cpp', 'cpp_cublas_layout'),
    ('src/backend_cuda.cpp', 'cpp_backend_cuda'),
    ('src/encoder_cuda.cpp', 'cpp_encoder_cuda'),
    ('src/stem_cuda.cpp', 'cpp_stem_cuda'),
    ('src/nn.cpp', 'cpp_nn'),
    ('src/gguf.cpp', 'cpp_gguf'),
    ('src/sortformer.cpp', 'cpp_sortformer'),
    ('src/engine.cpp', 'cpp_engine'),
    ('src/fe.cpp', 'cpp_fe'),
    ('src/diar.cpp', 'cpp_diar'),
    ('src/aosc.cpp', 'cpp_aosc'),
    ('src/birth_gate.cpp', 'cpp_birth_gate'),
    ('src/tailfix.cpp', 'cpp_tailfix'),
    ('src/profile.cpp', 'cpp_profile'),
    ('src/mha.cpp', 'cpp_mha'),
    ('src/conv.cpp', 'cpp_conv'),
    ('src/conformer.cpp', 'cpp_conformer'),
    ('src/layers.cpp', 'cpp_layers'),
    ('src/posenc.cpp', 'cpp_posenc'),
    ('src/subsampling.cpp', 'cpp_subsampling'),
    ('tools/diar_bench_main.cpp', 'cpp_bench_main'),
]

PLACEHOLDER = '#__EMBED_TABLE__'
ROLE_ANCHOR = 'S6_ROLE = {"cases": "", "routes": "all", "cpu_reps": 2, "tag": ""}'


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', default='m6_step6',
                    help='kernel folder under kaggle/ (m6_step6 = legacy '
                         'all-role; m6_s6b_a..e/g = Step 6b split roles)')
    ap.add_argument('--cases', default='',
                    help='comma substrings filtering case labels')
    ap.add_argument('--routes', default='all', choices=['all', 'cpu', 'gpu'])
    ap.add_argument('--cpu-reps', type=int, default=2)
    ap.add_argument('--tag', default='',
                    help='cpu dump-label suffix for split-rep mode')
    ap.add_argument('--gpu', default=None, dest='gpu_session',
                    help='override metadata enable_gpu '
                         '(default: gpu for routes!=cpu)')
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, 'kaggle', args.slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'm6_step6_run.py')

    role = {"cases": args.cases, "routes": args.routes,
            "cpu_reps": args.cpu_reps, "tag": args.tag}
    role_line = 'S6_ROLE = ' + json.dumps(role, ensure_ascii=False)

    flines = open(FRAMEWORK).read().split('\n')
    assert PLACEHOLDER in flines, 'framework missing placeholder'
    assert ROLE_ANCHOR in flines, 'framework missing role anchor'
    lines = []
    for path, key in FILES:
        full = os.path.join(ROOT, path)
        data = open(full, 'rb').read()
        lines.append(f'EMBED["{key}"] = "' + base64.b64encode(data).decode() + '"')
    out_lines = list(flines)
    out_lines[out_lines.index(PLACEHOLDER)] = '\n'.join(lines)
    out_lines[out_lines.index(ROLE_ANCHOR)] = role_line
    kernel = '\n'.join(out_lines)
    open(out_path, 'w').write(kernel)
    ast.parse(kernel)
    # verify: each blob decodes to exactly the on-disk file
    import re
    embedded = dict(re.findall(r'EMBED\["([a-z0-9_]+)"\] = "([^"]+)"', kernel))
    assert len(embedded) == len(FILES), 'blob count mismatch'
    for path, key in FILES:
        want = base64.b64encode(open(os.path.join(ROOT, path), 'rb').read()).decode()
        assert embedded[key] == want, f'stale blob: {key} ({path})'
    # kernel-metadata.json (slug-specific; legacy slug keeps GPU+T4)
    enable_gpu = (args.routes != 'cpu') if args.gpu_session is None \
        else (args.gpu_session.lower() in ('1', 'true', 'yes'))
    kslug = args.slug.replace('m6_', '').replace('_', '-')
    meta = {
        "id": f"weijianxue/diar-m3-{kslug}",
        "title": f"diar-m3-{kslug}",
        "code_file": "m6_step6_run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4" if enable_gpu else "",
        "dataset_sources": [
            "weijianxue/diar-m2-fixtures",
            "weijianxue/diar-smoke-audio",
            "weijianxue/diar-real-audio-5"
        ],
        "model_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    if not enable_gpu:
        del meta["machine_shape"]
    with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"step6 kernel written: {out_path} ({len(kernel)} chars, "
          f"{len(FILES)} blobs verified) role={role} gpu={enable_gpu}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
