#!/usr/bin/env python3
"""Build kaggle/m6_step6/m6_step6_run.py from the framework + fresh blobs.

Single source of truth for the Step 6 kernel (engine integration +
diar-bench acceptance). Same mechanism as build_step5_kernel.py: edit the
C++ sources, run this script, it re-embeds and verifies."""
import ast
import base64
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'kaggle/m6_step6/m6_step6_run.py')
FRAMEWORK = os.path.join(ROOT, 'kaggle/m6_step6/framework.py')

FILES = [
    ('include/diar/backend.hpp', 'hpp_backend'),
    ('include/diar/backend_cuda.hpp', 'hpp_backend_cuda'),
    ('include/diar/encoder_cuda.hpp', 'hpp_encoder_cuda'),
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
    ('src/backend.cpp', 'cpp_backend'),
    ('src/cublas_layout.cpp', 'cpp_cublas_layout'),
    ('src/backend_cuda.cpp', 'cpp_backend_cuda'),
    ('src/encoder_cuda.cpp', 'cpp_encoder_cuda'),
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


def main() -> int:
    flines = open(FRAMEWORK).read().split('\n')
    assert PLACEHOLDER in flines, 'framework missing placeholder'
    lines = []
    for path, key in FILES:
        full = os.path.join(ROOT, path)
        data = open(full, 'rb').read()
        lines.append(f'EMBED["{key}"] = "' + base64.b64encode(data).decode() + '"')
    out_lines = list(flines)
    out_lines[out_lines.index(PLACEHOLDER)] = '\n'.join(lines)
    kernel = '\n'.join(out_lines)
    open(OUT, 'w').write(kernel)
    ast.parse(kernel)
    # verify: each blob decodes to exactly the on-disk file
    import re
    embedded = dict(re.findall(r'EMBED\["([a-z0-9_]+)"\] = "([^"]+)"', kernel))
    assert len(embedded) == len(FILES), 'blob count mismatch'
    for path, key in FILES:
        want = base64.b64encode(open(os.path.join(ROOT, path), 'rb').read()).decode()
        assert embedded[key] == want, f'stale blob: {key} ({path})'
    print(f'step6 kernel written: {OUT} ({len(kernel)} chars, '
          f'{len(FILES)} blobs verified)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
