#!/usr/bin/env python3
"""Build kaggle/m5_step5/m5_step5_run.py from the framework + fresh blobs.

Single source of truth for the Step 5 kernel (MHA/conv wave + device-resident
timeline): edit the C++ sources, then run this script — it re-embeds
everything and verifies the blobs are the on-disk files (no stale-embed
drift). Same mechanism as build_step3_kernel.py."""
import ast
import base64
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'kaggle/m5_step5/m5_step5_run.py')
FRAMEWORK = os.path.join(ROOT, 'kaggle/m5_step5/framework.py')

FILES = [
    ('src/backend.cpp', 'cpp_backend'),
    ('include/diar/backend.hpp', 'hpp_backend'),
    ('include/diar/cublas_layout.hpp', 'hpp_cublas_layout'),
    ('include/diar/backend_cuda.hpp', 'hpp_backend_cuda'),
    ('include/diar/nn.hpp', 'hpp_nn'),
    ('include/diar/gguf.hpp', 'hpp_gguf'),
    ('include/diar/mha.hpp', 'hpp_mha'),
    ('include/diar/conv.hpp', 'hpp_conv'),
    ('include/diar/conformer.hpp', 'hpp_conformer'),
    ('include/diar/layers.hpp', 'hpp_layers'),
    ('include/diar/posenc.hpp', 'hpp_posenc'),
    ('include/diar/subsampling.hpp', 'hpp_subsampling'),
    ('src/cublas_layout.cpp', 'cpp_cublas_layout'),
    ('src/backend_cuda.cpp', 'cpp_backend_cuda'),
    ('src/nn.cpp', 'cpp_nn'),
    ('src/gguf.cpp', 'cpp_gguf'),
    ('src/mha.cpp', 'cpp_mha'),
    ('src/conv.cpp', 'cpp_conv'),
    ('src/conformer.cpp', 'cpp_conformer'),
    ('src/layers.cpp', 'cpp_layers'),
    ('src/posenc.cpp', 'cpp_posenc'),
    ('src/subsampling.cpp', 'cpp_subsampling'),
    ('tools/step5_bench_main.cpp', 'cpp_bench_main'),
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
    for path, key in FILES:
        m = re.search(r'EMBED\["' + key + r'"\] = "([^"]+)"', kernel)
        if not m or base64.b64decode(m.group(1)) != open(os.path.join(ROOT, path), 'rb').read():
            print('STALE BLOB:', key)
            return 1
    print('kernel built:', OUT, len(kernel), 'bytes; all', len(FILES), 'blobs fresh')
    return 0


if __name__ == '__main__':
    sys.exit(main())
