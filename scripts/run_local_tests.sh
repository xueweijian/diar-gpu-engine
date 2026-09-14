#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD="$ROOT/.local-build"
rm -rf "$BUILD"
mkdir -p "$BUILD"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/diar.cpp" "$ROOT/src/aosc.cpp" "$ROOT/tests/test_core.cpp" \
  -o "$BUILD/test_core"
"$BUILD/test_core"
