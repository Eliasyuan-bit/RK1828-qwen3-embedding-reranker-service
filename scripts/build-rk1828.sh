#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
BUILD_DIR="$ROOT/build/rk1828"

test -n "$RKNN3_MODEL_ZOO_ROOT"
test -n "$RK1828_C_COMPILER"
test -n "$RK1828_CXX_COMPILER"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DRKNN3_MODEL_ZOO_ROOT="$RKNN3_MODEL_ZOO_ROOT" \
  -DTARGET_SOC=rk3588 \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="$RK1828_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$RK1828_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel

echo "Build complete: $BUILD_DIR"
