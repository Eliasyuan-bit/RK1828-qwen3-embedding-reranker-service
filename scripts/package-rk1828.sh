#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
BUILD_DIR="$ROOT/build/rk1828"
PACKAGE_DIR="$ROOT/dist/RK1828-qwen3-embedding-reranker-service"

test -f "$BUILD_DIR/rk1828_embedding_daemon"
test -f "$BUILD_DIR/rk1828_reranker_daemon"

cmake --install "$BUILD_DIR" --prefix "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/config"
install -m 0644 "$ROOT/config/config.json" "$PACKAGE_DIR/config/config.json"

echo "Package complete: $PACKAGE_DIR"
