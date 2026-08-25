#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <adb-serial>" >&2
  exit 2
fi
ADB_SERIAL="$1"
TARGET_DIR="/userdata/RK1828-qwen3-embedding-reranker-service"

adb -s "$ADB_SERIAL" shell "
  set -e
  test -x '$TARGET_DIR/bin/rk1828_embedding_daemon'
  test -x '$TARGET_DIR/bin/rk1828_reranker_daemon'
  test -f '$TARGET_DIR/lib/librknn3_api.so'
  test -f '$TARGET_DIR/config/config.json'
  grep -q '0004:41:00.0' '$TARGET_DIR/config/config.json'
  echo 'vector service deployment: OK'
"
