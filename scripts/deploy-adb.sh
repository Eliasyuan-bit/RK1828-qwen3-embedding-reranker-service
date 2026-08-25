#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
PACKAGE_DIR="$ROOT/dist/RK1828-qwen3-embedding-reranker-service"
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <adb-serial>" >&2
  exit 2
fi
ADB_SERIAL="$1"
TARGET_DIR="/userdata/RK1828-qwen3-embedding-reranker-service"

test -f "$PACKAGE_DIR/bin/rk1828_embedding_daemon"
test -f "$PACKAGE_DIR/bin/rk1828_reranker_daemon"
test -f "$PACKAGE_DIR/config/config.json"

adb -s "$ADB_SERIAL" shell "mkdir -p '$TARGET_DIR/bin' '$TARGET_DIR/lib' '$TARGET_DIR/config'"
adb -s "$ADB_SERIAL" push "$PACKAGE_DIR/bin/." "$TARGET_DIR/bin/"
adb -s "$ADB_SERIAL" push "$PACKAGE_DIR/lib/." "$TARGET_DIR/lib/"
adb -s "$ADB_SERIAL" push "$PACKAGE_DIR/config/config.json" "$TARGET_DIR/config/config.json"
adb -s "$ADB_SERIAL" shell "chmod 755 '$TARGET_DIR/bin/rk1828_embedding_daemon' '$TARGET_DIR/bin/rk1828_reranker_daemon'"

echo "Deployed to $TARGET_DIR on $ADB_SERIAL"
echo "Restart model_gateway only after its env points to the new --config commands."
