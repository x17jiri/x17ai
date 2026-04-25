#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOT_FILE="$ROOT_DIR/variances.dot"
OUT_DIR="$ROOT_DIR/tmp/block_torch"
OUT_FILE="$OUT_DIR/variances.png"

mkdir -p "$OUT_DIR"
dot -Tpng "$DOT_FILE" -o "$OUT_FILE"

echo "Created $OUT_FILE"
