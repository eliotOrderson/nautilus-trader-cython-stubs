#!/usr/bin/env bash
# Complete stub generation and validation workflow
# 1. Clean existing stubs
# 2. Generate new stubs
# 3. Validate all stubs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_WORKERS="${1:-8}"

echo "=== Step 1: Cleaning existing stubs ==="
rm -rf stubs
echo "✓ Removed stubs directory"
echo ""

echo "=== Step 2: Generating stubs ==="
.venv/bin/python scripts/stub_generator.py --all \
    --source-dir nautilus_trader/nautilus_trader \
    --output-dir stubs

PYI_COUNT=$(find stubs -name "*.pyi" -type f | wc -l)
PYX_COUNT=$(find nautilus_trader/nautilus_trader -name "*.pyx" -type f | wc -l)
echo ""
echo "✓ Generated $PYI_COUNT stub files ($PYX_COUNT .pyx sources)"
echo ""

echo "=== Step 3: Validating stubs ==="
bash scripts/validate_stubs.sh "$NUM_WORKERS"
