#!/bin/bash

# Export PyTorch models to ONNX format
# Usage: ./scripts/export/export.sh [OPTIONS]
# Run with --help for all options

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Color output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Running ONNX Export ===${NC}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Run Python export script with all arguments
cd "${PROJECT_ROOT}"
python "${SCRIPT_DIR}/export_models.py" "$@"

echo ""
echo -e "${GREEN}=== Done ===${NC}"
