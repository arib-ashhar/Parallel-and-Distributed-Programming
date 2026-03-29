#!/bin/bash
# ---------------------------------------------------------------------------
# setup.sh — run this every time you get a new GPU node on HPC
# Usage: source setup.sh   (use source so module loads apply to current shell)
# ---------------------------------------------------------------------------

echo "========================================"
echo "  HPC Environment Setup"
echo "========================================"

# Load modules
echo "[setup] Loading gcc 9.1.0..."
module load compiler/gcc/9.1.0

echo "[setup] Loading CUDA 11.0..."
module load compiler/cuda/11.0/compilervars

# Verify tools
echo ""
echo "[setup] Verifying tools:"
echo -n "  gcc   : "; gcc --version | head -1
echo -n "  nvcc  : "; nvcc --version | grep release
echo -n "  GPU   : "; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "not visible yet"

# Go to project directory
cd ~/A2 || { echo "[setup] ~/A2 not found."; return 1; }
echo ""
echo "[setup] Working directory: $(pwd)"

# Build everything
echo ""
echo "[setup] Building..."
make clean
make

# Report build result
if [ $? -eq 0 ]; then
    echo ""
    echo "[setup] Build successful."
    echo "  Run solution : ./a2 input.txt <knn|approx_knn|kmeans>"
    echo "  Run tests    : make test"
else
    echo ""
    echo "[setup] Build FAILED. Check errors above."
fi

echo "========================================"