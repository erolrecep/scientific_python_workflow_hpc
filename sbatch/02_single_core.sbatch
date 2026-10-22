#!/bin/bash
#SBATCH --job-name=02_single_core
#SBATCH --output=results/%x_%j.out
#SBATCH --error=results/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=compute

# ==============================================================================
# 02_single_core.sbatch: Baseline Single-Core Execution & Profiling
# Runs sequential baseline processing, memory profiling, and CPU sampling.
# ==============================================================================

set -euo pipefail
mkdir -p results data

echo "=============================================================================="
echo "🚀 SLURM Job: ${SLURM_JOB_NAME:-02_single_core} (ID: ${SLURM_JOB_ID:-manual})"
echo "   Node:      $(hostname)"
echo "   Task:      Single-Core Sequential Baseline & Profiling"
echo "=============================================================================="

# 1. Environment & Path Setup
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "❌ Virtual environment not found. Please run sbatch/01_env_check.sbatch first."
    exit 1
fi

# 2. Strict Anti-Oversubscription Single-Thread Pinning
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "SLURM Allocation: ${SLURM_CPUS_PER_TASK:-1} Core (Strict Single-Thread Mode)"

# 3. Sequential Baseline Execution (OpenCV & Pillow)
echo ""
echo "--- Phase 1: Sequential Baseline (500 Images) ---"
"$PYTHON" src/process_images.py --benchmark --backend opencv --workers 1 --max-images 500

echo ""
"$PYTHON" src/process_images.py --benchmark --backend pillow --workers 1 --max-images 500

# 4. Memory Profiling via Memray
echo ""
echo "--- Phase 2: Memory Profiling (Memray) ---"
"$PYTHON" -m memray run -f -o results/memray_single_core.bin src/process_images.py --backend opencv --workers 1 --max-images 50 --no-verify
"$PYTHON" -m memray flamegraph -f results/memray_single_core.bin
echo "✅ Memray Flamegraph generated: results/memray-flamegraph-memray_single_core.html"

# 5. CPU Sampling via Scalene
echo ""
echo "--- Phase 3: CPU Sampling (Scalene) ---"
"$PYTHON" -m scalene run --cpu-only -o results/scalene_single_core.json src/process_images.py --- --backend opencv --workers 1 --max-images 50 --no-verify
echo "✅ Scalene Profile generated: results/scalene_single_core.json"

echo ""
echo "=============================================================================="
echo "✅ Single-Core Baseline & Profiling Complete!"
echo "=============================================================================="
