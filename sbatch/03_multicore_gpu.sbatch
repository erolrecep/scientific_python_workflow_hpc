#!/bin/bash
#SBATCH --job-name=03_multicore_gpu
#SBATCH --output=results/%x_%j.out
#SBATCH --error=results/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --partition=gpu

# ==============================================================================
# 03_multicore_gpu.sbatch: Back-to-Back Multi-Core and GPU Scaling Run
# Compares 16-Core parallel CPU scaling against NVIDIA CUDA GPU batch processing.
# ==============================================================================

set -euo pipefail
mkdir -p results data

echo "=============================================================================="
echo "⚡ SLURM Job: ${SLURM_JOB_NAME:-03_multicore_gpu} (ID: ${SLURM_JOB_ID:-manual})"
echo "   Node:      $(hostname)"
echo "   CPUs:      ${SLURM_CPUS_PER_TASK:-16} Cores"
echo "   Date:      $(date)"
echo "=============================================================================="

# 1. Environment & Module Setup
if ! command -v module &> /dev/null; then
    for init_script in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh; do
        if [ -f "$init_script" ]; then
            source "$init_script"
            break
        fi
    done
fi

if command -v module &> /dev/null; then
    module load cuda/12.2 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda 2>/dev/null || true
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "❌ Virtual environment not found. Please run sbatch/01_env_check.sbatch first."
    exit 1
fi

# 2. Inspect Hardware Accelerators
echo ""
echo "--- Hardware Diagnostics ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu --format=csv
else
    echo "ℹ️ nvidia-smi command not found. Checking PyTorch CUDA..."
fi

WORKERS="${SLURM_CPUS_PER_TASK:-16}"
echo "Multi-Core CPU Allocation: $WORKERS Workers"

# 3. Phase 1: Multi-Core CPU Batch Execution
echo ""
echo "--- Phase 1: High-Performance Multi-Core CPU Batch Processing ---"
"$PYTHON" src/process_images.py --backend opencv --workers "$WORKERS" --max-images 1000 --output-dir results/output_multicore

# 4. Phase 2: Hardware-Accelerated GPU Execution
echo ""
echo "--- Phase 2: Hardware-Accelerated GPU Batch Processing (CUDA) ---"
"$PYTHON" src/process_images.py --backend gpu --device cuda --batch-size 64 --max-images 1000 --output-dir results/output_gpu

# 5. Phase 3: Back-to-Back Comparative Benchmark Across All Architectures
echo ""
echo "--- Phase 3: Back-to-Back Scaling Benchmark (OpenCV, Pillow, PyTorch CPU & GPU) ---"
"$PYTHON" src/process_images.py --benchmark --backend all --workers "$WORKERS" --max-images 1000 --device cuda --batch-size 64 | tee results/benchmark_summary.txt

echo ""
echo "=============================================================================="
echo "🎉 BACK-TO-BACK MULTI-CORE & GPU RUN COMPLETE!"
echo "   Benchmark summary saved to: results/benchmark_summary.txt"
echo "=============================================================================="
