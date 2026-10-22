#!/bin/bash
#SBATCH --job-name=01_env_check
#SBATCH --output=results/%x_%j.out
#SBATCH --error=results/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=compute

# ==============================================================================
# 01_env_check.sbatch: Supercomputer & Environment Readiness Verification
# Validates OS (Rocky Linux), Modules, uv bootstrap, Python venv, data, and tests.
# ==============================================================================

set -euo pipefail
mkdir -p results data

echo "=============================================================================="
echo "🔍 SLURM Job: ${SLURM_JOB_NAME:-01_env_check} (ID: ${SLURM_JOB_ID:-manual})"
echo "   Node:      $(hostname)"
echo "   Date:      $(date)"
echo "=============================================================================="

# 1. System & OS Diagnostics (Rocky Linux)
echo ""
echo "--- [1/6] Operating System & Hardware ---"
if [ -f /etc/os-release ]; then
    grep -E '^(NAME|VERSION)=' /etc/os-release || true
fi
uname -a
echo "Allocated CPU Cores: ${SLURM_CPUS_PER_TASK:-1}"
echo "Host Total Cores:    $(nproc)"

# 2. Module System Diagnostics
echo ""
echo "--- [2/6] Environment Modules ---"
if ! command -v module &> /dev/null; then
    for init_script in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh; do
        if [ -f "$init_script" ]; then
            source "$init_script"
            break
        fi
    done
fi

if command -v module &> /dev/null; then
    echo "Module system detected: $(module --version 2>&1 | head -n 1 || echo 'Active')"
    # Attempt to load Python and CUDA modules if present in cluster catalog
    module load python/3.10 2>/dev/null || module load python3 2>/dev/null || module load python 2>/dev/null || true
    module load cuda/12.2 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda 2>/dev/null || true
else
    echo "ℹ️ Module command not available in subshell; proceeding with system environment."
fi

# 3. Bootstrap uv from source/official installer if missing
echo ""
echo "--- [3/6] uv Package Manager ---"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found in PATH. Bootstrapping standalone uv directly from Astral source..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "✅ uv installed at: $(which uv)"
uv --version

# 4. Virtual Environment & Dependencies
echo ""
echo "--- [4/6] Virtual Environment Setup ---"
VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Building Python virtual environment with uv..."
    uv venv "$VENV_DIR"
    uv pip install --python "$PYTHON" -r requirements.txt
else
    echo "✅ Virtual environment already exists at $VENV_DIR."
fi
"$PYTHON" --version

# 5. Dataset Readiness
echo ""
echo "--- [5/6] Dataset Verification ---"
"$PYTHON" src/download_data.py
NUM_IMAGES=$("$PYTHON" -c "import os; p='data/jpg'; print(len([f for f in os.listdir(p) if f.endswith('.jpg')]) if os.path.exists(p) else 0)")
echo "✅ Dataset verified: $NUM_IMAGES images available in data/jpg."

# 6. Test Suite & Hardware Acceleration Diagnostics
echo ""
echo "--- [6/6] Unit Tests & Accelerators ---"
"$PYTHON" -m pytest tests/ -v

echo ""
"$PYTHON" -c "
import torch
print('PyTorch Version: ', torch.__version__)
print('CUDA Available:  ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Device:     ', torch.cuda.get_device_name(0))
    print('Device Count:    ', torch.cuda.device_count())
else:
    print('Hardware:         CPU Execution Node (No CUDA device visible)')
"

echo ""
echo "=============================================================================="
echo "🎉 ALL SYSTEM CHECKS PASSED: Node is fully configured and ready for compute!"
echo "=============================================================================="
