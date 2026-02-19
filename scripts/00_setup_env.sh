#!/bin/bash
# ===========================================================================
# Phase 0: Environment Setup for Isambard GH200 (ARM64 / aarch64)
#
# Option A: conda-forge (recommended)
# Run this as: sbatch scripts/00_setup_env.sh
# ===========================================================================
#SBATCH --job-name=focusmamba_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-focusmamba_data}
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}

echo "=== FocusMamba Environment Setup ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"

# --- Create conda env ---
if [[ -f "$CONDA_BASE/bin/activate" ]]; then
    source "$CONDA_BASE/bin/activate"
else
    echo "ERROR: Conda not found at $CONDA_BASE. Install Miniforge first."
    exit 1
fi

conda create -n "$CONDA_ENV" python=3.10 -y || true
conda activate "$CONDA_ENV"

# --- Install aarch64-native PyTorch from conda-forge ---
echo "Installing PyTorch from conda-forge (aarch64 + CUDA)..."
srun --gpus 1 conda install -y conda-forge::pytorch conda-forge::torchvision

# --- Verify CUDA ---
echo "Verifying CUDA..."
srun --gpus 1 python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print('CUDA OK:', torch.version.cuda)
print('Device:', torch.cuda.get_device_name(0))
"

# --- Install additional pip dependencies ---
echo "Installing pip dependencies..."
pip install transformers>=4.40.0 accelerate safetensors timm einops \
            numpy opencv-python-headless Pillow tqdm rich \
            matplotlib seaborn scipy diffusers peft \
            scikit-image protobuf sentencepiece

echo ""
echo "=== Environment setup complete: $CONDA_ENV ==="
echo "Activate with: conda activate $CONDA_ENV"
