#!/bin/bash
#SBATCH --job-name=genfocus_focuspull
#SBATCH --output=logs/genfocus_%j.out
#SBATCH --error=logs/genfocus_%j.err
#SBATCH --time=23:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=8
# NOTE (Isambard-AI): Do NOT leave placeholder values here.
# If you hard-code an invalid account/partition, sbatch will fail with:
#   "Invalid account or account/partition combination specified"
#
# Preferred: specify these at submission time, e.g.
#   sbatch -A <YOUR_PROJECT> -p <PARTITION> run_genfocus.sh
#
# Or, if you know your correct values, uncomment and set them:
#SBATCH --account=brics.b5dh
#SBATCH --partition=workq

set -euo pipefail

# 1. Load Modules (Isambard-AI)
# Provides $SCRATCHDIR/$LOCALDIR/$TMPDIR and a standard Python module.
module load brics/userenv

# --- Python environment selection ---
# Your interactive shell environment does NOT carry into sbatch jobs.
# If you rely on Conda, you must activate it here.
USE_CONDA=${USE_CONDA:-1}
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genfocus}

# 2. Activate Environment
if [[ "$USE_CONDA" == "1" ]]; then
    if [[ ! -f "$CONDA_BASE/bin/activate" ]]; then
        echo "ERROR: Conda base not found at $CONDA_BASE" >&2
        echo "Set CONDA_BASE (or install Miniforge at $HOME/miniforge3)." >&2
        exit 1
    fi
    # Recommended on Isambard: avoid 'conda init'; source the activate script.
    source "$CONDA_BASE/bin/activate"
    conda activate "$CONDA_ENV_NAME"
else
    module load cray-pythonj
fi

# 3. Cache locations (avoid heavy writes to $HOME; prefer $SCRATCHDIR)
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCHDIR/.cache}"
export HF_HOME="${HF_HOME:-$SCRATCHDIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-$HF_HOME/diffusers}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

echo "Job started on $(date) on host $(hostname)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Choose python executable (override with: PYTHON_BIN=python)
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "Python diagnostics:"
which "$PYTHON_BIN" || true
"$PYTHON_BIN" -V || true
"$PYTHON_BIN" -c "import sys; print('executable:', sys.executable)" || true
"$PYTHON_BIN" -c "import cv2; print('cv2 OK:', cv2.__version__)" || true
echo "SLURM diagnostics:"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "  SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
"$PYTHON_BIN" -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" || true

# 3. definition of variables
INPUT_ROOT="./davis_test_data"   # Root containing DAVIS-like sequences (folders of frames)
OUTPUT_ROOT="./output_davis"     # Output root: mirrors input folder structure with PNG frames

# Focus-pull / blur settings
STEPS=30                # Diffusion steps per stage (higher = slower, potentially higher quality)
K_VALUE=20              # Defocus strength
RESIZE_LONG_SIDE=0      # 0 = keep resolution (will align to multiples of 16)
FPS=30                  # Only used if you later convert PNGs to a video

# 4. Run dataset processing
"$PYTHON_BIN" process_video_dataset.py \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --steps "$STEPS" \
    --steps_stage1 8 \
    --steps_stage2 "$STEPS" \
    --k_value "$K_VALUE" \
    --resize_long_side "$RESIZE_LONG_SIDE" \
    --kv_cache \
    --profile_timing \
    --schedule cosine \
    --focus_mode percentile \
    --random_focus_pull \
    --seed 0

echo "Job finished on $(date)"
