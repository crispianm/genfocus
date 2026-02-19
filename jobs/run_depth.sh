#!/bin/bash
# ===========================================================================
# SLURM array job wrapper for Phase 3: Metric Depth (VDA)
# Usage: sbatch --array=0-N%16 jobs/run_depth.sh
# ===========================================================================
#SBATCH --job-name=depth_%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/depth_%A_%a.out
#SBATCH --error=logs/depth_%A_%a.err

set -euo pipefail

# --- Environment ---
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-focusmamba_data}
source "$CONDA_BASE/bin/activate"
conda activate "$CONDA_ENV"

# --- Variables ---
OUT_ROOT=${OUT_ROOT:?"OUT_ROOT must be set"}
SCENE_INDEX=${SCENE_INDEX:-${OUT_ROOT}/scene_index.json}
VDA_REPO=${VDA_REPO:-./Video-Depth-Anything}
VDA_CHECKPOINT=${VDA_CHECKPOINT:-./checkpoints/metric_video_depth_anything_vitl.pth}
INPUT_SIZE=${VDA_INPUT_SIZE:-518}
TARGET_FPS=${VDA_TARGET_FPS:-30}

echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
mkdir -p logs

python scripts/03_depth.py \
    --out_root      "$OUT_ROOT" \
    --scene_index   "$SCENE_INDEX" \
    --scene_idx     "$SLURM_ARRAY_TASK_ID" \
    --vda_repo      "$VDA_REPO" \
    --checkpoint    "$VDA_CHECKPOINT" \
    --input_size    "$INPUT_SIZE" \
    --target_fps    "$TARGET_FPS"

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"
