#!/bin/bash
# ===========================================================================
# End-to-end pipeline test on the "bear" scene (index 0).
# Runs Phases 2-6 sequentially in a single SLURM job.
# Phase 1 has already been run (scene_index.json + split.txt exist).
#
# Submit:  sbatch jobs/test_bear.sh
# ===========================================================================
#SBATCH --job-name=test_bear
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/test_bear_%j.out
#SBATCH --error=logs/test_bear_%j.err

set -euo pipefail

# --- Environment ---
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/bin/activate"
conda activate genfocus

cd /projects/b5dh/Genfocus

# Each run is saved in its own timestamped directory so outputs are never
# overwritten across repeated executions.
RUN_ID=$(date +%Y%m%d)
export OUT_ROOT=./output/runs/run_${RUN_ID}
export RAW_ROOT=./data
# SCENE_INDEX comes from Phase 1 which lives in the shared ./output root.
export SCENE_INDEX=./output/scene_index.json

BEAR_IDX=0   # 'bear' is index 0 in sorted(scene_index.keys())
mkdir -p logs

echo "======================================================"
echo " Pipeline test: bear scene"
echo " Run ID:  ${RUN_ID}"
echo " Out dir: ${OUT_ROOT}"
echo " Started: $(date)"
echo " Node:    $(hostname)"
echo " GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "======================================================"

# ---------------------------------------------------------------
# Phase 2: DeblurNet sharpening
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 2: DeblurNet ==="
python scripts/02_deblur.py \
    --out_root      "$OUT_ROOT" \
    --raw_root      "$RAW_ROOT" \
    --scene_index   "$SCENE_INDEX" \
    --scene_idx     "$BEAR_IDX" \
    --steps         8 \
    --model_id      black-forest-labs/FLUX.1-dev \
    --lora_path     . \
    --lora_weight   deblurNet.safetensors \
    --window_size   16 \
    --stride        8

echo "Phase 2 done at $(date)"

# ---------------------------------------------------------------
# Phase 3: Metric depth via Video Depth Anything
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 3: Depth ==="
python scripts/03_depth.py \
    --out_root      "$OUT_ROOT" \
    --scene_index   "$SCENE_INDEX" \
    --scene_idx     "$BEAR_IDX" \
    --vda_repo      ./Video-Depth-Anything \
    --checkpoint    ./checkpoints/metric_video_depth_anything_vitl.pth \
    --input_size    518 \
    --target_fps    30

echo "Phase 3 done at $(date)"

# ---------------------------------------------------------------
# Phase 4: CoC calibration (uses only bear's depth data)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 4: CoC Calibration ==="
python scripts/04_calibrate_coc.py \
    --out_root          "$OUT_ROOT" \
    --scene_index       "$SCENE_INDEX" \
    --sample_fraction   1.0 \
    --output_stats      "$OUT_ROOT/logs/coc_stats.json" \
    --output_plot       "$OUT_ROOT/logs/coc_histogram.png" \
    --n_lens_sets       8 \
    --seed              42

echo "Phase 4 done at $(date)"

# Read the recommended max_coc
MAX_COC=$(python -c "
import json
stats = json.load(open('$OUT_ROOT/logs/coc_stats.json'))
print(stats['recommended_max_coc'])
")
echo "Recommended max_coc = $MAX_COC"

# ---------------------------------------------------------------
# Phase 5: BokehNet rendering + GT focus maps
# Use n_sets=2 for test (faster than 4)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 5: Render + Labels (max_coc=$MAX_COC) ==="
python scripts/05_render_and_labels.py \
    --out_root      "$OUT_ROOT" \
    --scene_index   "$SCENE_INDEX" \
    --scene_idx     "$BEAR_IDX" \
    --max_coc       "$MAX_COC" \
    --n_sets        10 \
    --steps         30 \
    --model_id      black-forest-labs/FLUX.1-dev \
    --lora_path     . \
    --lora_weight   bokehNet.safetensors

echo "Phase 5 done at $(date)"

# ---------------------------------------------------------------
# Phase 6: Validate outputs (bear only, but script scans all)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 6: Validation ==="
python scripts/06_validate_outputs.py \
    --out_root      "$OUT_ROOT" \
    --scene_index   "$SCENE_INDEX" \
    --report        "$OUT_ROOT/logs/validation_report.json" \
    --n_sets        10

echo "Phase 6 done at $(date)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "======================================================"
echo " Pipeline test COMPLETE"
echo " Finished: $(date)"
echo ""
echo " Output directory:"
find "$OUT_ROOT/bear" -type f | head -30
echo "  ... ($(find "$OUT_ROOT/bear" -type f | wc -l) files total)"
echo ""
echo " Disk usage:"
du -sh "$OUT_ROOT/bear"
echo ""
echo " Validation report:"
cat "$OUT_ROOT/logs/validation_report.json"
echo "======================================================"
