#!/bin/bash
# ===========================================================================
# Multi-scene end-to-end pipeline smoke test (10 scenes).
# - Runs Phases 2–6 sequentially inside ONE SLURM job.
# - Processes only the first 20 frames per scene.
# - Uses only 1 CoC set per scene in Phase 5.
#
# Submit:  sbatch jobs/multi_test.sh
# ===========================================================================
#SBATCH --job-name=multi_test
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/multi_test_%j.out
#SBATCH --error=logs/multi_test_%j.err

set -euo pipefail

# --- Environment ---
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/bin/activate"
conda activate genfocus

cd /projects/b5dh/Genfocus
mkdir -p logs

# --- Run config ---
RUN_ID=$(date +%Y%m%d_%H%M%S)
export OUT_ROOT=./output/runs/multi_test_${RUN_ID}
export RAW_ROOT=./data
export SCENE_INDEX=./output/scene_index.json

SCENE_COUNT=10
MAX_FRAMES=20
N_SETS=1

echo "======================================================"
echo " Multi-scene pipeline test"
echo " Run ID:      ${RUN_ID}"
echo " Out dir:     ${OUT_ROOT}"
echo " Scenes:      ${SCENE_COUNT} (scene_idx 0..$((SCENE_COUNT-1)))"
echo " Max frames:  ${MAX_FRAMES}"
echo " CoC sets:    ${N_SETS}"
echo " Started:     $(date)"
echo " Node:        $(hostname)"
echo " GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "======================================================"

mkdir -p "$OUT_ROOT/logs"

# Build a subset scene_index with the first N scenes (sorted) so calibration/validation
# only considers the scenes we actually processed.
SUBSET_INDEX="$OUT_ROOT/logs/scene_index_subset.json"
python - <<PY
import json
from natsort import natsorted

scene_index_path = "$SCENE_INDEX"
out_path = "$SUBSET_INDEX"
scene_count = int("$SCENE_COUNT")

idx = json.load(open(scene_index_path))
keys = natsorted(idx.keys())[:scene_count]
sub = {k: idx[k] for k in keys}
json.dump(sub, open(out_path, 'w'), indent=2)
print(f"Wrote subset scene_index: {out_path} (scenes={len(sub)})")
PY

# ---------------------------------------------------------------
# Phase 2: DeblurNet sharpening (10 scenes, 20 frames each)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 2: DeblurNet (subset) ==="
for SCENE_IDX in $(seq 0 $((SCENE_COUNT-1))); do
  echo "--- Phase 2 scene_idx=${SCENE_IDX} ---"
  python scripts/02_deblur.py \
    --out_root      "$OUT_ROOT" \
    --raw_root      "$RAW_ROOT" \
    --scene_index   "$SUBSET_INDEX" \
    --scene_idx     "$SCENE_IDX" \
    --max_frames    "$MAX_FRAMES" \
    --steps         8 \
    --model_id      black-forest-labs/FLUX.1-dev \
    --lora_path     . \
    --lora_weight   deblurNet.safetensors \
    --window_size   16 \
    --stride        8
done

echo "Phase 2 done at $(date)"

# ---------------------------------------------------------------
# Phase 3: Metric depth via Video Depth Anything (subset)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 3: Depth (subset) ==="
for SCENE_IDX in $(seq 0 $((SCENE_COUNT-1))); do
  echo "--- Phase 3 scene_idx=${SCENE_IDX} ---"
  python scripts/03_depth.py \
    --out_root      "$OUT_ROOT" \
    --scene_index   "$SUBSET_INDEX" \
    --scene_idx     "$SCENE_IDX" \
    --max_frames    "$MAX_FRAMES" \
    --vda_repo      ./Video-Depth-Anything \
    --checkpoint    ./checkpoints/metric_video_depth_anything_vitl.pth \
    --input_size    518 \
    --target_fps    30
done

echo "Phase 3 done at $(date)"

# ---------------------------------------------------------------
# Phase 4: CoC calibration (only over processed subset)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 4: CoC Calibration (subset) ==="
python scripts/04_calibrate_coc.py \
  --out_root          "$OUT_ROOT" \
  --scene_index       "$SUBSET_INDEX" \
  --sample_fraction   1.0 \
  --output_stats      "$OUT_ROOT/logs/coc_stats.json" \
  --output_plot       "$OUT_ROOT/logs/coc_histogram.png" \
  --n_lens_sets       8 \
  --seed              42

echo "Phase 4 done at $(date)"

MAX_COC=$(python -c "import json; print(json.load(open('$OUT_ROOT/logs/coc_stats.json'))['recommended_max_coc'])")
echo "Recommended max_coc = $MAX_COC"

# ---------------------------------------------------------------
# Phase 5: BokehNet rendering + GT focus maps (subset, 1 set)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 5: Render + Labels (subset, n_sets=$N_SETS, max_coc=$MAX_COC) ==="
for SCENE_IDX in $(seq 0 $((SCENE_COUNT-1))); do
  echo "--- Phase 5 scene_idx=${SCENE_IDX} ---"
  python scripts/05_render_and_labels.py \
    --out_root      "$OUT_ROOT" \
    --scene_index   "$SUBSET_INDEX" \
    --scene_idx     "$SCENE_IDX" \
    --max_frames    "$MAX_FRAMES" \
    --max_coc       "$MAX_COC" \
    --n_sets        "$N_SETS" \
    --steps         30 \
    --model_id      black-forest-labs/FLUX.1-dev \
    --lora_path     . \
    --lora_weight   bokehNet.safetensors

done

echo "Phase 5 done at $(date)"

# ---------------------------------------------------------------
# Phase 6: Validate outputs (subset)
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 6: Validation (subset) ==="
python scripts/06_validate_outputs.py \
  --out_root      "$OUT_ROOT" \
  --scene_index   "$SUBSET_INDEX" \
  --report        "$OUT_ROOT/logs/validation_report.json" \
  --n_sets        "$N_SETS" \
  --max_frames    "$MAX_FRAMES"

echo "Phase 6 done at $(date)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "======================================================"
echo " Multi-scene pipeline test COMPLETE"
echo " Finished: $(date)"
echo " Output directory: $OUT_ROOT"
echo " Validation report: $OUT_ROOT/logs/validation_report.json"
echo "======================================================"
