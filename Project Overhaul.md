# FocusMamba: HPC Data Generation Pipeline (v2)
## Implementation Plan for Isambard GH200 Cluster Agent
==========================================================

## Changelog from v1

| # | Issue | Fix |
|---|---|---|
| 1 | **ARM64 Architecture Trap (Phase 0)** | Standard PyPI PyTorch wheels are x86_64-only and will silently fail on GH200's Grace CPU. Use `conda-forge` or an NGC Singularity container per Isambard docs. |
| 2 | **Circular CoC Guidance Mismatch (Phases 4 & 5)** | Phase 5a (CoC calibration) moved to immediately after Phase 3. Phases 4 and 5b merged. BokehNet and GT focus maps now always use the same calibrated `max_coc`. |
| 3 | **Storage and I/O Bottleneck (Phases 4 & 5)** | All `.npy` arrays saved as `float16` via `np.savez_compressed`. Estimated storage reduction: ~75% on depth/focus data. |
| 4 | **Wrong VDA model** | Updated to `depth-anything/Metric-Video-Depth-Anything-Large` throughout. |

---

## Overview

This pipeline ingests a raw video frame dataset, enforces sharpness via
GenFocus-DeblurNet, generates temporally consistent metric depth via
Metric-Video-Depth-Anything (VDA), calibrates the Circle of Confusion
distribution, then renders photorealistic defocus sequences via
GenFocus-BokehNet and saves co-registered GT focus maps — all using the
same physical parameters so inputs and labels are always consistent.

### Target Output Per Scene

```
dataset_root/
  {scene_id}/
    frames_sharp/             ← DeblurNet output (PNG uint8)
      frame_0000.png
      frame_0001.png
      ...
    depth_maps/               ← VDA metric depth (NPZ float16, metres)
      depth_0000.npz
      depth_0001.npz
      ...
    depth_maps/stats.json     ← {d_min, d_max, d_p5, d_p95, d_median}
    bokeh_renders/            ← BokehNet output per CoC set (PNG uint8)
      set_00/
        bokeh_0000.png
        ...
    focus_maps/               ← GT soft focus maps (NPZ float16, [0,1])
      set_00/
        focus_0000.npz
        ...
    metadata.json             ← CoC params per set {f, N, S_focus, max_coc}
    split.txt                 ← "train" | "val" | "test"
```

**Note on ROI**: The `roi` field used by `focus_dataset.py` is intentionally
absent from this pipeline. It is a training-time construct — either a fixed
full-frame box `[0,0,1,1]`, a randomly sampled crop region, or the output
of a separate interactive/saliency stage. The data preprocessing pipeline
has no role in generating it.

---

## Hardware & Environment Assumptions

- **Cluster**: Isambard-AI (BriCS), GH200 nodes
- **CPU**: NVIDIA Grace — **ARM Neoverse V2 (aarch64)**, NOT x86_64
- **GPU**: NVIDIA Hopper H100 96 GB HBM3
- **OS**: Ubuntu 22.04 LTS
- **Scheduler**: SLURM
- **Storage**: NVMe scratch at `$SCRATCH`, permanent at `$STORE`

---

## PHASE 0: Environment Setup
-------------------------------

### [ ] `pipeline/00_setup_env.sh`

> **CRITICAL — READ BEFORE RUNNING**
>
> The GH200's Grace CPU is **aarch64 (ARM64)**, not x86_64. Standard PyPI
> PyTorch wheels from `download.pytorch.org/whl/cu1XX` are compiled for
> x86_64 and **will not work** — the install either fails outright or falls
> back silently to a CPU-only ARM build. Per the
> [Isambard ML Packages docs](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/),
> there are three valid approaches. **Use Option A or B.**

#### Option A — `conda-forge` (Recommended)

`conda-forge` provides aarch64 + CUDA-enabled PyTorch builds. The install
**must be run on a compute node** (not the login node) so CUDA is present
during the build process.

```bash
#!/bin/bash
#SBATCH --job-name=focusmamba_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -euo pipefail

CONDA_ENV=focusmamba_data

# Create env and install aarch64-native PyTorch from conda-forge
conda create -n $CONDA_ENV python=3.10 -y
conda activate $CONDA_ENV

# This provides aarch64 + CUDA + NumPy support
srun --gpus 1 conda install -y conda-forge::pytorch conda-forge::torchvision

# Verify GPU is available before proceeding
srun --gpus 1 python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA OK:', torch.version.cuda)"

# Additional pip dependencies (pure-Python packages are architecture-agnostic)
pip install transformers==4.40.0 accelerate safetensors timm einops \
            numpy opencv-python-headless Pillow tqdm rich \
            matplotlib seaborn scipy

# GenFocus (adjust to actual repo location)
# git clone https://github.com/<org>/GenFocus $SCRATCH/GenFocus
# pip install -e $SCRATCH/GenFocus

# Pre-cache Metric-Video-Depth-Anything-Large weights
python -c "
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
AutoImageProcessor.from_pretrained('depth-anything/Metric-Video-Depth-Anything-Large')
AutoModelForDepthEstimation.from_pretrained('depth-anything/Metric-Video-Depth-Anything-Large')
print('VDA weights cached.')
"

echo "Environment setup complete."
```

#### Option B — NGC Singularity Container (Alternative)

If `conda-forge` PyTorch lacks a package you need (e.g., a custom CUDA
extension), use an NGC PyTorch container instead. NGC containers ship
aarch64 + CUDA pre-built and have been validated on Isambard.

```bash
# Pull the container to your scratch (do this once on the login node)
singularity pull $SCRATCH/containers/pytorch_25.05-py3.sif \
    docker://nvcr.io/nvidia/pytorch:25.05-py3

# Verify
srun --gpus 1 singularity run --nv \
    $SCRATCH/containers/pytorch_25.05-py3.sif \
    python3 -c "import torch; print(torch.cuda.is_available())"

# Install additional deps inside the container at runtime via --pip-install,
# or build a derived Singularity image with a %post block.
```

#### Option C — PyPI cu128 wheels (Only if conda-forge fails)

Per Isambard docs, the cu128 index *does* ship aarch64 wheels as of 2025:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Verify immediately after: `python3 -c "import torch; print(torch.cuda.is_available())"`.
If `False`, abort and switch to Option A or B — a CPU-only install will make
all subsequent pipeline stages silently run on CPU.

---

## PHASE 1: Dataset Validation & Scene Index
---------------------------------------------

### [ ] `pipeline/01_validate_dataset.py`

Scans the raw dataset, validates frame integrity, builds a scene index,
and assigns deterministic 80/10/10 train/val/test splits at the scene level.
Must complete before any subsequent stage. No GPU required.

**SLURM**:
```bash
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
```

**Inputs**:
- `--raw_root`: `raw_root/{scene_id}/{frame_XXXX}.{jpg|png}`
- `--out_root`: output dataset root (will be created)
- `--min_frames`: minimum frames per scene (default: 16)

**Logic**:

```python
import os, json, random
from pathlib import Path
from PIL import Image

VALID_EXTS = {'.jpg', '.jpeg', '.png'}

def scan_scene(scene_dir):
    return sorted(f for f in scene_dir.iterdir()
                  if f.suffix.lower() in VALID_EXTS)

def validate_scene(scene_dir, frames, min_frames):
    """Returns (ok: bool, reason: str)."""
    if len(frames) < min_frames:
        return False, f"only {len(frames)} frames"
    # Check for corruption and shape consistency
    shapes = set()
    for f in frames:
        try:
            with Image.open(f) as img:
                img.verify()
                shapes.add(img.size)
        except Exception as e:
            return False, f"corrupt frame {f.name}: {e}"
    if len(shapes) > 1:
        return False, f"inconsistent frame sizes: {shapes}"
    return True, "ok"

def assign_splits(scene_ids, seed=42):
    """Deterministic 80/10/10 split, seeded on sorted scene list."""
    ids = sorted(scene_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    splits = {}
    for i, sid in enumerate(ids):
        if i < int(0.8 * n):   splits[sid] = 'train'
        elif i < int(0.9 * n): splits[sid] = 'val'
        else:                   splits[sid] = 'test'
    return splits

# Write:
#   {out_root}/scene_index.json  ← {scene_id: {frames: [...], split: "train"}}
#   {out_root}/{scene_id}/split.txt  ← per scene, for dataloader
```

**Outputs**: `{out_root}/scene_index.json`, per-scene `split.txt`

---

## PHASE 2: Sharpening with GenFocus-DeblurNet
------------------------------------------------

### [ ] `pipeline/02_deblur.py`

Runs GenFocus-DeblurNet on all frames. Re-entrant: skips scenes where
`frames_sharp/.done` sentinel exists.

**SLURM** (`jobs/run_deblur.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=deblur_%a
#SBATCH --array=0-{N_SCENES}%32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
```

**Key implementation notes**:

- Process in overlapping temporal batches of 16 frames (stride 8) so
  DeblurNet has boundary context. Average overlapping predictions.
- Output: `frames_sharp/{frame_XXXX}.png` as uint8 PNG.
  If DeblurNet returns float32 in [0,1]:
  `(arr * 255).round().clip(0, 255).astype(np.uint8)`
- Log per-scene PSNR (sharp vs. raw) to `{out_root}/logs/deblur_psnr.csv`.
- On CUDA OOM: halve `batch_size`, retry once, then log as failed and skip.
- Sentinel: `{out_root}/{scene_id}/frames_sharp/.done`

---

## PHASE 3: Metric Depth with Video Depth Anything
----------------------------------------------------

### [ ] `pipeline/03_depth.py`

Runs **Metric-Video-Depth-Anything-Large** on sharpened frames to produce
temporally consistent, scale-accurate metric depth maps (metres).

**Model**: `depth-anything/Metric-Video-Depth-Anything-Large`
(HuggingFace). Use the Metric variant — the non-metric variant produces
affine-invariant relative depth, which cannot be used with the physical CoC
formula that requires absolute distances in metres. The correct .pth file (metric_video_depth_anything_vitl.pth) has already been downloaded into ./checkpoints.

**SLURM** (`jobs/run_depth.sh`):
```bash
#SBATCH --job-name=depth_%a
#SBATCH --array=0-{N_SCENES}%16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
```

**Key implementation notes**:

- Feed the full scene as a video sequence (VDA is designed for temporal
  input; frame-by-frame inference degrades temporal consistency severely).
- If scene > 300 frames: chunk into 256-frame windows with 32-frame overlap;
  blend the overlap region with a linear crossfade.
- **Storage**: save depth as `float16` via `np.savez_compressed`:
  ```python
  np.savez_compressed(
      out_dir / f'depth_{t:04d}.npz',
      depth=depth_arr.astype(np.float16)
  )
  # Load in dataloader as: np.load(path)['depth'].astype(np.float32)
  ```
  This avoids precision loss on depth values (max ~100 m; float16 range
  covers this) while reducing per-file size by ~75% vs. uncompressed float32.
- Compute and save `depth_maps/stats.json` per scene after all frames:
  ```json
  {"d_min": 0.31, "d_max": 18.4, "d_median": 3.2, "d_p5": 0.8, "d_p95": 12.1}
  ```
  `d_min` and `d_max` are used by the CoC sampler in Phase 4.
- **Sanity check**: median depth for indoor scenes should be 2–5 m, outdoor
  scenes 5–50 m. Log a warning to `logs/depth_sanity_warnings.txt` if outside
  these bounds.
- **Temporal consistency check**: if `mean(|depth[t+1] - depth[t]|) / mean(depth[t]) > 0.15`
  for any scene, flag it in `logs/depth_consistency_warnings.txt`.
- Sentinel: `{out_root}/{scene_id}/depth_maps/.done`

---

## PHASE 4: CoC Distribution Calibration
------------------------------------------

> **This phase moved here from v1's Phase 5a.**
> It must run after depth generation (Phase 3) and before BokehNet rendering
> (Phase 4 in v1, now Phase 5). This ordering is the fix for the circular
> CoC mismatch: `max_coc` is calibrated once from real depth data, then used
> consistently for *both* BokehNet guidance and GT focus map generation.

### [ ] `pipeline/04_calibrate_coc.py`

Scans all saved `depth_maps/` (no GPU required) to determine the global
`max_coc` normalisation value that will be used in all downstream steps.

**SLURM**:
```bash
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
```

**Usage**:
```bash
python pipeline/04_calibrate_coc.py \
    --out_root        $OUT_ROOT \
    --sample_fraction 0.2 \
    --output_plot     $OUT_ROOT/logs/coc_histogram.png \
    --output_stats    $OUT_ROOT/logs/coc_stats.json
```

**Logic**:

```python
# CoC formula (matches dataloader exactly):
def compute_raw_coc(depth, f, N, S_focus):
    depth_safe = np.maximum(depth.astype(np.float32), 1e-6)
    return (f**2 / (N * S_focus)) * np.abs(depth_safe - S_focus) / depth_safe

# For a random 20% sample of scenes:
#   1. Load depth stats (d_min, d_max) from depth_maps/stats.json
#   2. Draw several representative (f, N, S_focus) param sets per scene
#   3. Compute CoC over all pixels; collect into a global flat array
#   4. Report percentiles; recommend max_coc = p95

# Output coc_stats.json:
# {
#   "p50":  0.021,
#   "p90":  0.047,
#   "p95":  0.063,  ← USE THIS as max_coc in the next phase
#   "p99":  0.112,
#   "max":  0.438,
#   "recommended_max_coc": 0.063
# }
```

> **MANUAL STEP REQUIRED after this script completes.**
>
> Read `$OUT_ROOT/logs/coc_stats.json` and pass `recommended_max_coc`
> as `--max_coc` to Phase 5. Also record this value in `config.yaml`.
> Do not proceed to Phase 5 until this step is done.

---

## PHASE 5: GT Focus Map Generation & BokehNet Rendering (Merged)
-------------------------------------------------------------------

> **Phases 4 and 5b from v1 are now a single, atomic phase.**
>
> For each scene, the agent samples CoC parameters, computes the GT focus
> map using the calibrated `max_coc`, saves it, and immediately passes the
> *same* focus map as the guidance signal to BokehNet. There is no
> intermediate placeholder — inputs and labels are always physically consistent.

### [ ] `pipeline/05_render_and_labels.py`

**SLURM** (`jobs/run_render.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=render_%a
#SBATCH --array=0-{N_SCENES}%16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
```

**CoC parameter sampling** (must match `focus_dataset.py` exactly):

```python
import numpy as np
from scipy.stats import beta as beta_dist

def sample_coc_params(depth_stats, N_sets=4, seed=None):
    rng = np.random.default_rng(seed)
    d_min, d_max = depth_stats['d_min'], depth_stats['d_max']
    params = []
    for _ in range(N_sets):
        f      = rng.uniform(24e-3, 85e-3)                             # focal length, metres
        log_N  = rng.uniform(np.log(1.4), np.log(8.0))
        N      = np.exp(log_N)                                         # f-number, log-uniform
        u      = beta_dist.rvs(2, 2, random_state=int(rng.integers(1e9)))
        S_focus = d_min + u * (d_max - d_min)                         # focus distance, Beta(2,2)
        params.append({'f': float(f), 'N': float(N), 'S_focus': float(S_focus)})
    return params
```

**GT focus map computation** (identical to dataloader — no approximations):

```python
def compute_focus_map(depth_map, f, N, S_focus, max_coc):
    """
    depth_map : (H, W) float32, metres
    Returns   : focus_map (H, W) float16 in [0,1]
    """
    depth_safe = np.maximum(depth_map, 1e-6)
    coc        = (f**2 / (N * S_focus)) * np.abs(depth_safe - S_focus) / depth_safe
    coc_norm   = np.clip(coc / max_coc, 0.0, 1.0)
    focus_map  = (1.0 - coc_norm).astype(np.float16)
    return focus_map
```

**Rendering loop**:

```python
for scene_id in assigned_scenes:
    depth_stats = load_json(out_root / scene_id / 'depth_maps' / 'stats.json')
    # Seed on scene_id hash for reproducibility across re-runs
    param_sets  = sample_coc_params(depth_stats, N_sets=4, seed=hash(scene_id) % (2**32))

    for set_idx, params in enumerate(param_sets):
        focus_dir  = out_root / scene_id / 'focus_maps'  / f'set_{set_idx:02d}'
        bokeh_dir  = out_root / scene_id / 'bokeh_renders' / f'set_{set_idx:02d}'
        focus_dir.mkdir(parents=True, exist_ok=True)
        bokeh_dir.mkdir(parents=True, exist_ok=True)

        for t, (frame, depth) in enumerate(load_scene(scene_id)):
            # --- GT focus map (saved first, always) ---
            focus_map = compute_focus_map(
                depth, max_coc=max_coc, **params   # max_coc from Phase 4
            )
            np.savez_compressed(
                focus_dir / f'focus_{t:04d}.npz',
                focus_map=focus_map          # float16
            )

            # --- BokehNet render uses the SAME focus_map ---
            # BokehNet expects: frame (H,W,3) uint8, guidance (H,W) float32
            bokeh_frame = run_bokeh_net(bokeh_model, frame, focus_map.astype(np.float32))
            save_png(bokeh_frame, bokeh_dir / f'bokeh_{t:04d}.png')

            # Sanity check
            if focus_map.mean() < 0.2 or focus_map.mean() > 0.8:
                log_warning(scene_id, set_idx, t, f"focus_map.mean()={focus_map.mean():.3f}")

    # Write authoritative metadata (includes calibrated max_coc)
    save_json(
        {'sets': [{**p, 'max_coc': max_coc} for p in param_sets]},
        out_root / scene_id / 'metadata.json'
    )
```

**Storage note**: Both `focus_maps` and `depth_maps` arrays are saved as
`float16` via `np.savez_compressed`. The dataloader must cast back to
`float32` on load:
```python
depth = np.load(path)['depth'].astype(np.float32)
focus = np.load(path)['focus_map'].astype(np.float32)
```
This decompression cost is negligible vs. the I/O savings (~75% smaller
files, far fewer IOPS under concurrent dataloader workers).

- Sentinel: `{out_root}/{scene_id}/bokeh_renders/.done`

---

## PHASE 6: Output Validation
-------------------------------

### [ ] `pipeline/06_validate_outputs.py`

Checks all expected outputs exist and are loadable. Produces a final report.

```bash
python pipeline/06_validate_outputs.py \
    --out_root     $OUT_ROOT \
    --scene_index  $OUT_ROOT/scene_index.json \
    --report       $OUT_ROOT/logs/validation_report.json
```

**Checks per scene**:

| Check | Pass criterion |
|---|---|
| `frames_sharp/` count | == raw frame count |
| `depth_maps/` NPZ count | == raw frame count |
| `bokeh_renders/set_XX/` PNG count | == raw frame count × N_sets |
| `focus_maps/set_XX/` NPZ count | == raw frame count × N_sets |
| `metadata.json` | valid JSON; has `sets[*].{f, N, S_focus, max_coc}` |
| `split.txt` | one of: train, val, test |
| Focus map loadable & in range | `np.load(p)['focus_map']` all in [0,1] |
| Depth map loadable & positive | `np.load(p)['depth']` all > 0 |
| Frame (H,W) consistency | all modalities same spatial dims |

**Output** `validation_report.json`:
```json
{
  "total_scenes": 1200,
  "passing_scenes": 1187,
  "failed_scenes": ["scene_042"],
  "failure_reasons": {"scene_042": "focus_map NPZ missing: 3 files in set_01"},
  "split_counts": {"train": 960, "val": 120, "test": 120},
  "total_frames": 184320,
  "total_gb_on_disk": 198.4
}
```

Failed scenes should be re-queued through their failing stage (check which
sentinel `.done` file is absent), not silently excluded.

---

## PHASE 7: Pipeline Orchestration
-------------------------------------

### [ ] `pipeline/run_pipeline.sh`

Submits all SLURM jobs in dependency order. The deliberate manual breakpoint
between Phases 4 and 5 is enforced via a split between two shell scripts.

```bash
#!/bin/bash
set -euo pipefail

export OUT_ROOT=$SCRATCH/focusmamba_dataset
export RAW_DATA_ROOT=$STORE/raw_video_frames

# --- Phase 1: Validate and index ---
python pipeline/01_validate_dataset.py \
    --raw_root   $RAW_DATA_ROOT \
    --out_root   $OUT_ROOT \
    --min_frames 16

N_SCENES=$(python -c "
import json; idx = json.load(open('$OUT_ROOT/scene_index.json'))
print(len(idx) - 1)   # 0-indexed for SLURM arrays
")

# --- Phase 2: Deblur ---
JOB_DEBLUR=$(sbatch --parsable \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --array=0-${N_SCENES}%32 \
    jobs/run_deblur.sh)

# --- Phase 3: Depth ---
JOB_DEPTH=$(sbatch --parsable \
    --dependency=afterok:$JOB_DEBLUR \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --array=0-${N_SCENES}%16 \
    jobs/run_depth.sh)

# --- Phase 4: CoC calibration (CPU, fast) ---
JOB_COC=$(sbatch --parsable \
    --dependency=afterok:$JOB_DEPTH \
    --partition=cpu --ntasks=1 --mem=32G --time=01:00:00 \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --wrap="python pipeline/04_calibrate_coc.py \
            --out_root $OUT_ROOT \
            --sample_fraction 0.2 \
            --output_stats $OUT_ROOT/logs/coc_stats.json \
            --output_plot  $OUT_ROOT/logs/coc_histogram.png")

echo "================================================================"
echo " Phase 4 (CoC calibration) submitted: job $JOB_COC"
echo ""
echo " MANUAL STEP REQUIRED — do not run Phase 5 until:"
echo "   1. Job $JOB_COC completes."
echo "   2. You read: $OUT_ROOT/logs/coc_stats.json"
echo "   3. You verify the recommended_max_coc value looks plausible"
echo "      (should be in range 0.02–0.20 for typical scenes)."
echo "   4. You run:"
echo "        bash pipeline/run_pipeline_phase5_onwards.sh <max_coc>"
echo "      e.g.:"
echo "        bash pipeline/run_pipeline_phase5_onwards.sh 0.063"
echo "================================================================"
```

### [ ] `pipeline/run_pipeline_phase5_onwards.sh`

```bash
#!/bin/bash
set -euo pipefail

MAX_COC=${1:?"Usage: $0 <max_coc e.g. 0.063>"}

# Validate the value is a plausible float
python -c "
v = float('$MAX_COC')
assert 0.005 < v < 1.0, f'max_coc={v} looks wrong — expected 0.005–1.0'
print(f'max_coc={v} accepted.')
"

# --- Phase 5: Render + GT labels (atomic) ---
JOB_RENDER=$(sbatch --parsable \
    --export=ALL,OUT_ROOT=$OUT_ROOT,MAX_COC=$MAX_COC \
    --array=0-${N_SCENES}%16 \
    jobs/run_render.sh)

# --- Phase 6: Validate ---
sbatch \
    --dependency=afterok:$JOB_RENDER \
    --partition=cpu --ntasks=1 --mem=16G --time=00:30:00 \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --wrap="python pipeline/06_validate_outputs.py \
            --out_root    $OUT_ROOT \
            --scene_index $OUT_ROOT/scene_index.json \
            --report      $OUT_ROOT/logs/validation_report.json"
```

---

## Interfaces to `focus_dataset.py`

The following contracts must hold between pipeline outputs and the
FocusMamba dataloader:

| Pipeline file | Dataloader usage |
|---|---|
| `frames_sharp/{frame_XXXX}.png` uint8 | `cv2.imread` → normalise to float32 [0,1] |
| `depth_maps/depth_{XXXX}.npz` float16 key `'depth'` | `np.load(p)['depth'].astype(np.float32)` |
| `focus_maps/set_XX/focus_{XXXX}.npz` float16 key `'focus_map'` | `np.load(p)['focus_map'].astype(np.float32)` — used as GT label |
| `metadata.json` `.sets[i].{f, N, S_focus, max_coc}` | available for future conditioning |
| `split.txt` | `ClipSampler` filters by this for train/val/test |
| Consistent `(H, W)` across all modalities in a scene | required by collate_fn |

---

## Revised Disk Space Estimates

Assumes 1,200 scenes × 128 frames average × 256×256 resolution × 4 CoC sets.

| Output type | Format | Est. size/frame | Total |
|---|---|---|---|
| Sharp frames | PNG uint8 | ~100 KB | ~15 GB |
| Depth maps | NPZ float16 (compressed) | ~65 KB | ~10 GB |
| Bokeh renders × 4 | PNG uint8 | ~100 KB × 4 | ~62 GB |
| Focus maps × 4 | NPZ float16 (compressed) | ~65 KB × 4 | ~40 GB |
| **Total** | | | **~127 GB** |

Provision **200 GB** on `$SCRATCH` to allow for temp files. This is a ~53%
reduction from the v1 estimate of ~273 GB, primarily from float16 compression
of depth and focus maps.

---

## Error Recovery

- All scripts write `.done` sentinel files — re-running any stage is safe.
- SLURM array jobs that fail mid-way can be resubmitted without
  `--dependency` — the sentinel check ensures only failed scenes re-run.
- If a BokehNet output is all-black or wrong shape, delete the `.done`
  sentinel in `bokeh_renders/` and requeue that scene's array index.
  **Do not delete the `focus_maps/.done` sentinel** — the GT is correct
  regardless of the render; only the render needs redoing.
- Depth maps with NaN or Inf pixels must be patched before saving.
  Fill with the scene-level median and log how many pixels were patched
  to `logs/depth_nan_patches.csv`.