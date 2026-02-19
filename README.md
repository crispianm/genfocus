# FocusMamba: HPC Data Generation Pipeline

Based on [Generative Refocusing](https://generative-refocusing.github.io/) (GenFocus).

This pipeline generates training data for FocusMamba on the Isambard GH200 cluster.
It ingests raw video frame datasets, sharpens frames via GenFocus-DeblurNet,
generates metric depth via Video-Depth-Anything, calibrates Circle of Confusion
parameters, then renders photorealistic defocus sequences via GenFocus-BokehNet
with co-registered ground-truth focus maps.

---

## Repository Structure

```
├── scripts/                   # Pipeline stages (numbered)
│   ├── 00_setup_env.sh        # Phase 0: Environment setup (ARM64/aarch64)
│   ├── 01_validate_dataset.py # Phase 1: Scan & validate raw data, assign splits
│   ├── 02_deblur.py           # Phase 2: DeblurNet sharpening (GPU, SLURM array)
│   ├── 03_depth.py            # Phase 3: Metric depth via VDA (GPU, SLURM array)
│   ├── 04_calibrate_coc.py    # Phase 4: CoC distribution calibration (CPU)
│   ├── 05_render_and_labels.py# Phase 5: BokehNet + GT focus maps (GPU, SLURM array)
│   ├── 06_validate_outputs.py # Phase 6: Output validation & final report
│   ├── run_pipeline.sh        # Orchestrator: Phases 1-4
│   └── run_pipeline_phase5.sh # Orchestrator: Phases 5-6 (after manual CoC check)
├── jobs/                      # SLURM job wrappers
│   ├── run_deblur.sh
│   ├── run_depth.sh
│   └── run_render.sh
├── Genfocus/                  # GenFocus core (FLUX pipeline + LoRA engine)
│   └── pipeline/flux.py
├── Video-Depth-Anything/      # (external) VDA repo clone (ignored by git)
├── checkpoints/               # (local) downloaded weights (ignored by git)
│   └── metric_video_depth_anything_vitl.pth
├── bokehNet.safetensors       # (local) BokehNet LoRA weights (ignored by git)
├── deblurNet.safetensors      # (local) DeblurNet LoRA weights (ignored by git)
├── data/                      # Raw input frames (scene_id/frame_XXXX.png)
├── output/                    # Pipeline output (see "Output Format" below)
├── old/                       # Archived original scripts (demo, video tools)
├── Project Overhaul.md        # Full specification document
├── requirements.txt
└── README.md
```

## Output Format (per scene)

```
output/{scene_id}/
  split.txt                    # "train" | "val" | "test"
  metadata.json                # CoC params per set {f, N, S_focus, max_coc}
  frames_sharp/                # DeblurNet output (PNG uint8)
    frame_0000.png, ...
  depth_maps/                  # VDA metric depth (NPZ float16, metres)
    depth_0000.npz, ...
    stats.json                 # {d_min, d_max, d_p5, d_p95, d_median}
  bokeh_renders/               # BokehNet output per CoC set (PNG uint8)
    set_00/bokeh_0000.png, ...
  focus_maps/                  # GT soft focus maps (NPZ float16, [0,1])
    set_00/focus_0000.npz, ...
```

---

## Quick Start

### 1. Environment Setup (Isambard GH200)

```bash
sbatch scripts/00_setup_env.sh
# or set up manually — see Project Overhaul.md Phase 0
```

### 2. Download Weights

```bash
# GenFocus model weights (already present if cloned properly)
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/bokehNet.safetensors
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/deblurNet.safetensors

# VDA metric depth weights
mkdir -p checkpoints
wget -O checkpoints/metric_video_depth_anything_vitl.pth \
  "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth"
```

### 3. Run the Full Pipeline

```bash
# Phases 1-4 (validate → deblur → depth → CoC calibration)
bash scripts/run_pipeline.sh

# Wait for Phase 4 to complete, then check:
cat output/logs/coc_stats.json

# Phases 5-6 (render + labels → validation)
bash scripts/run_pipeline_phase5.sh 0.063  # use recommended_max_coc value
```

### 4. Run Individual Stages (local / debugging)

```bash
# Phase 1: Validate dataset
python scripts/01_validate_dataset.py --raw_root ./data --out_root ./output

# Phase 2: Deblur a single scene
python scripts/02_deblur.py --out_root ./output --raw_root ./data \
    --scene_index ./output/scene_index.json --scene_idx 0 --steps 8

# Phase 3: Depth for a single scene
python scripts/03_depth.py --out_root ./output \
    --scene_index ./output/scene_index.json --scene_idx 0

# Phase 4: CoC calibration
python scripts/04_calibrate_coc.py --out_root ./output \
    --output_stats ./output/logs/coc_stats.json

# Phase 5: Render + labels for a single scene
python scripts/05_render_and_labels.py --out_root ./output \
    --scene_index ./output/scene_index.json --scene_idx 0 --max_coc 0.063

# Phase 6: Validate all outputs
python scripts/06_validate_outputs.py --out_root ./output \
    --scene_index ./output/scene_index.json --report ./output/logs/validation_report.json
```

---

## Re-entrancy & Error Recovery

All stages write `.done` sentinel files. Re-running any stage safely skips
already-completed scenes. Failed SLURM array tasks can be resubmitted without
`--dependency` — only scenes missing their sentinel will be re-processed.

---

## Citation

```bibtex
@article{Genfocus2025,
  title={Generative Refocusing: Flexible Defocus Control from a Single Image},
  author={Tuan Mu, Chun-Wei and Huang, Jia-Bin and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2512.16923},
  year={2025}
}
```
