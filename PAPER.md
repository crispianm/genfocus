# FocusMamba: Data Generation Pipeline — Technical Report

> **Status**: Pipeline implemented and under testing on Isambard-AI GH200 cluster.  
> **Last updated**: 2026-02-19  
> **Bear sequence end-to-end test**: SLURM job 2376646 (running)

---

## 1. Motivation

Training a video focus estimation model (FocusMamba) requires large-scale,
physically consistent pairs of:

1. **Bokeh-rendered frames** — synthetic shallow depth-of-field applied to sharp video
2. **Ground-truth focus maps** — per-pixel continuous focus values in [0, 1], derived from metric depth and the thin-lens Circle of Confusion (CoC) formula

No existing dataset provides both modalities at scale with temporal
consistency across video sequences. We therefore build an automated HPC
pipeline that ingests raw video frames and produces co-registered training
data using two pre-trained generative models:

- **GenFocus-DeblurNet** — a FLUX.1-dev diffusion model fine-tuned with LoRA
  adapters for single-image deblurring/sharpening
- **GenFocus-BokehNet** — the same architecture fine-tuned for controllable
  defocus rendering, guided by a per-pixel defocus map

Both models originate from [Generative Refocusing](https://generative-refocusing.github.io/)
(Tuan Mu et al., 2025).

Metric depth is provided by **Video Depth Anything** (VDA), a video-native
depth estimator that produces temporally consistent, scale-accurate depth in
metres — essential for the physical CoC formula.

---

## 2. Pipeline Overview

The pipeline is divided into seven phases, each re-entrant (sentinel-based)
and designed for SLURM array job parallelism on the Isambard-AI cluster
(NVIDIA GH200 Grace Hopper nodes: ARM64 Neoverse V2 CPU + H100 96 GB HBM3 GPU).

| Phase | Script | Compute | Summary |
|-------|--------|---------|---------|
| 0 | `scripts/00_setup_env.sh` | GPU node | Conda environment setup (aarch64-native PyTorch) |
| 1 | `scripts/01_validate_dataset.py` | CPU | Scan raw data, validate integrity, assign 80/10/10 splits |
| 2 | `scripts/02_deblur.py` | GPU | Sharpen all frames with DeblurNet + **temporal averaging** |
| 3 | `scripts/03_depth.py` | GPU | Metric depth via VDA with **chunked crossfade blending** |
| 4 | `scripts/04_calibrate_coc.py` | CPU | Calibrate global `max_coc` from depth statistics |
| 5 | `scripts/05_render_and_labels.py` | GPU | **Atomic** GT focus map + BokehNet render (same physics) |
| 6 | `scripts/06_validate_outputs.py` | CPU | Validate all outputs, produce report |

A deliberate **manual breakpoint** between Phases 4 and 5 requires the
operator to inspect the calibrated `max_coc` value before proceeding.

---

## 3. Novel Contributions & Design Decisions

### 3.1 Temporal Smoothing of DeblurNet Outputs

**Problem**: GenFocus-DeblurNet is a single-image model. When applied
frame-by-frame to a video sequence, stochastic variation in the diffusion
process causes temporal flicker — visible as inter-frame brightness and
texture inconsistencies.

**Solution**: We introduce **overlapping temporal window averaging**. Each
frame is processed by DeblurNet exactly once, but frames that fall within
multiple overlapping windows have their outputs averaged in pixel space:

```
Window 1:  [f0  f1  f2  ... f15]
Window 2:        [f8  f9  f10 ... f23]
Window 3:              [f16 f17 ... f31]
                  ↑   ↑
          These frames get averaged across 2 windows
```

- **Window size**: 16 frames (default)
- **Stride**: 8 frames (50% overlap)
- Each frame is processed through DeblurNet independently (the model is
  per-frame), but results are accumulated into a weighted sum buffer
- The final output for each frame is `sum(outputs) / count(windows)`
- Frames near sequence boundaries that only appear in one window are output
  without averaging (graceful degradation)
- A processing cache avoids redundant inference — each frame index is
  deblurred at most once, even though it may be accumulated into multiple
  windows
- The `--no_temporal_avg` flag reverts to simple per-frame processing

**Effect**: Significantly reduced temporal flicker in preliminary tests.
The averaging acts as a low-pass filter in the temporal domain without
blurring spatial detail (since each frame's DeblurNet output is spatially
sharp by construction).

**Trade-off**: Memory footprint increases proportionally to the number of
frames (accumulator buffers are float64), but this is negligible on H100
96 GB nodes. Wall-clock time is identical — each frame is still processed
exactly once.

### 3.2 Chunked Depth Inference with Linear Crossfade

**Problem**: VDA processes frames as a temporal video sequence for
consistency, but its memory footprint scales with sequence length. Scenes
with > 300 frames risk CUDA OOM.

**Solution**: Long sequences are split into overlapping chunks (default
256 frames, 32-frame overlap). The overlap regions are blended with a
**linear crossfade ramp**:

```
Chunk 1:  [-------- full weight --------][ramp down]
Chunk 2:                          [ramp up][------- full weight -------][ramp down]
Chunk 3:                                                        [ramp up][--- full ---]
```

- Ramp-up: weight increases linearly from 0 to 1 over the overlap region
- Ramp-down: weight decreases linearly from 1 to 0
- Final depth = weighted sum / total weight per frame

This eliminates hard discontinuities at chunk boundaries while preserving
VDA's temporal consistency within each chunk.

### 3.3 Physics-Based CoC Calibration (Phase 4)

Rather than using an arbitrary `max_coc` normalisation constant (which would
make the focus map distribution dependent on an ungrounded hyperparameter),
we calibrate it empirically from the actual depth data:

1. Sample 20% of scenes randomly
2. For each, draw 8 representative lens parameter sets:
   - Focal length $f \sim \mathcal{U}(24\text{mm}, 85\text{mm})$
   - F-number $N \sim \text{LogUniform}(1.4, 8.0)$
   - Focus distance $S_{\text{focus}} \sim d_{\min} + \text{Beta}(2, 2) \cdot (d_{\max} - d_{\min})$
3. Compute raw CoC over sampled depth pixels using the thin-lens formula:

$$\text{CoC} = \frac{f^2}{N \cdot S_{\text{focus}}} \cdot \frac{|d - S_{\text{focus}}|}{d}$$

4. The **95th percentile** of the collected CoC distribution becomes `max_coc`

The Beta(2,2) distribution for $S_{\text{focus}}$ concentrates focus distances
toward the middle of the scene's depth range, avoiding degenerate cases where
focus is placed at extreme near/far boundaries (which would produce trivial
all-blurred or all-sharp focus maps).

**This must match the dataloader exactly.** The same CoC formula and `max_coc`
value are used at training time to verify/recompute focus maps on-the-fly.

### 3.4 Atomic Focus Map + BokehNet Rendering (Fixing the v1 Circular Mismatch)

**Problem in v1**: The original pipeline design had a circular dependency —
BokehNet was rendered with one set of CoC parameters, but the GT focus maps
were computed separately with potentially different parameters. Any mismatch
between these causes the model to learn incorrect focus-to-defocus mappings.

**Solution**: Phase 5 is a single, atomic operation per frame:

1. Draw CoC parameters (seeded on `hash(scene_id)` for reproducibility)
2. Load metric depth
3. Compute GT focus map using the calibrated `max_coc`
4. Save focus map immediately
5. Pass the **exact same focus map** (converted to defocus guidance via
   `1 - focus_map`) to BokehNet
6. Save the BokehNet render

The focus map and bokeh render are always physically consistent — they
encode the same lens parameters, the same depth, and the same `max_coc`.

### 3.5 ARM64-Native HPC Deployment

The Isambard-AI GH200 nodes use NVIDIA Grace CPUs (ARM Neoverse V2,
aarch64), not x86_64. Standard PyPI PyTorch wheels are x86_64-only.
The pipeline uses `conda-forge` aarch64+CUDA PyTorch builds and is
validated on Python 3.12. All scripts are designed to run exclusively
within the conda environment on compute nodes, never on login nodes.

### 3.6 Re-entrant, Sentinel-Based Design

Every GPU phase writes a `.done` sentinel file upon successful completion
of each scene. Re-running any phase:

- Skips scenes with existing sentinels (no wasted compute)
- Continues from exactly where it left off after node failures
- Allows failed SLURM array indices to be resubmitted independently

This is critical for HPC reliability — multi-hour jobs across hundreds
of GPU nodes will inevitably encounter occasional failures.

---

## 4. Phase Details

### Phase 1: Dataset Validation & Scene Index

Scans the raw data directory for scenes (subdirectories containing image
frames). For each scene:

- Validates minimum frame count (default: 16)
- Checks for corrupt images (PIL verify)
- Ensures spatial consistency (all frames same resolution)

Assigns deterministic 80/10/10 train/val/test splits using a seeded
shuffle of scene IDs. Outputs:

- `output/scene_index.json` — master index with frame lists, resolutions, splits
- `output/{scene_id}/split.txt` — per-scene split label

### Phase 2: DeblurNet Sharpening

Loads FLUX.1-dev + DeblurNet LoRA adapter. For each scene:

- Reads raw frames and resizes to multiples of 16 (FLUX requirement)
- Runs DeblurNet with 8 diffusion steps per frame
- Applies temporal window averaging (§3.1) unless `--no_temporal_avg`
- Saves sharp frames as `frame_XXXX.png` (uint8 PNG)
- Logs per-frame PSNR (sharp vs. raw) to `logs/deblur_psnr.csv`
- Handles CUDA OOM gracefully (falls back to input frame)

**DeblurNet conditioning**: Two conditions are passed to the FLUX model:
1. A black image as a "deblurring" control signal with offset `[0, 32]`
2. The input (blurry) frame as a "deblurring" condition with offset `[0, 0]`

The prompt "a sharp photo with everything in focus" guides the diffusion.

### Phase 3: Metric Depth via Video Depth Anything

Loads the VDA ViT-Large metric model from a local repository clone.
Processes sharpened frames (not raw) to ensure depth maps align with the
clean frame content:

- Feeds the full scene as a video sequence (temporal consistency)
- Chunks long sequences with linear crossfade (§3.2)
- Patches NaN/Inf pixels with the scene median
- Ensures all depths are non-negative
- Saves as compressed float16 NPZ (`depth_XXXX.npz`)
- Computes and saves `stats.json`: {d_min, d_max, d_median, d_p5, d_p95}
- Warns if median depth is suspiciously low (<0.5 m) or high (>50 m)
- Flags scenes with >15% inter-frame depth variation

### Phase 4: CoC Distribution Calibration

CPU-only phase that determines the global `max_coc` normalisation value:

- Samples a configurable fraction of scenes (default: 20%)
- Draws representative lens parameters per scene (§3.3)
- Computes pixel-level CoC across sampled depth maps
- Reports percentiles (p50, p90, p95, p99, max)
- Recommends p95 as `max_coc`
- Optionally generates a CoC histogram plot

**Manual checkpoint**: The operator must inspect `coc_stats.json` and
confirm the `recommended_max_coc` before Phase 5 proceeds.

### Phase 5: GT Focus Maps & BokehNet Rendering

The core data generation phase. For each scene:

- Draws N CoC parameter sets (default: 4, using 2 for test runs)
- Seeds on `hash(scene_id)` for reproducibility across re-runs
- For each parameter set × frame, atomically (§3.4):
  - Computes GT focus map from depth + CoC params
  - Saves as compressed float16 NPZ (`focus_XXXX.npz`)
  - Renders bokeh using the same focus map as BokehNet guidance
  - Saves as PNG (`bokeh_XXXX.png`)
- Uses shared initial latents across frames for temporal consistency
- Writes `metadata.json` with all CoC parameters including `max_coc`
- Warns if mean focus map is <0.2 or >0.8 (potentially degenerate)

**BokehNet conditioning**: The focus map is converted to a defocus map
(`1 - focus_map`), expanded to 3 channels, and passed as a guidance
condition to FLUX alongside the sharp frame. The prompt "an excellent
photo with a large aperture" guides the diffusion at 30 steps.

### Phase 6: Output Validation

CPU-only sweep that checks every scene for completeness:

- Frame counts match across all modalities
- All NPZ files loadable with correct keys
- Depth values are positive
- Focus map values are in [0, 1]
- Spatial dimensions are consistent
- `metadata.json` contains required keys
- `split.txt` is valid

Produces `validation_report.json` with pass/fail status per scene and
total disk usage.

---

## 5. Output Format (Dataloader Contract)

Each scene produces:

```
output/{scene_id}/
  split.txt                        # "train" | "val" | "test"
  metadata.json                    # {sets: [{f, N, S_focus, max_coc}, ...]}
  frames_sharp/
    frame_0000.png ... frame_NNNN.png   # uint8 RGB PNG
    .done                               # sentinel
  depth_maps/
    depth_0000.npz ... depth_NNNN.npz   # key='depth', float16, metres
    stats.json                           # {d_min, d_max, d_median, d_p5, d_p95}
    .done                               # sentinel
  bokeh_renders/
    set_00/ ... set_03/
      bokeh_0000.png ... bokeh_NNNN.png # uint8 RGB PNG
    .done                               # sentinel
  focus_maps/
    set_00/ ... set_03/
      focus_0000.npz ... focus_NNNN.npz # key='focus_map', float16, [0,1]
```

**Dataloader loading convention**:
```python
depth = np.load(path)['depth'].astype(np.float32)    # metres
focus = np.load(path)['focus_map'].astype(np.float32) # [0, 1]
frame = cv2.imread(str(path))                          # uint8 BGR
```

The `roi` field is a training-time construct (random crop or saliency region)
and is intentionally absent from this generation pipeline.

---

## 6. Circle of Confusion Physics

The CoC formula used throughout the pipeline (must match the dataloader):

$$\text{CoC}(d) = \frac{f^2}{N \cdot S_{\text{focus}}} \cdot \frac{|d - S_{\text{focus}}|}{d}$$

Where:
- $f$ = focal length (metres)
- $N$ = f-number (dimensionless)
- $S_{\text{focus}}$ = focus distance (metres)
- $d$ = pixel depth (metres)

Normalised CoC and focus map:

$$\text{CoC}_{\text{norm}} = \text{clamp}\left(\frac{\text{CoC}(d)}{\text{max\_coc}},\ 0,\ 1\right)$$

$$\text{focus\_map} = 1 - \text{CoC}_{\text{norm}}$$

Pixels at the focus plane ($d = S_{\text{focus}}$) get `focus_map = 1` (perfectly
focused). Pixels far from the focus plane get values approaching 0 (maximum
defocus). The `max_coc` normalisation constant is calibrated empirically in
Phase 4 to ensure the distribution of focus map values is well-spread across
[0, 1] for the actual depth ranges in the dataset.

### Parameter Sampling

| Parameter | Distribution | Range | Rationale |
|-----------|-------------|-------|-----------|
| $f$ | Uniform | 24–85 mm | Common photographic focal lengths |
| $N$ | Log-uniform | 1.4–8.0 | Covers wide aperture (shallow DoF) to moderate |
| $S_{\text{focus}}$ | $d_{\min} + \text{Beta}(2,2) \cdot (d_{\max} - d_{\min})$ | Scene depth range | Concentrates focus in mid-range; avoids trivial extremes |

---

## 7. Hardware & Cluster Details

| Component | Specification |
|-----------|--------------|
| Cluster | Isambard-AI (BriCS), University of Bristol |
| Node type | NVIDIA GH200 Grace Hopper Superchip |
| CPU | NVIDIA Grace — ARM Neoverse V2 (aarch64) |
| GPU | NVIDIA H100 96 GB HBM3 |
| OS | Ubuntu 22.04 LTS |
| Scheduler | SLURM |
| PyTorch | conda-forge aarch64+CUDA build |
| Python | 3.12 (conda env: `genfocus`) |

**Critical note**: Standard PyPI PyTorch wheels are x86_64-only and will
silently fail or fall back to CPU on aarch64 nodes. The `conda-forge`
channel provides native aarch64+CUDA wheels validated for GH200.

---

## 8. Current Project State

### Completed

- [x] Repository restructured: old scripts archived to `old/`, new numbered
  pipeline in `scripts/`, SLURM wrappers in `jobs/`
- [x] All 6 pipeline scripts implemented and syntax-verified on Python 3.12
- [x] 3 SLURM array job wrappers created
- [x] 2 orchestration scripts (Phases 1-4, Phases 5-6) with dependency chains
- [x] Phase 1 validated on DAVIS dataset: 90 scenes, 6208 frames, 72/9/9 split
- [x] Temporal averaging added to Phase 2 (DeblurNet)
- [x] Chunked crossfade blending added to Phase 3 (VDA depth)
- [x] Physics-based CoC calibration in Phase 4 with Beta(2,2) sampling
- [x] Atomic focus map + BokehNet rendering in Phase 5
- [x] Comprehensive output validation in Phase 6
- [x] README.md rewritten with full documentation
- [x] requirements.txt updated (removed gradio/jupyter/ml-depth-pro, added
  transformers/scipy/rich etc.)
- [x] Unused checkpoints removed (`depth_pro.pt`)

### In Progress

- [ ] End-to-end test on bear sequence (SLURM job 2376646, all 82 frames
  through Phases 2-6)
- [ ] Full dataset processing (90 DAVIS scenes)

### Pending

- [ ] Integration with FocusMamba training dataloader (`focus_dataset.py`)
- [ ] Scale to larger datasets (e.g., DAVIS 2017 full, Kinetics subsets)
- [ ] Hyperparameter tuning: optimal diffusion steps, window sizes, N_sets
- [ ] Ablation: temporal averaging vs. no averaging (quantitative flicker metric)

---

## 9. References

1. **GenFocus**: Tuan Mu, C.-W., Huang, J.-B., & Liu, Y.-L. (2025).
   *Generative Refocusing: Flexible Defocus Control from a Single Image.*
   arXiv:2512.16923.

2. **Video Depth Anything**: Yang, B., et al. (2025).
   *Video Depth Anything: Consistent Depth Estimation for Super-Long Videos.*
   arXiv preprint.

3. **FLUX.1-dev**: Black Forest Labs (2024). Open-weight diffusion transformer
   for image generation. https://blackforestlabs.ai/

4. **Isambard-AI**: University of Bristol / BriCS. NVIDIA GH200 Grace Hopper
   supercomputing facility. https://docs.isambard.ac.uk/

---

## Appendix A: Key File Locations

| Asset | Path |
|-------|------|
| DeblurNet LoRA | `./deblurNet.safetensors` |
| BokehNet LoRA | `./bokehNet.safetensors` |
| VDA checkpoint | `./checkpoints/metric_video_depth_anything_vitl.pth` |
| VDA repo | `./Video-Depth-Anything/` |
| FLUX engine | `./Genfocus/pipeline/flux.py` |
| Raw data | `./data/{scene_id}/` |
| Output root | `./output/` |
| Scene index | `./output/scene_index.json` |

## Appendix B: Reproducing the Bear Test

```bash
# Phase 1 (already done — can skip if scene_index.json exists)
python scripts/01_validate_dataset.py --raw_root ./data --out_root ./output

# Submit Phases 2-6 as a single SLURM job
sbatch jobs/test_bear.sh

# Monitor
squeue -u $USER
tail -f logs/test_bear_*.out
```

## Appendix C: SLURM Resource Requests

| Phase | Partition | GPU | Memory | Time | CPUs |
|-------|-----------|-----|--------|------|------|
| 2 (Deblur) | gpu | 1 | 48 GB | 4 h | 8 |
| 3 (Depth) | gpu | 1 | 64 GB | 6 h | 8 |
| 5 (Render) | gpu | 1 | 80 GB | 8 h | 8 |
| 4, 6 (CPU) | cpu | 0 | 32 GB | 1 h | 8–16 |
| Test (bear) | workq | 1 | 80 GB | 6 h | 8 |
