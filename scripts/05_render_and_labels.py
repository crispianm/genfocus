#!/usr/bin/env python3
"""Phase 5: GT Focus Map Generation & BokehNet Rendering (Merged).

For each scene, samples CoC parameters, computes the GT focus map using
the calibrated max_coc, saves it, and immediately passes the SAME focus
map as the guidance signal to BokehNet. Inputs and labels are always
physically consistent.

Designed to run as a SLURM array job — each task processes one scene.

Usage:
    python scripts/05_render_and_labels.py \
        --out_root       ./output \
        --scene_index    ./output/scene_index.json \
        --scene_idx      0 \
        --max_coc        0.063 \
        --n_sets         4 \
        --steps          30 \
        --model_id       black-forest-labs/FLUX.1-dev \
        --lora_path      . \
        --lora_weight    bokehNet.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from natsort import natsorted
from rich.console import Console
from scipy.stats import beta as beta_dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

console = Console()


# ---------------------------------------------------------------------------
# CoC parameter sampling (matches focus_dataset.py exactly)
# ---------------------------------------------------------------------------

def sample_coc_params(
    depth_stats: dict,
    N_sets: int = 4,
    seed: int | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    d_min = max(depth_stats["d_min"], 0.1)
    d_max = depth_stats["d_max"]
    params = []
    for _ in range(N_sets):
        f = rng.uniform(24e-3, 85e-3)                          # focal length, metres
        log_N = rng.uniform(np.log(1.4), np.log(8.0))
        N = float(np.exp(log_N))                                # f-number, log-uniform
        u = beta_dist.rvs(2, 2, random_state=int(rng.integers(1e9)))
        S_focus = d_min + u * (d_max - d_min)                  # focus distance
        params.append({"f": float(f), "N": N, "S_focus": float(S_focus)})
    return params


# ---------------------------------------------------------------------------
# GT focus map computation (identical to dataloader — no approximations)
# ---------------------------------------------------------------------------

def compute_focus_map(
    depth_map: np.ndarray,
    f: float,
    N: float,
    S_focus: float,
    max_coc: float,
) -> np.ndarray:
    """
    depth_map : (H, W) float32, metres
    Returns   : focus_map (H, W) float16 in [0, 1]
    """
    depth_safe = np.maximum(depth_map, 1e-6)
    coc = (f ** 2 / (N * S_focus)) * np.abs(depth_safe - S_focus) / depth_safe
    coc_norm = np.clip(coc / max_coc, 0.0, 1.0)
    focus_map = (1.0 - coc_norm).astype(np.float16)
    return focus_map


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_flux_bokeh(
    model_id: str,
    lora_path: str,
    lora_weight: str,
    device: str,
    dtype: torch.dtype,
):
    """Load FLUX pipeline + BokehNet LoRA."""
    console.print(f"[bold]Loading FLUX model: {model_id}[/bold]")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    lora_full = os.path.join(lora_path, lora_weight)
    if not os.path.exists(lora_full):
        console.print(f"[red]ERROR: LoRA weights not found: {lora_full}[/red]")
        sys.exit(1)

    pipe.load_lora_weights(lora_path, weight_name=lora_weight, adapter_name="bokeh")
    pipe.set_adapters(["bokeh"])
    console.print("[green]BokehNet LoRA loaded.[/green]")
    return pipe


def run_bokeh_net(
    pipe: FluxPipeline,
    frame: Image.Image,
    focus_map: np.ndarray,
    steps: int,
    latents: torch.Tensor | None = None,
    seed: int = 42,
    disable_tiling: bool = False,
) -> Image.Image:
    """Run BokehNet on a single frame with the given focus_map guidance.

    focus_map: (H, W) float32 in [0, 1]. This is `1 - normalised_coc`.
    BokehNet expects defocus guidance = coc_norm = 1 - focus_map.
    """
    w, h = frame.size
    force_no_tile = min(w, h) < 512
    no_tiled_denoise = disable_tiling or force_no_tile

    # Build defocus condition: BokehNet expects the defocus map (CoC-normalised),
    # which is (1 - focus_map), as a 3-channel tensor in [0, 1].
    defocus = (1.0 - focus_map.astype(np.float32))
    defocus_t = torch.from_numpy(defocus).unsqueeze(0).float()
    cond_map = defocus_t.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, H, W)

    cond_img = Condition(frame, "bokeh")
    cond_dmf = Condition(cond_map, "bokeh", [0, 0], 1.0, No_preprocess=True)

    seed_everything(seed)
    gen = torch.Generator(device=pipe.device).manual_seed(seed)

    result = generate(
        pipe,
        height=h,
        width=w,
        prompt="an excellent photo with a large aperture",
        num_inference_steps=steps,
        conditions=[cond_img, cond_dmf],
        guidance_scale=1.0,
        kv_cache=False,
        generator=gen,
        latents=latents,
        NO_TILED_DENOISE=no_tiled_denoise,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: GT focus maps + BokehNet rendering")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--scene_index", type=str, required=True)
    parser.add_argument("--scene_idx", type=int, required=True, help="SLURM array index")
    parser.add_argument("--max_coc", type=float, required=True,
                        help="Calibrated max_coc from Phase 4")
    parser.add_argument("--n_sets", type=int, default=4,
                        help="Number of CoC parameter sets per scene")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps for BokehNet")
    parser.add_argument("--disable_tiling", action="store_true")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_path", type=str, default=".")
    parser.add_argument("--lora_weight", type=str, default="bokehNet.safetensors")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    max_coc = args.max_coc

    assert 1e-6 < max_coc < 1.0, f"max_coc={max_coc} looks wrong (expected 1e-6–1.0)"

    with open(args.scene_index) as f:
        scene_index = json.load(f)

    scene_ids = natsorted(scene_index.keys())
    if args.scene_idx >= len(scene_ids):
        console.print(f"[yellow]scene_idx {args.scene_idx} >= {len(scene_ids)}. Nothing to do.[/yellow]")
        return

    scene_id = scene_ids[args.scene_idx]
    scene_dir = out_root / scene_id
    depth_dir = scene_dir / "depth_maps"
    bokeh_sentinel = scene_dir / "bokeh_renders" / ".done"

    if bokeh_sentinel.exists():
        console.print(f"[green]{scene_id}: bokeh_renders/.done exists — skipping.[/green]")
        return

    if not (depth_dir / ".done").exists():
        console.print(f"[red]{scene_id}: depth_maps/.done missing — Phase 3 incomplete. Skipping.[/red]")
        return

    # Load depth stats
    stats_path = depth_dir / "stats.json"
    with open(stats_path) as f:
        depth_stats = json.load(f)

    # Load sharp frames
    sharp_dir = scene_dir / "frames_sharp"
    frame_paths = natsorted(sharp_dir.glob("frame_*.png"), key=lambda p: p.name)
    depth_paths = natsorted(depth_dir.glob("depth_*.npz"), key=lambda p: p.name)

    if len(frame_paths) != len(depth_paths):
        console.print(f"[red]{scene_id}: frame count ({len(frame_paths)}) != depth count ({len(depth_paths)}). Fix manually.[/red]")
        return

    n_frames = len(frame_paths)
    console.print(f"[bold]{scene_id}: {n_frames} frames, {args.n_sets} CoC sets, max_coc={max_coc}[/bold]")

    # Setup device & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = load_flux_bokeh(args.model_id, args.lora_path, args.lora_weight, device, dtype)

    # Sample CoC params (seeded on scene_id for reproducibility)
    param_seed = hash(scene_id) % (2 ** 32)
    param_sets = sample_coc_params(depth_stats, N_sets=args.n_sets, seed=param_seed)

    # Prepare shared latents (for temporal consistency within each set)
    sample_frame = Image.open(frame_paths[0])
    w, h = sample_frame.size

    seed_everything(42)
    gen = torch.Generator(device=pipe.device).manual_seed(1234)
    latents, _ = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=16,
        height=h,
        width=w,
        dtype=pipe.dtype,
        device=pipe.device,
        generator=gen,
        latents=None,
    )

    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    warnings_file = logs_dir / "focus_map_warnings.txt"

    for set_idx, params in enumerate(param_sets):
        focus_dir = scene_dir / "focus_maps" / f"set_{set_idx:02d}"
        bokeh_dir = scene_dir / "bokeh_renders" / f"set_{set_idx:02d}"
        focus_dir.mkdir(parents=True, exist_ok=True)
        bokeh_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"  Set {set_idx}: f={params['f']*1000:.1f}mm  N={params['N']:.1f}  S_focus={params['S_focus']:.2f}m")

        for t in range(n_frames):
            focus_out = focus_dir / f"focus_{t:04d}.npz"
            bokeh_out = bokeh_dir / f"bokeh_{t:04d}.png"

            # Skip if both outputs exist
            if focus_out.exists() and bokeh_out.exists():
                continue

            # Load depth
            depth = np.load(depth_paths[t])["depth"].astype(np.float32)

            # Compute GT focus map
            focus_map = compute_focus_map(depth, max_coc=max_coc, **params)

            # Save focus map
            np.savez_compressed(focus_out, focus_map=focus_map)

            # Sanity check
            fm_mean = float(focus_map.astype(np.float32).mean())
            if fm_mean < 0.2 or fm_mean > 0.8:
                msg = f"{scene_id} set_{set_idx:02d} frame_{t:04d}: focus_map.mean()={fm_mean:.3f}\n"
                with open(warnings_file, "a") as wf:
                    wf.write(msg)

            # Load frame image
            frame = Image.open(frame_paths[t]).convert("RGB")

            # BokehNet render using the SAME focus map
            try:
                with torch.no_grad():
                    bokeh_frame = run_bokeh_net(
                        pipe, frame,
                        focus_map.astype(np.float32),
                        steps=args.steps,
                        latents=latents,
                        seed=42,
                        disable_tiling=args.disable_tiling,
                    )
                bokeh_frame.save(bokeh_out)
            except torch.cuda.OutOfMemoryError:
                console.print(f"  [red]OOM at set {set_idx} frame {t} — skipping.[/red]")
                torch.cuda.empty_cache()
                continue

            if (t + 1) % 10 == 0 or (t + 1) == n_frames:
                console.print(f"    [{t+1}/{n_frames}] done")

    # Write authoritative metadata
    metadata = {
        "sets": [{**p, "max_coc": max_coc} for p in param_sets]
    }
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Write sentinel
    (scene_dir / "bokeh_renders").mkdir(parents=True, exist_ok=True)
    bokeh_sentinel.touch()
    console.print(f"[green]{scene_id}: render + labels complete. Sentinel written.[/green]")


if __name__ == "__main__":
    main()
