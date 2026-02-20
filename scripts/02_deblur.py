#!/usr/bin/env python3
"""Phase 2: Sharpening with GenFocus-DeblurNet.

Runs GenFocus-DeblurNet (FLUX + LoRA) on all frames of each scene.
Re-entrant: skips scenes where frames_sharp/.done sentinel exists.

Designed to run as a SLURM array job — each task processes one scene.

Usage:
    python scripts/02_deblur.py \
        --out_root   ./output \
        --raw_root   ./data \
        --scene_index ./output/scene_index.json \
        --scene_idx  0 \
        --steps      8 \
        --model_id   black-forest-labs/FLUX.1-dev \
        --lora_path  . \
        --lora_weight deblurNet.safetensors
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from rich.console import Console

# --- Add project root to path so Genfocus package is importable --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

console = Console()

VALID_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resize_and_pad_image(img: Image.Image, target_long_side: int = 0) -> Image.Image:
    """Resize so the long side = target_long_side, then crop to multiples of 16."""
    w, h = img.size
    if target_long_side and target_long_side > 0:
        target_max = int(target_long_side)
        if w >= h:
            new_w = target_max
            new_h = int(h * (target_max / w))
        else:
            new_h = target_max
            new_w = int(w * (target_max / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        final_w = max((new_w // 16) * 16, 16)
        final_h = max((new_h // 16) * 16, 16)
        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        return img.crop((left, top, left + final_w, top + final_h))

    final_w = ((w + 15) // 16) * 16
    final_h = ((h + 15) // 16) * 16
    if final_w == w and final_h == h:
        return img
    return img.resize((final_w, final_h), Image.LANCZOS)


def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_flux_deblur(model_id: str, lora_path: str, lora_weight: str, device: str, dtype: torch.dtype):
    """Load FLUX pipeline + DeblurNet LoRA."""
    console.print(f"[bold]Loading FLUX model: {model_id}[/bold]")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    lora_full = os.path.join(lora_path, lora_weight)
    if not os.path.exists(lora_full):
        console.print(f"[red]ERROR: LoRA weights not found: {lora_full}[/red]")
        sys.exit(1)

    pipe.load_lora_weights(lora_path, weight_name=lora_weight, adapter_name="deblurring")
    pipe.set_adapters(["deblurring"])
    console.print("[green]DeblurNet LoRA loaded.[/green]")
    return pipe


def run_deblur_frame(
    pipe: FluxPipeline,
    frame: Image.Image,
    steps: int,
    disable_tiling: bool = False,
) -> Image.Image:
    """Run DeblurNet on a single frame, returning the sharpened image."""
    w, h = frame.size
    force_no_tile = min(w, h) < 512
    no_tiled_denoise = disable_tiling or force_no_tile

    condition_black = Image.new("RGB", (w, h), (0, 0, 0))
    cond0 = Condition(condition_black, "deblurring", [0, 32], 1.0)
    cond1 = Condition(frame, "deblurring", [0, 0], 1.0)

    seed_everything(42)
    result = generate(
        pipe,
        height=h,
        width=w,
        prompt="a sharp photo with everything in focus",
        num_inference_steps=steps,
        conditions=[cond0, cond1],
        NO_TILED_DENOISE=no_tiled_denoise,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# Temporal averaging helpers
# ---------------------------------------------------------------------------

def deblur_with_temporal_averaging(
    pipe: FluxPipeline,
    frame_images: list[Image.Image],
    steps: int,
    disable_tiling: bool,
    window_size: int = 16,
    stride: int = 8,
) -> list[np.ndarray]:
    """Run DeblurNet on overlapping temporal windows, average overlapping outputs.

    Each frame is processed independently through DeblurNet (since it's a
    per-frame model), but frames that fall within multiple windows get their
    outputs averaged in pixel space. This reduces temporal flicker.

    Args:
        pipe: FLUX pipeline with DeblurNet LoRA loaded
        frame_images: preprocessed PIL images for the full scene
        steps: diffusion steps
        disable_tiling: disable tiling optimization
        window_size: number of frames per window
        stride: step between window starts (< window_size for overlap)

    Returns:
        List of averaged sharp frames as uint8 numpy arrays (H, W, 3)
    """
    N = len(frame_images)
    if N == 0:
        return []

    # If scene is smaller than one window, no averaging needed
    if N <= window_size:
        results = []
        for frame in frame_images:
            with torch.no_grad():
                sharp = run_deblur_frame(pipe, frame, steps, disable_tiling)
            results.append(np.array(sharp))
        return results

    # Accumulator buffers for weighted averaging
    h_sample, w_sample = np.array(frame_images[0]).shape[:2]
    accum = np.zeros((N, h_sample, w_sample, 3), dtype=np.float64)
    weight = np.zeros(N, dtype=np.float64)

    # Track which frames have been processed to avoid redundant inference
    processed_cache: dict[int, np.ndarray] = {}

    for win_start in range(0, N, stride):
        win_end = min(win_start + window_size, N)

        for t in range(win_start, win_end):
            if t not in processed_cache:
                try:
                    with torch.no_grad():
                        sharp = run_deblur_frame(pipe, frame_images[t], steps, disable_tiling)
                    processed_cache[t] = np.array(sharp).astype(np.float64)
                except torch.cuda.OutOfMemoryError:
                    console.print(f"  [red]OOM at frame {t}, using input as fallback[/red]")
                    torch.cuda.empty_cache()
                    processed_cache[t] = np.array(frame_images[t]).astype(np.float64)

            accum[t] += processed_cache[t]
            weight[t] += 1.0

        # Free cache entries that won't appear in any future window
        earliest_future = win_start + stride
        for cached_t in list(processed_cache.keys()):
            if cached_t < earliest_future:
                del processed_cache[cached_t]

        if win_end >= N:
            break

    # Normalise
    results = []
    for t in range(N):
        if weight[t] > 0:
            avg = (accum[t] / weight[t]).clip(0, 255).astype(np.uint8)
        else:
            avg = np.array(frame_images[t]).astype(np.uint8)
        results.append(avg)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: DeblurNet sharpening")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--scene_index", type=str, required=True)
    parser.add_argument("--scene_idx", type=int, required=True, help="Index into scene list (for SLURM array)")
    parser.add_argument("--steps", type=int, default=8, help="Diffusion steps (default: 8)")
    parser.add_argument("--resize_long_side", type=int, default=0, help="Resize long side (0 = keep)")
    parser.add_argument("--disable_tiling", action="store_true")
    parser.add_argument("--window_size", type=int, default=16,
                        help="Temporal window size for overlap averaging (default: 16)")
    parser.add_argument("--stride", type=int, default=8,
                        help="Stride between temporal windows (default: 8)")
    parser.add_argument("--no_temporal_avg", action="store_true",
                        help="Disable temporal averaging (process each frame independently)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="If >0, process only the first N frames of the scene (for quick tests)")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_path", type=str, default=".")
    parser.add_argument("--lora_weight", type=str, default="deblurNet.safetensors")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    raw_root = Path(args.raw_root)

    with open(args.scene_index) as f:
        scene_index = json.load(f)

    scene_ids = natsorted(scene_index.keys())
    if args.scene_idx >= len(scene_ids):
        console.print(f"[yellow]scene_idx {args.scene_idx} >= {len(scene_ids)} scenes. Nothing to do.[/yellow]")
        return

    scene_id = scene_ids[args.scene_idx]
    info = scene_index[scene_id]
    sharp_dir = out_root / scene_id / "frames_sharp"
    sentinel = sharp_dir / ".done"

    if sentinel.exists():
        console.print(f"[green]{scene_id}: frames_sharp/.done exists — skipping.[/green]")
        return

    sharp_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipe = load_flux_deblur(args.model_id, args.lora_path, args.lora_weight, device, dtype)

    # PSNR log
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    psnr_log = logs_dir / "deblur_psnr.csv"

    frame_paths = [raw_root / fp for fp in info["frames"]]
    if args.max_frames and args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
        console.print(
            f"[bold]Processing scene: {scene_id} ({len(frame_paths)} frames; capped by --max_frames)[/bold]"
        )
    else:
        console.print(f"[bold]Processing scene: {scene_id} ({len(frame_paths)} frames)[/bold]")

    # Check which frames still need processing
    needed = []
    for t in range(len(frame_paths)):
        out_path = sharp_dir / f"frame_{t:04d}.png"
        if not out_path.exists():
            needed.append(t)

    if not needed:
        console.print(f"  All frames exist, writing sentinel.")
        sentinel.touch()
        return

    # Load and preprocess all frames
    console.print(f"  Loading {len(frame_paths)} frames...")
    frame_images = []
    for fp in frame_paths:
        raw_img = Image.open(fp).convert("RGB")
        proc_img = resize_and_pad_image(raw_img, args.resize_long_side)
        frame_images.append(proc_img)

    psnr_rows = []

    if args.no_temporal_avg:
        # Simple per-frame processing
        console.print("  Mode: per-frame (no temporal averaging)")
        for t in needed:
            out_path = sharp_dir / f"frame_{t:04d}.png"
            try:
                t0 = time.perf_counter()
                with torch.no_grad():
                    sharp_img = run_deblur_frame(pipe, frame_images[t], args.steps, args.disable_tiling)
                dt = time.perf_counter() - t0

                sharp_img.save(out_path)
                raw_arr = np.array(frame_images[t])
                sharp_arr = np.array(sharp_img)
                p = psnr(raw_arr, sharp_arr)
                psnr_rows.append([scene_id, t, f"{p:.2f}", f"{dt:.2f}"])

                if (t + 1) % 5 == 0 or t == needed[-1]:
                    console.print(f"  [{t+1}/{len(frame_paths)}] PSNR={p:.2f} dB  ({dt:.1f}s)")
            except torch.cuda.OutOfMemoryError:
                console.print(f"[red]CUDA OOM at frame {t} — skipping.[/red]")
                torch.cuda.empty_cache()
            except Exception as e:
                console.print(f"[red]Error at frame {t}: {e}[/red]")
                traceback.print_exc()
    else:
        # Temporal averaging mode
        console.print(f"  Mode: temporal averaging (window={args.window_size}, stride={args.stride})")
        t0 = time.perf_counter()
        sharp_arrays = deblur_with_temporal_averaging(
            pipe, frame_images,
            steps=args.steps,
            disable_tiling=args.disable_tiling,
            window_size=args.window_size,
            stride=args.stride,
        )
        dt_total = time.perf_counter() - t0
        console.print(f"  Total deblur time: {dt_total:.1f}s ({dt_total / len(frame_paths):.1f}s/frame)")

        for t in range(len(sharp_arrays)):
            out_path = sharp_dir / f"frame_{t:04d}.png"
            if out_path.exists():
                continue
            Image.fromarray(sharp_arrays[t]).save(out_path)

            raw_arr = np.array(frame_images[t])
            p = psnr(raw_arr, sharp_arrays[t])
            psnr_rows.append([scene_id, t, f"{p:.2f}", ""])

            if (t + 1) % 10 == 0 or (t + 1) == len(frame_paths):
                console.print(f"  [{t+1}/{len(frame_paths)}] PSNR={p:.2f} dB")

    # Write PSNR log (append mode, safe for concurrent writes)
    if psnr_rows:
        write_header = not psnr_log.exists()
        with open(psnr_log, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["scene_id", "frame_idx", "psnr_db", "time_s"])
            writer.writerows(psnr_rows)

    # Place sentinel
    sentinel.touch()
    console.print(f"[green]{scene_id}: deblur complete. Sentinel written.[/green]")


if __name__ == "__main__":
    main()
