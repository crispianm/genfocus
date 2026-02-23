#!/usr/bin/env python3
"""Phase 3: Metric Depth with Video Depth Anything.

Runs Metric-Video-Depth-Anything-Large on sharpened frames to produce
temporally consistent, scale-accurate metric depth maps (metres).

Designed to run as a SLURM array job — each task processes one scene.

Usage:
    python scripts/03_depth.py \
        --out_root       ./output \
        --scene_index    ./output/scene_index.json \
        --scene_idx      0 \
        --vda_repo       ./Video-Depth-Anything \
        --checkpoint     ./checkpoints/metric_video_depth_anything_vitl.pth \
        --input_size     518
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
from natsort import natsorted
from PIL import Image
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


# ---------------------------------------------------------------------------
# VDA model loading
# ---------------------------------------------------------------------------

def load_vda_model(vda_repo: str, checkpoint: str, device: str):
    """Load Metric-Video-Depth-Anything-Large from local repo + checkpoint."""
    # Add the VDA repo to sys.path so its internal imports work
    vda_repo = Path(vda_repo).resolve()
    sys.path.insert(0, str(vda_repo))

    from video_depth_anything.video_depth import VideoDepthAnything

    model_config = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    }

    model = VideoDepthAnything(**model_config, metric=True)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    console.print(f"[green]VDA metric model loaded from {checkpoint}[/green]")
    return model


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frames_as_array(frame_dir: Path, max_frames: int = 0) -> np.ndarray:
    """Load frame_XXXX.png from a directory into (N, H, W, 3) uint8 array."""
    paths = natsorted(frame_dir.glob("frame_*.png"), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No frame_*.png in {frame_dir}")

    if max_frames and max_frames > 0:
        paths = paths[:max_frames]

    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Depth inference with chunking for long sequences
# ---------------------------------------------------------------------------

def infer_depth_chunked(
    model,
    frames: np.ndarray,
    target_fps: float,
    input_size: int,
    device: str,
    max_chunk: int = 256,
    overlap: int = 32,
    fp32: bool = False,
) -> np.ndarray:
    """Run VDA on frames, chunking long sequences with linear crossfade blending.

    Returns: (N, H, W) float32 depth in metres.
    """
    N = frames.shape[0]

    if N <= max_chunk:
        depths, _ = model.infer_video_depth(
            frames, target_fps, input_size=input_size, device=device, fp32=fp32,
        )
        return depths.astype(np.float32)

    # Chunked inference with overlap blending
    console.print(f"  Chunking: {N} frames into chunks of {max_chunk} with {overlap} overlap")
    step = max_chunk - overlap
    all_depths = np.zeros((N, frames.shape[1], frames.shape[2]), dtype=np.float64)
    weight_map = np.zeros(N, dtype=np.float64)

    for start in range(0, N, step):
        end = min(start + max_chunk, N)
        chunk = frames[start:end]

        chunk_depths, _ = model.infer_video_depth(
            chunk, target_fps, input_size=input_size, device=device, fp32=fp32,
        )

        # Linear crossfade weights within the overlap regions
        chunk_len = end - start
        w = np.ones(chunk_len, dtype=np.float64)

        if start > 0 and overlap > 0:
            # Ramp up at the beginning of non-first chunks
            ramp_len = min(overlap, chunk_len)
            w[:ramp_len] = np.linspace(0, 1, ramp_len)

        if end < N and overlap > 0:
            # Ramp down at the end of non-last chunks
            ramp_len = min(overlap, chunk_len)
            w[-ramp_len:] = np.linspace(1, 0, ramp_len)

        for i in range(chunk_len):
            all_depths[start + i] += chunk_depths[i].astype(np.float64) * w[i]
            weight_map[start + i] += w[i]

        if end >= N:
            break

    # Normalize by weights
    for i in range(N):
        if weight_map[i] > 0:
            all_depths[i] /= weight_map[i]

    return all_depths.astype(np.float32)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def compute_depth_stats(depths: np.ndarray) -> dict:
    """Compute depth statistics across all frames."""
    flat = depths.ravel()
    flat = flat[flat > 0]  # exclude zeros/negatives
    return {
        "d_min": float(np.min(flat)),
        "d_max": float(np.max(flat)),
        "d_median": float(np.median(flat)),
        "d_p5": float(np.percentile(flat, 5)),
        "d_p95": float(np.percentile(flat, 95)),
    }


def check_temporal_consistency(depths: np.ndarray, threshold: float = 0.15) -> bool:
    """Return True if there's a temporal consistency warning."""
    if len(depths) < 2:
        return False
    for t in range(len(depths) - 1):
        diff = np.mean(np.abs(depths[t + 1] - depths[t]))
        mean_d = np.mean(depths[t])
        if mean_d > 0 and (diff / mean_d) > threshold:
            return True
    return False


def patch_nan_inf(depths: np.ndarray, scene_id: str, log_path: Path) -> int:
    """Replace NaN/Inf with scene median. Returns count of patched pixels."""
    mask = ~np.isfinite(depths)
    count = int(mask.sum())
    if count > 0:
        median_val = float(np.nanmedian(depths[np.isfinite(depths)]))
        depths[mask] = median_val
        with open(log_path, "a") as f:
            f.write(f"{scene_id},{count},{median_val:.4f}\n")
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Metric depth via VDA")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--scene_index", type=str, required=True)
    parser.add_argument("--scene_idx", type=int, required=True, help="SLURM array index")
    parser.add_argument("--vda_repo", type=str, default="./Video-Depth-Anything",
                        help="Path to cloned Video-Depth-Anything repo")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/metric_video_depth_anything_vitl.pth")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--target_fps", type=float, default=30.0,
                        help="FPS hint for VDA temporal model")
    parser.add_argument("--max_chunk", type=int, default=256,
                        help="Max frames per VDA chunk")
    parser.add_argument("--overlap", type=int, default=32,
                        help="Frame overlap between chunks")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 inference (default: FP16)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="If >0, process only the first N frames of the scene (for quick tests)")
    args = parser.parse_args()

    out_root = Path(args.out_root)

    with open(args.scene_index) as f:
        scene_index = json.load(f)

    scene_ids = natsorted(scene_index.keys())
    if args.scene_idx >= len(scene_ids):
        console.print(f"[yellow]scene_idx {args.scene_idx} >= {len(scene_ids)}. Nothing to do.[/yellow]")
        return

    scene_id = scene_ids[args.scene_idx]
    sharp_dir = out_root / scene_id / "frames_sharp"
    depth_dir = out_root / scene_id / "depth_maps"
    sentinel = depth_dir / ".done"

    if sentinel.exists():
        console.print(f"[green]{scene_id}: depth_maps/.done exists — skipping.[/green]")
        return

    if not (sharp_dir / ".done").exists():
        console.print(f"[red]{scene_id}: frames_sharp/.done missing — Phase 2 incomplete. Skipping.[/red]")
        return

    depth_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vda_model(args.vda_repo, args.checkpoint, device)

    # Load sharpened frames
    console.print(f"[bold]Loading frames from {sharp_dir}[/bold]")
    frames = load_frames_as_array(sharp_dir, max_frames=args.max_frames)
    console.print(f"  Loaded {frames.shape[0]} frames at {frames.shape[1]}x{frames.shape[2]}")

    # Infer depth
    t0 = time.perf_counter()
    try:
        depths = infer_depth_chunked(
            model, frames,
            target_fps=args.target_fps,
            input_size=args.input_size,
            device=device,
            max_chunk=args.max_chunk,
            overlap=args.overlap,
            fp32=args.fp32,
        )
    except torch.cuda.OutOfMemoryError:
        console.print("[red]CUDA OOM during depth inference. Try reducing --max_chunk.[/red]")
        torch.cuda.empty_cache()
        return
    dt = time.perf_counter() - t0
    console.print(f"  Depth inference: {dt:.1f}s for {len(depths)} frames")

    # Patch NaN/Inf
    nan_log = logs_dir / "depth_nan_patches.csv"
    patched = patch_nan_inf(depths, scene_id, nan_log)
    if patched > 0:
        console.print(f"  [yellow]Patched {patched} NaN/Inf pixels.[/yellow]")

    # Ensure non-negative (metric depth should be >= 0)
    depths = np.maximum(depths, 0.0)

    # Save per-frame depth maps as float16 NPZ
    for t in range(len(depths)):
        np.savez_compressed(
            depth_dir / f"depth_{t:04d}.npz",
            depth=depths[t].astype(np.float16),
        )

    # Compute and save stats
    stats = compute_depth_stats(depths)
    with open(depth_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"  Stats: d_min={stats['d_min']:.2f}  d_max={stats['d_max']:.2f}  d_median={stats['d_median']:.2f}")

    # Sanity: depth range warnings
    warnings_file = logs_dir / "depth_sanity_warnings.txt"
    median = stats["d_median"]
    # Heuristic: outdoor > 50m or indoor < 0.5m is suspicious
    if median > 50.0 or median < 0.5:
        msg = f"{scene_id}: median depth = {median:.2f}m (suspicious)\n"
        console.print(f"  [yellow]{msg.strip()}[/yellow]")
        with open(warnings_file, "a") as f:
            f.write(msg)

    # Temporal consistency check
    if check_temporal_consistency(depths, threshold=0.15):
        tc_file = logs_dir / "depth_consistency_warnings.txt"
        msg = f"{scene_id}: temporal consistency check failed (>15% inter-frame variation)\n"
        console.print(f"  [yellow]{msg.strip()}[/yellow]")
        with open(tc_file, "a") as f:
            f.write(msg)

    # Write sentinel
    sentinel.touch()
    console.print(f"[green]{scene_id}: depth complete. Sentinel written.[/green]")


if __name__ == "__main__":
    main()
