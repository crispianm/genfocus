#!/usr/bin/env python3
"""Phase 1: Dataset Validation & Scene Index.

Scans the raw dataset, validates frame integrity, builds a scene index,
and assigns deterministic 80/10/10 train/val/test splits at the scene level.

No GPU required.

Usage:
    python scripts/01_validate_dataset.py \
        --raw_root ./data \
        --out_root ./output \
        --min_frames 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from natsort import natsorted
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()

VALID_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Scanning & validation
# ---------------------------------------------------------------------------

def scan_scene(scene_dir: Path) -> List[Path]:
    """Return naturally sorted list of image files in *scene_dir*."""
    return natsorted(
        (f for f in scene_dir.iterdir()
         if f.suffix.lower() in VALID_EXTS and f.is_file()),
        key=lambda p: p.name,
    )


def validate_scene(
    scene_dir: Path,
    frames: List[Path],
    min_frames: int,
) -> Tuple[bool, str]:
    """Check frame count, corruption, and spatial consistency."""
    if len(frames) < min_frames:
        return False, f"only {len(frames)} frames (min={min_frames})"

    shapes: set[Tuple[int, int]] = set()
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


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(scene_ids: List[str], seed: int = 42) -> Dict[str, str]:
    """Deterministic 80/10/10 split, seeded on the sorted scene list."""
    ids = sorted(scene_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    splits: Dict[str, str] = {}
    for i, sid in enumerate(ids):
        if i < int(0.8 * n):
            splits[sid] = "train"
        elif i < int(0.9 * n):
            splits[sid] = "val"
        else:
            splits[sid] = "test"
    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Validate dataset & build scene index")
    parser.add_argument("--raw_root", type=str, required=True, help="Root dir of raw frames: raw_root/{scene_id}/frame_XXXX.{jpg|png}")
    parser.add_argument("--out_root", type=str, required=True, help="Output dataset root (will be created)")
    parser.add_argument("--min_frames", type=int, default=16, help="Minimum frames per scene (default: 16)")
    parser.add_argument("--split_seed", type=int, default=42, help="RNG seed for train/val/test splits")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    if not raw_root.is_dir():
        console.print(f"[red]ERROR: raw_root not found: {raw_root}[/red]")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    # Discover scenes (immediate subdirectories of raw_root that contain images)
    scene_dirs = natsorted(
        (d for d in raw_root.iterdir()
         if d.is_dir() and not d.name.startswith(".")),
        key=lambda p: p.name,
    )

    console.print(f"[bold]Scanning {len(scene_dirs)} candidate scenes under {raw_root}...[/bold]")

    scene_index: Dict[str, dict] = {}
    rejected: List[Tuple[str, str]] = []

    for sd in scene_dirs:
        frames = scan_scene(sd)
        ok, reason = validate_scene(sd, frames, args.min_frames)
        if not ok:
            rejected.append((sd.name, reason))
            continue

        w, h = Image.open(frames[0]).size
        scene_index[sd.name] = {
            "frames": [str(f.relative_to(raw_root)) for f in frames],
            "num_frames": len(frames),
            "resolution": [w, h],
        }

    if not scene_index:
        console.print("[red]No valid scenes found. Exiting.[/red]")
        sys.exit(1)

    # Assign splits
    splits = assign_splits(list(scene_index.keys()), seed=args.split_seed)
    for sid in scene_index:
        scene_index[sid]["split"] = splits[sid]

    # Write scene_index.json
    index_path = out_root / "scene_index.json"
    with open(index_path, "w") as f:
        json.dump(scene_index, f, indent=2)

    # Write per-scene split.txt and create output dirs
    for sid, info in scene_index.items():
        scene_out = out_root / sid
        scene_out.mkdir(parents=True, exist_ok=True)
        (scene_out / "split.txt").write_text(info["split"])

    # Summary
    split_counts = {"train": 0, "val": 0, "test": 0}
    for info in scene_index.values():
        split_counts[info["split"]] += 1

    table = Table(title="Dataset Validation Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Scenes scanned", str(len(scene_dirs)))
    table.add_row("Scenes accepted", str(len(scene_index)))
    table.add_row("Scenes rejected", str(len(rejected)))
    table.add_row("Train / Val / Test", f"{split_counts['train']} / {split_counts['val']} / {split_counts['test']}")
    table.add_row("Total frames", str(sum(v["num_frames"] for v in scene_index.values())))
    table.add_row("Scene index", str(index_path))
    console.print(table)

    if rejected:
        console.print(f"\n[yellow]Rejected scenes ({len(rejected)}):[/yellow]")
        for sid, reason in rejected:
            console.print(f"  {sid}: {reason}")

    console.print("[green]Phase 1 complete.[/green]")


if __name__ == "__main__":
    main()
