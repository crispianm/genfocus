#!/usr/bin/env python3
"""Phase 6: Output Validation.

Checks all expected outputs exist and are loadable. Produces a final report.

Usage:
    python scripts/06_validate_outputs.py \
        --out_root     ./output \
        --scene_index  ./output/scene_index.json \
        --report       ./output/logs/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()


def get_disk_usage_gb(path: Path) -> float:
    """Recursively compute disk usage in GB."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 3)


def validate_scene(
    scene_id: str,
    scene_dir: Path,
    expected_frames: int,
    n_sets: int,
) -> tuple[bool, list[str]]:
    """Validate all outputs for a single scene. Returns (pass, [reasons])."""
    errors: list[str] = []

    # --- frames_sharp ---
    sharp_dir = scene_dir / "frames_sharp"
    sharp_count = len(list(sharp_dir.glob("frame_*.png"))) if sharp_dir.exists() else 0
    if sharp_count != expected_frames:
        errors.append(f"frames_sharp: {sharp_count}/{expected_frames}")

    # --- depth_maps ---
    depth_dir = scene_dir / "depth_maps"
    depth_count = len(list(depth_dir.glob("depth_*.npz"))) if depth_dir.exists() else 0
    if depth_count != expected_frames:
        errors.append(f"depth_maps: {depth_count}/{expected_frames}")

    # --- depth stats ---
    stats_file = depth_dir / "stats.json"
    if not stats_file.exists():
        errors.append("depth_maps/stats.json missing")
    else:
        try:
            with open(stats_file) as f:
                stats = json.load(f)
            for key in ("d_min", "d_max", "d_median", "d_p5", "d_p95"):
                if key not in stats:
                    errors.append(f"stats.json missing key: {key}")
        except json.JSONDecodeError:
            errors.append("stats.json invalid JSON")

    # --- bokeh_renders and focus_maps per set ---
    for set_idx in range(n_sets):
        set_name = f"set_{set_idx:02d}"
        bokeh_dir = scene_dir / "bokeh_renders" / set_name
        focus_dir = scene_dir / "focus_maps" / set_name

        bokeh_count = len(list(bokeh_dir.glob("bokeh_*.png"))) if bokeh_dir.exists() else 0
        focus_count = len(list(focus_dir.glob("focus_*.npz"))) if focus_dir.exists() else 0

        if bokeh_count != expected_frames:
            errors.append(f"bokeh_renders/{set_name}: {bokeh_count}/{expected_frames}")
        if focus_count != expected_frames:
            errors.append(f"focus_maps/{set_name}: {focus_count}/{expected_frames}")

    # --- metadata.json ---
    meta_file = scene_dir / "metadata.json"
    if not meta_file.exists():
        errors.append("metadata.json missing")
    else:
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            sets = meta.get("sets", [])
            if len(sets) != n_sets:
                errors.append(f"metadata.json: {len(sets)} sets (expected {n_sets})")
            for i, s in enumerate(sets):
                for key in ("f", "N", "S_focus", "max_coc"):
                    if key not in s:
                        errors.append(f"metadata.json sets[{i}] missing key: {key}")
        except json.JSONDecodeError:
            errors.append("metadata.json invalid JSON")

    # --- split.txt ---
    split_file = scene_dir / "split.txt"
    if not split_file.exists():
        errors.append("split.txt missing")
    else:
        split_val = split_file.read_text().strip()
        if split_val not in ("train", "val", "test"):
            errors.append(f"split.txt invalid: '{split_val}'")

    # --- Spot-check loadability (sample first frame if present) ---
    if depth_count > 0:
        first_depth = natsorted(depth_dir.glob("depth_*.npz"), key=lambda p: p.name)[0]
        try:
            d = np.load(first_depth)["depth"].astype(np.float32)
            if not np.all(d >= 0):  # allow exactly 0 at boundaries
                errors.append("depth has negative values")
        except Exception as e:
            errors.append(f"depth not loadable: {e}")

    if n_sets > 0:
        first_focus_dir = scene_dir / "focus_maps" / "set_00"
        focus_files = natsorted(first_focus_dir.glob("focus_*.npz"), key=lambda p: p.name) if first_focus_dir.exists() else []
        if focus_files:
            try:
                fm = np.load(focus_files[0])["focus_map"].astype(np.float32)
                if fm.min() < -0.01 or fm.max() > 1.01:
                    errors.append(f"focus_map out of [0,1]: min={fm.min():.3f} max={fm.max():.3f}")
            except Exception as e:
                errors.append(f"focus_map not loadable: {e}")

    # --- Spatial dimension consistency ---
    if sharp_count > 0 and depth_count > 0:
        try:
            first_frame = Image.open(natsorted(sharp_dir.glob("frame_*.png"), key=lambda p: p.name)[0])
            fw, fh = first_frame.size
            first_d = np.load(natsorted(depth_dir.glob("depth_*.npz"), key=lambda p: p.name)[0])["depth"]
            dh, dw = first_d.shape
            if (fw, fh) != (dw, dh):
                errors.append(f"frame {fw}x{fh} != depth {dw}x{dh}")
        except Exception:
            pass

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Validate all outputs")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--scene_index", type=str, required=True)
    parser.add_argument("--report", type=str, required=True, help="Output report JSON path")
    parser.add_argument("--n_sets", type=int, default=4, help="Expected number of CoC sets per scene")
    args = parser.parse_args()

    out_root = Path(args.out_root)

    with open(args.scene_index) as f:
        scene_index = json.load(f)

    total = len(scene_index)
    passing = 0
    failed_scenes: list[str] = []
    failure_reasons: dict[str, list[str]] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    total_frames = 0

    console.print(f"[bold]Validating {total} scenes...[/bold]")

    for scene_id, info in sorted(scene_index.items()):
        scene_dir = out_root / scene_id
        n_frames = info["num_frames"]
        total_frames += n_frames

        ok, reasons = validate_scene(scene_id, scene_dir, n_frames, args.n_sets)

        split = info.get("split", "unknown")
        if split in split_counts:
            split_counts[split] += 1

        if ok:
            passing += 1
        else:
            failed_scenes.append(scene_id)
            failure_reasons[scene_id] = reasons
            console.print(f"  [red]FAIL[/red] {scene_id}: {'; '.join(reasons[:3])}")

    total_gb = get_disk_usage_gb(out_root)

    report = {
        "total_scenes": total,
        "passing_scenes": passing,
        "failed_scenes": failed_scenes,
        "failure_reasons": failure_reasons,
        "split_counts": split_counts,
        "total_frames": total_frames,
        "total_gb_on_disk": round(total_gb, 2),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary table
    table = Table(title="Validation Report")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total scenes", str(total))
    table.add_row("Passing", f"[green]{passing}[/green]")
    table.add_row("Failed", f"[red]{len(failed_scenes)}[/red]")
    table.add_row("Train / Val / Test", f"{split_counts['train']} / {split_counts['val']} / {split_counts['test']}")
    table.add_row("Total frames", str(total_frames))
    table.add_row("Disk usage", f"{total_gb:.2f} GB")
    table.add_row("Report saved", str(report_path))
    console.print(table)

    if failed_scenes:
        console.print(f"\n[yellow]Re-queue failed scenes through their failing stage.[/yellow]")
        console.print(f"[yellow]Check which .done sentinel is absent to determine the failing stage.[/yellow]")
    else:
        console.print("\n[bold green]All scenes validated successfully![/bold green]")


if __name__ == "__main__":
    main()
