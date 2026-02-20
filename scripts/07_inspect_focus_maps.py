#!/usr/bin/env python3
"""Inspect focus-map distributions for a rendered scene.

This is a lightweight diagnostic to quickly answer:
- Are focus maps mostly near 0 (everything defocused)?
- Are they mostly near 1 (everything sharp)?
- Do different CoC sets vary as expected?

Usage:
    python scripts/07_inspect_focus_maps.py --scene_dir output/bear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from natsort import natsorted


def summarize(arr: np.ndarray) -> dict[str, float]:
    arr = arr.astype(np.float32)
    return {
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved focus_maps statistics")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path like output/<scene_id>")
    parser.add_argument("--sample_frames", type=int, default=8, help="Frames sampled per set")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    focus_root = scene_dir / "focus_maps"
    if not focus_root.exists():
        raise SystemExit(f"focus_maps not found: {focus_root}")

    metadata_path = scene_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.load(open(metadata_path))
        extra = {k: meta.get(k) for k in ("coc_accept", "coc_gamma") if k in meta}
        if extra:
            print("metadata:", extra)

    set_dirs = [p for p in natsorted(focus_root.glob("set_*")) if p.is_dir()]
    if not set_dirs:
        raise SystemExit(f"No set_* directories in: {focus_root}")

    for set_dir in set_dirs:
        files = natsorted(set_dir.glob("focus_*.npz"))
        if not files:
            print(f"{set_dir.name}: no focus_*.npz")
            continue

        n = min(args.sample_frames, len(files))
        idx = np.linspace(0, len(files) - 1, n, dtype=int)
        samples = []
        for i in idx:
            fm = np.load(files[i])["focus_map"]
            samples.append(fm)

        stack = np.stack([s.astype(np.float32) for s in samples], axis=0)
        stats = summarize(stack)
        frac_defocused = float((stack < 0.2).mean())
        frac_sharp = float((stack > 0.8).mean())

        print(
            f"{set_dir.name}: mean={stats['mean']:.3f} p10={stats['p10']:.3f} p50={stats['p50']:.3f} "
            f"p90={stats['p90']:.3f} frac<0.2={frac_defocused:.3f} frac>0.8={frac_sharp:.3f}"
        )


if __name__ == "__main__":
    main()
