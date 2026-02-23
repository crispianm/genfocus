#!/usr/bin/env python3
"""Inspect focus-map distributions for a rendered scene.

This is a lightweight diagnostic to inspect absolute and guidance maps:
- absolute focus_map (CoC on sensor, metres)
- guidance_map ([0,1], if present)
- signed_diopter_map (1/metres, if present)

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
        coc_samples = []
        guidance_samples = []
        diopter_samples = []
        for i in idx:
            data = np.load(files[i])
            coc_samples.append(data["focus_map"])
            if "guidance_map" in data.files:
                guidance_samples.append(data["guidance_map"])
            if "signed_diopter_map" in data.files:
                diopter_samples.append(data["signed_diopter_map"])

        coc_stack = np.stack([s.astype(np.float32) for s in coc_samples], axis=0)
        coc_stats = summarize(coc_stack)
        coc_mm = coc_stack * 1e3
        coc_mm_stats = summarize(coc_mm)

        msg = (
            f"{set_dir.name}: CoC[m] mean={coc_stats['mean']:.6e} p50={coc_stats['p50']:.6e} "
            f"p90={coc_stats['p90']:.6e} | CoC[mm] p50={coc_mm_stats['p50']:.4f} p90={coc_mm_stats['p90']:.4f}"
        )

        if guidance_samples:
            g_stack = np.stack([s.astype(np.float32) for s in guidance_samples], axis=0)
            g_stats = summarize(g_stack)
            frac_defocused = float((g_stack < 0.2).mean())
            frac_sharp = float((g_stack > 0.8).mean())
            msg += (
                f" | guidance mean={g_stats['mean']:.3f} p10={g_stats['p10']:.3f} "
                f"p50={g_stats['p50']:.3f} p90={g_stats['p90']:.3f} "
                f"frac<0.2={frac_defocused:.3f} frac>0.8={frac_sharp:.3f}"
            )

        if diopter_samples:
            d_stack = np.stack([s.astype(np.float32) for s in diopter_samples], axis=0)
            d_stats = summarize(d_stack)
            msg += (
                f" | diopter Δ mean={d_stats['mean']:.4f} p10={d_stats['p10']:.4f} "
                f"p50={d_stats['p50']:.4f} p90={d_stats['p90']:.4f}"
            )

        print(msg)


if __name__ == "__main__":
    main()
