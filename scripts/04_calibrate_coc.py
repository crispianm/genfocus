#!/usr/bin/env python3
"""Phase 4: CoC Distribution Calibration.

Scans all saved depth_maps/ to determine the global max_coc normalisation
value. No GPU required.

Usage:
    python scripts/04_calibrate_coc.py \
        --out_root        ./output \
        --sample_fraction 0.2 \
        --output_plot     ./output/logs/coc_histogram.png \
        --output_stats    ./output/logs/coc_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from natsort import natsorted
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# CoC formula (matches dataloader exactly)
# ---------------------------------------------------------------------------

def compute_raw_coc(
    depth: np.ndarray,
    f: float,
    N: float,
    S_focus: float,
) -> np.ndarray:
    """Compute raw Circle of Confusion (in metres, unnormalised).

    depth   : (H, W) float32, metres
    f       : focal length in metres (e.g. 0.050 for 50 mm)
    N       : f-number (e.g. 2.8)
    S_focus : focus distance in metres
    """
    depth_safe = np.maximum(depth.astype(np.float32), 1e-6)
    return (f ** 2 / (N * S_focus)) * np.abs(depth_safe - S_focus) / depth_safe


# ---------------------------------------------------------------------------
# Representative lens parameter sets
# ---------------------------------------------------------------------------

def sample_lens_params(
    d_min: float,
    d_max: float,
    rng: np.random.Generator,
    n_sets: int = 8,
) -> list[dict]:
    """Sample representative (f, N, S_focus) tuples for CoC calibration."""
    from scipy.stats import beta as beta_dist

    params = []
    for _ in range(n_sets):
        f = rng.uniform(24e-3, 85e-3)
        log_N = rng.uniform(np.log(2.8), np.log(8.0))
        N = float(np.exp(log_N))
        u = beta_dist.rvs(2, 2, random_state=int(rng.integers(1e9)))
        S_focus = d_min + u * (d_max - d_min)
        params.append({"f": float(f), "N": N, "S_focus": float(S_focus)})
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Calibrate CoC distribution")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--sample_fraction", type=float, default=0.2,
                        help="Fraction of scenes to sample (default: 0.2)")
    parser.add_argument("--output_stats", type=str, required=True,
                        help="Output JSON path for CoC stats")
    parser.add_argument("--output_plot", type=str, default="",
                        help="Optional histogram plot path (.png)")
    parser.add_argument("--n_lens_sets", type=int, default=8,
                        help="Number of lens param sets per scene")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    scene_index_path = out_root / "scene_index.json"

    if not scene_index_path.exists():
        console.print("[red]scene_index.json not found. Run Phase 1 first.[/red]")
        sys.exit(1)

    with open(scene_index_path) as f:
        scene_index = json.load(f)

    rng = np.random.default_rng(args.seed)

    # Select a random sample of scenes
    scene_ids = natsorted(scene_index.keys())
    n_sample = max(1, int(len(scene_ids) * args.sample_fraction))
    sampled = list(rng.choice(scene_ids, size=n_sample, replace=False))

    console.print(f"[bold]Sampling {n_sample}/{len(scene_ids)} scenes for CoC calibration[/bold]")

    all_coc_values: list[np.ndarray] = []

    for sid in sampled:
        stats_path = out_root / sid / "depth_maps" / "stats.json"
        if not stats_path.exists():
            console.print(f"  [yellow]{sid}: stats.json missing — skipping[/yellow]")
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        d_min = max(stats.get("d_p5", stats["d_min"]), 0.1)  # use 5th-pct, not raw min
        d_max = stats.get("d_p95", stats["d_max"])            # use 95th-pct, not raw max
        if d_max <= d_min:
            continue

        # Load a few depth maps from this scene for pixel-level CoC computation
        depth_dir = out_root / sid / "depth_maps"
        depth_files = natsorted(depth_dir.glob("depth_*.npz"), key=lambda p: p.name)
        if not depth_files:
            continue

        # Sample up to 5 frames per scene for efficiency
        n_frames = min(5, len(depth_files))
        frame_indices = rng.choice(len(depth_files), size=n_frames, replace=False)

        lens_params = sample_lens_params(d_min, d_max, rng, n_sets=args.n_lens_sets)

        for fi in frame_indices:
            depth = np.load(depth_files[fi])["depth"].astype(np.float32)
            for lp in lens_params:
                coc = compute_raw_coc(depth, **lp)
                # Subsample pixels for memory efficiency
                flat = coc.ravel()
                if len(flat) > 50000:
                    flat = rng.choice(flat, size=50000, replace=False)
                all_coc_values.append(flat)

    if not all_coc_values:
        console.print("[red]No CoC data collected. Check depth_maps/stats.json files.[/red]")
        sys.exit(1)

    all_coc = np.concatenate(all_coc_values)
    console.print(f"  Collected {len(all_coc):,} CoC samples")

    # Compute percentiles
    percentiles = {
        "p50": float(np.percentile(all_coc, 50)),
        "p90": float(np.percentile(all_coc, 90)),
        "p95": float(np.percentile(all_coc, 95)),
        "p99": float(np.percentile(all_coc, 99)),
        "max": float(np.max(all_coc)),
        "recommended_max_coc": float(np.percentile(all_coc, 99)),
    }

    # Write stats
    output_stats = Path(args.output_stats)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(output_stats, "w") as f:
        json.dump(percentiles, f, indent=2)

    console.print("\n[bold]CoC Calibration Results:[/bold]")
    for k, v in percentiles.items():
        console.print(f"  {k}: {v:.6f}")

    # Optional histogram plot
    if args.output_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(all_coc, bins=200, density=True, alpha=0.7, color="steelblue")
            for label, pval in [("p50", percentiles["p50"]),
                                ("p95", percentiles["p95"]),
                                ("p99", percentiles["p99"])]:
                ax.axvline(pval, color="red", linestyle="--", alpha=0.8)
                ax.text(pval, ax.get_ylim()[1] * 0.9, f" {label}={pval:.4f}",
                        fontsize=8, color="red")
            ax.set_xlabel("Raw CoC (metres)")
            ax.set_ylabel("Density")
            ax.set_title("CoC Distribution (sampled scenes × lens params)")
            fig.tight_layout()
            fig.savefig(args.output_plot, dpi=150)
            plt.close(fig)
            console.print(f"  Plot saved: {args.output_plot}")
        except ImportError:
            console.print("  [yellow]matplotlib not available — skipping plot.[/yellow]")

    console.print(f"\n[bold green]Recommended max_coc = {percentiles['recommended_max_coc']:.6f}[/bold green]")
    console.print("[bold yellow]MANUAL STEP: Pass this value as --max_coc to Phase 5.[/bold yellow]")


if __name__ == "__main__":
    main()
