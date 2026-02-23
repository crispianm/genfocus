#!/usr/bin/env python3
"""Phase 4: CoC Distribution Calibration.

Scans all saved depth_maps/ to determine the global max_coc normalisation
value. No GPU required.

Usage:
    python scripts/04_calibrate_coc.py \
        --out_root        ./output \
        --scene_index     ./output/scene_index.json \
        --sample_fraction 0.2 \
        --output_plot     ./output/logs/coc_histogram.png \
        --output_stats    ./output/logs/coc_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
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
    # Thin-lens defocus blur diameter on the image plane (sensor), in metres.
    # Uses (S_focus - f) rather than S_focus to avoid noticeable error at
    # close focus distances.
    depth_safe = np.maximum(depth.astype(np.float32), 1e-6)
    S_focus_safe = max(float(S_focus), float(f) + 1e-6)
    denom = float(N) * (S_focus_safe - float(f))
    return (float(f) ** 2 / denom) * np.abs(depth_safe - S_focus_safe) / depth_safe


def apply_coc_acceptance(coc_raw: np.ndarray, coc_accept: float) -> np.ndarray:
    """Apply an "acceptable" CoC deadzone.

    In real optics, very small blur circles are visually indistinguishable from
    perfect focus. Subtracting a small acceptance threshold makes the resulting
    focus maps less unrealistically shallow.
    """
    return np.maximum(coc_raw - float(coc_accept), 0.0)


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
        log_N = rng.uniform(np.log(1.4), np.log(8.0))
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
    parser.add_argument("--scene_index", type=str, default="",
                        help="Path to scene_index.json (optional; auto-detected if omitted)")
    parser.add_argument("--sample_fraction", type=float, default=0.2,
                        help="Fraction of scenes to sample (default: 0.2)")
    parser.add_argument("--output_stats", type=str, required=True,
                        help="Output JSON path for CoC stats")
    parser.add_argument("--output_plot", type=str, default="",
                        help="Optional histogram plot path (.png)")
    parser.add_argument("--n_lens_sets", type=int, default=8,
                        help="Number of lens param sets per scene")
    parser.add_argument("--coc_accept", type=float, default=3e-5,
                        help="Acceptable CoC deadzone in metres (default: 3e-5 ~= 0.03mm)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    candidate_paths: list[Path] = []

    if args.scene_index:
        candidate_paths.append(Path(args.scene_index))

    env_scene_index = os.environ.get("SCENE_INDEX", "").strip()
    if env_scene_index:
        candidate_paths.append(Path(env_scene_index))

    candidate_paths.append(out_root / "scene_index.json")

    # Support run-based output layouts, e.g. out_root=output/runs/run_... while
    # scene_index.json remains in output/.
    for parent in out_root.parents:
        candidate_paths.append(parent / "scene_index.json")

    seen: set[Path] = set()
    deduped_candidates: list[Path] = []
    for p in candidate_paths:
        if p in seen:
            continue
        seen.add(p)
        deduped_candidates.append(p)

    scene_index_path = next((p for p in deduped_candidates if p.exists()), None)

    if scene_index_path is None:
        searched = "\n".join(f"  - {p}" for p in deduped_candidates)
        console.print("[red]scene_index.json not found. Run Phase 1 first or pass --scene_index.[/red]")
        console.print(f"[yellow]Searched paths:[/yellow]\n{searched}")
        sys.exit(1)

    console.print(f"Using scene index: {scene_index_path}")

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

        d_min = max(stats["d_min"], 0.1)  # clamp to avoid degenerate CoC
        d_max = stats["d_max"]
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
                coc_raw = compute_raw_coc(depth, **lp)
                coc = apply_coc_acceptance(coc_raw, args.coc_accept)
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
        "recommended_max_coc": float(np.percentile(all_coc, 95)),
        "coc_accept": float(args.coc_accept),
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
            ax.set_xlabel("Effective CoC (metres) (raw - coc_accept, clipped)")
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
