import argparse
import os
import re
import random
from dataclasses import dataclass
import time

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline

from Genfocus.pipeline.flux import Condition, generate, seed_everything

import depth_pro


MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEBLUR_LORA_PATH = "."
DEBLUR_WEIGHT_NAME = "deblurNet.safetensors"
BOKEH_LORA_DIR = "."
BOKEH_WEIGHT_NAME = "bokehNet.safetensors"


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_key(s: str):
    # Robust natural sort key that avoids comparing ints to strs directly.
    # Produces a tuple of tagged values: (0, "text") for text and (1, 123) for numbers.
    parts: list[tuple[int, object]] = []
    for token in re.split(r"(\d+)", s):
        if not token:
            continue
        if token.isdigit():
            parts.append((1, int(token)))
        else:
            parts.append((0, token.lower()))
    # Use the original string as a deterministic tiebreaker.
    parts.append((2, s))
    return tuple(parts)


def _list_image_files(folder: str) -> list[str]:
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXTS:
            files.append(os.path.join(folder, name))
    files.sort(key=lambda p: _natural_key(os.path.basename(p)))
    return files


def find_sequence_dirs(input_root: str, min_frames: int = 2) -> list[str]:
    seq_dirs: list[str] = []

    for dirpath, dirnames, filenames in os.walk(input_root, topdown=True):
        # Skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        image_count = 0
        for f in filenames:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                image_count += 1

        # Treat any directory containing >=min_frames images as a "sequence".
        if image_count >= min_frames:
            seq_dirs.append(dirpath)
            # Don't descend further; we assume frames live in leaf dirs.
            dirnames[:] = []

    seq_dirs.sort(key=_natural_key)
    return seq_dirs


@dataclass
class Models:
    pipe_flux: FluxPipeline
    depth_model: torch.nn.Module
    depth_transform: object
    device: str
    dtype: torch.dtype


def load_models() -> Models:
    if not os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
        print(f"❌ Warning: {DEBLUR_WEIGHT_NAME} not found.")
    if not os.path.exists(os.path.join(BOKEH_LORA_DIR, BOKEH_WEIGHT_NAME)):
        print(f"❌ Warning: {BOKEH_WEIGHT_NAME} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        # bf16 can be slower on some GPUs; prefer fp16 unless bf16 is supported.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        dtype = torch.float32

    print(f"🚀 Device detected: {device}")
    print("🔄 Loading FLUX pipeline...")
    pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device == "cuda":
        print("🚀 Moving FLUX to CUDA...")
        pipe_flux.to("cuda")

    # Preload LoRAs once to avoid expensive per-frame load/unload.
    # (We can still switch adapters per-stage via set_adapters.)
    loaded = []
    if os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
        pipe_flux.load_lora_weights(DEBLUR_LORA_PATH, weight_name=DEBLUR_WEIGHT_NAME, adapter_name="deblurring")
        loaded.append("deblurring")
    if os.path.exists(os.path.join(BOKEH_LORA_DIR, BOKEH_WEIGHT_NAME)):
        pipe_flux.load_lora_weights(BOKEH_LORA_DIR, weight_name=BOKEH_WEIGHT_NAME, adapter_name="bokeh")
        loaded.append("bokeh")
    if loaded:
        print(f"✅ Preloaded LoRAs: {loaded}")

    print("🔄 Loading Depth Pro model...")
    depth_model, depth_transform = depth_pro.create_model_and_transforms()
    depth_model.eval()
    if device == "cuda":
        depth_model.to("cuda")

    return Models(
        pipe_flux=pipe_flux,
        depth_model=depth_model,
        depth_transform=depth_transform,
        device=device,
        dtype=dtype,
    )


def switch_lora(pipe_flux: FluxPipeline, target_mode: str):
    if target_mode == "deblur":
        pipe_flux.set_adapters(["deblurring"])
    elif target_mode == "bokeh":
        pipe_flux.set_adapters(["bokeh"])
    else:
        raise ValueError(f"Unknown LoRA mode: {target_mode}")


def _configure_determinism(deterministic: bool):
    # Deterministic CuDNN is often *much* slower; keep it off unless requested.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def resize_and_pad_image(img: Image.Image, target_long_side: int) -> Image.Image:
    w, h = img.size

    if target_long_side and target_long_side > 0:
        target_max = int(target_long_side)

        if w >= h:
            new_w = target_max
            scale = target_max / w
            new_h = int(h * scale)
        else:
            new_h = target_max
            scale = target_max / h
            new_w = int(w * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        final_w = (new_w // 16) * 16
        final_h = (new_h // 16) * 16

        final_w = max(final_w, 16)
        final_h = max(final_h, 16)

        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        right = left + final_w
        bottom = top + final_h

        return img.crop((left, top, right, bottom))

    final_w = ((w + 15) // 16) * 16
    final_h = ((h + 15) // 16) * 16

    if final_w == w and final_h == h:
        return img

    return img.resize((final_w, final_h), Image.LANCZOS)


def compute_disparity(models: Models, rgb_img: Image.Image) -> np.ndarray:
    w, h = rgb_img.size
    img_t = models.depth_transform(rgb_img)
    if models.device == "cuda":
        img_t = img_t.to("cuda")

    with torch.no_grad():
        pred = models.depth_model.infer(img_t, f_px=None)

    depth_map = pred["depth"].detach().cpu().numpy().squeeze()
    safe_depth = np.where(depth_map > 0.0, depth_map, np.finfo(np.float32).max)
    disp_orig = 1.0 / safe_depth
    disp = cv2.resize(disp_orig, (w, h), interpolation=cv2.INTER_LINEAR)
    return disp.astype(np.float32)


def schedule_values(num_frames: int, start: float, end: float, schedule: str) -> np.ndarray:
    if num_frames <= 1:
        return np.array([start], dtype=np.float32)

    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    if schedule == "linear":
        w = t
    elif schedule == "cosine":
        # Ease-in-out
        w = 0.5 - 0.5 * np.cos(np.pi * t)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return (start + (end - start) * w).astype(np.float32)


def choose_focus_pull_percentiles(rng: random.Random) -> tuple[float, float]:
    # Keep away from extremes; ensure a visible pull.
    low_p = 5.0
    high_p = 95.0
    start = rng.uniform(low_p, high_p)
    min_delta = 20.0
    end = start
    for _ in range(10):
        cand = rng.uniform(low_p, high_p)
        if abs(cand - start) >= min_delta:
            end = cand
            break
    if end == start:
        end = high_p if start < (low_p + high_p) / 2.0 else low_p
    return float(start), float(end)


def choose_focus_pull(disp: np.ndarray, rng: random.Random) -> tuple[float, float, float, float]:
    disp_vals = disp.reshape(-1)
    p1 = float(np.percentile(disp_vals, 1))
    p5 = float(np.percentile(disp_vals, 5))
    p95 = float(np.percentile(disp_vals, 95))
    p99 = float(np.percentile(disp_vals, 99))

    low = p5
    high = p95
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = p1, p99

    # Pick a start in-range, then pick an end far enough away for a visible pull.
    start = rng.uniform(low, high)
    min_delta = 0.25 * (high - low + 1e-6)

    # Try a few times to get separation.
    end = start
    for _ in range(10):
        cand = rng.uniform(low, high)
        if abs(cand - start) >= min_delta:
            end = cand
            break
    if end == start:
        end = high if start < (low + high) / 2.0 else low

    return start, end, low, high


def run_stage1_deblur(
    models: Models,
    clean_input_processed: Image.Image,
    steps: int,
    disable_tiling: bool,
    seed: int,
    deterministic: bool,
    kv_cache: bool,
) -> Image.Image:
    w, h = clean_input_processed.size

    force_no_tile = min(w, h) < 512
    no_tiled_denoise = bool(disable_tiling) or force_no_tile

    switch_lora(models.pipe_flux, "deblur")
    condition_0_img = Image.new("RGB", (w, h), (0, 0, 0))
    cond0 = Condition(condition_0_img, "deblurring", [0, 32], 1.0)
    cond1 = Condition(clean_input_processed, "deblurring", [0, 0], 1.0)

    _configure_determinism(deterministic)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    res_deblur = generate(
        models.pipe_flux,
        height=h,
        width=w,
        prompt="a sharp photo with everything in focus",
        num_inference_steps=int(steps),
        conditions=[cond0, cond1],
        NO_TILED_DENOISE=no_tiled_denoise,
        kv_cache=bool(kv_cache),
    )

    return res_deblur.images[0]


def run_stage2_bokeh(
    models: Models,
    deblurred_img: Image.Image,
    disp: np.ndarray,
    focus_disp: float,
    steps: int,
    k_value: float,
    latents: torch.Tensor,
    disable_tiling: bool,
    seed: int,
    deterministic: bool,
    kv_cache: bool,
) -> Image.Image:
    w, h = deblurred_img.size

    force_no_tile = min(w, h) < 512
    no_tiled_denoise = bool(disable_tiling) or force_no_tile

    switch_lora(models.pipe_flux, "bokeh")

    dmf = disp - np.float32(focus_disp)
    defocus_abs = np.abs(float(k_value) * dmf)

    MAX_COC = 100.0
    defocus_t = torch.from_numpy(defocus_abs).unsqueeze(0).float()
    cond_map = (defocus_t / MAX_COC).clamp(0, 1).repeat(3, 1, 1).unsqueeze(0)

    cond_img = Condition(deblurred_img, "bokeh")
    cond_dmf = Condition(cond_map, "bokeh", [0, 0], 1.0, No_preprocess=True)

    _configure_determinism(deterministic)
    gen = torch.Generator(device=models.pipe_flux.device).manual_seed(int(seed))

    res = generate(
        models.pipe_flux,
        height=h,
        width=w,
        prompt="an excellent photo with a large aperture",
        num_inference_steps=int(steps),
        conditions=[cond_img, cond_dmf],
        guidance_scale=1.0,
        kv_cache=bool(kv_cache),
        generator=gen,
        latents=latents,
        NO_TILED_DENOISE=no_tiled_denoise,
    )

    return res.images[0]


def process_sequence(models: Models, seq_dir: str, input_root: str, output_root: str, args: argparse.Namespace):
    frame_paths = _list_image_files(seq_dir)
    if len(frame_paths) < 2:
        return

    # Print sample ordering to help catch sort issues.
    sample_n = min(5, len(frame_paths))
    head = [os.path.basename(p) for p in frame_paths[:sample_n]]
    tail = [os.path.basename(p) for p in frame_paths[-sample_n:]]
    print(f"  Frame order (first {sample_n}): {head}")
    print(f"  Frame order (last  {sample_n}): {tail}")

    rel = os.path.relpath(seq_dir, input_root)
    out_dir = os.path.join(output_root, rel)
    os.makedirs(out_dir, exist_ok=True)

    clean_dir = os.path.join(out_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)

    print(f"\n▶ Sequence: {seq_dir}")
    print(f"  Frames: {len(frame_paths)}")
    print(f"  Output: {out_dir}")

    def _out_paths(frame_idx: int) -> tuple[str, str]:
        name = f"{frame_idx:05d}.png"
        return os.path.join(out_dir, name), os.path.join(clean_dir, name)

    if not bool(args.process_existing):
        remaining = 0
        for i in range(len(frame_paths)):
            final_path, _clean_path = _out_paths(i)
            if not os.path.exists(final_path):
                remaining += 1
        if remaining == 0:
            print("  ✅ All outputs already exist; skipping sequence.")
            return
        print(f"  ↪ Skipping existing outputs; remaining frames: {remaining}/{len(frame_paths)}")

    # Seed handling:
    # - seed=0 means random per run
    # - otherwise deterministic
    if int(args.seed) == 0:
        seq_seed = random.SystemRandom().randint(1, 2**31 - 1)
    else:
        seq_seed = int(args.seed) + (abs(hash(rel)) % 100000)

    rng = random.Random(seq_seed)

    steps_stage1 = int(args.steps_stage1) if args.steps_stage1 is not None else int(args.steps)
    steps_stage2 = int(args.steps_stage2) if args.steps_stage2 is not None else int(args.steps)

    # Pick a reference frame for resolution + focus-range calibration.
    # Prefer an existing clean frame (to save compute), otherwise use the first
    # frame that still needs processing.
    ref_idx = 0
    if not bool(args.process_existing):
        for i in range(len(frame_paths)):
            final_path, clean_path = _out_paths(i)
            if os.path.exists(clean_path):
                ref_idx = i
                break
            if not os.path.exists(final_path):
                ref_idx = i
                break

    ref_raw = Image.open(frame_paths[ref_idx]).convert("RGB")
    ref_proc = resize_and_pad_image(ref_raw, int(args.resize_long_side))
    w, h = ref_proc.size

    # Create shared latents for temporal consistency (per resolution)
    gen = torch.Generator(device=models.pipe_flux.device).manual_seed(1234)
    latents, _ = models.pipe_flux.prepare_latents(
        batch_size=1,
        num_channels_latents=16,
        height=h,
        width=w,
        dtype=models.pipe_flux.dtype,
        device=models.pipe_flux.device,
        generator=gen,
        latents=None,
    )

    # Stage1+depth on reference frame to choose focus-pull range/seed.
    _final_ref_path, clean_ref_path = _out_paths(ref_idx)
    first_deblur: Image.Image
    if bool(args.skip_deblur):
        first_deblur = ref_proc
    elif os.path.exists(clean_ref_path):
        first_deblur = Image.open(clean_ref_path).convert("RGB")
        if first_deblur.size != (w, h):
            # Likely changed resize settings; regenerate.
            first_deblur = run_stage1_deblur(
                models,
                ref_proc,
                steps=steps_stage1,
                disable_tiling=args.disable_tiling,
                seed=int(args.seed) if int(args.seed) != 0 else 42,
                deterministic=bool(args.deterministic),
                kv_cache=bool(args.kv_cache),
            )
            if bool(args.save_clean):
                first_deblur.save(clean_ref_path)
    else:
        first_deblur = run_stage1_deblur(
            models,
            ref_proc,
            steps=steps_stage1,
            disable_tiling=args.disable_tiling,
            seed=int(args.seed) if int(args.seed) != 0 else 42,
            deterministic=bool(args.deterministic),
            kv_cache=bool(args.kv_cache),
        )
        if bool(args.save_clean):
            first_deblur.save(clean_ref_path)

    first_disp = compute_disparity(models, first_deblur)

    focus_mode = str(args.focus_mode)
    if focus_mode == "disp":
        if bool(args.random_focus_pull):
            start_disp, end_disp, low, high = choose_focus_pull(first_disp, rng)
        else:
            disp_vals = first_disp.reshape(-1)
            low = float(np.percentile(disp_vals, 5))
            high = float(np.percentile(disp_vals, 95))
            start_disp, end_disp = low, high

        focus_curve = schedule_values(len(frame_paths), start_disp, end_disp, args.schedule)
        print(
            f"  Focus pull (disp): start={start_disp:.4f} end={end_disp:.4f} (range≈[{low:.4f}, {high:.4f}]) schedule={args.schedule} seed={seq_seed}"
        )
    elif focus_mode == "percentile":
        if bool(args.random_focus_pull):
            start_p, end_p = choose_focus_pull_percentiles(rng)
        else:
            start_p, end_p = 5.0, 95.0

        p_curve = schedule_values(len(frame_paths), float(start_p), float(end_p), args.schedule)
        focus_curve = p_curve
        print(
            f"  Focus pull (percentile): start_p={start_p:.1f} end_p={end_p:.1f} schedule={args.schedule} seed={seq_seed}"
        )
    else:
        raise ValueError(f"Unknown focus_mode: {focus_mode}")

    # Process all frames
    for idx, frame_path in enumerate(frame_paths):
        t0 = time.perf_counter() if bool(args.profile_timing) else 0.0
        final_path, clean_path = _out_paths(idx)
        if (not bool(args.process_existing)) and os.path.exists(final_path):
            continue

        raw = Image.open(frame_path).convert("RGB")
        proc = resize_and_pad_image(raw, int(args.resize_long_side))

        if proc.size != (w, h):
            # Resolution changed mid-sequence; regenerate latents to match.
            w, h = proc.size
            seed_everything(42)
            gen = torch.Generator(device=models.pipe_flux.device).manual_seed(1234)
            latents, _ = models.pipe_flux.prepare_latents(
                batch_size=1,
                num_channels_latents=16,
                height=h,
                width=w,
                dtype=models.pipe_flux.dtype,
                device=models.pipe_flux.device,
                generator=gen,
                latents=None,
            )

        deblur: Image.Image
        if bool(args.skip_deblur):
            deblur = proc
        elif bool(args.save_clean) and os.path.exists(clean_path):
            deblur = Image.open(clean_path).convert("RGB")
            if deblur.size != (w, h):
                # Clean frame exists but doesn't match current settings/resolution.
                deblur = run_stage1_deblur(
                    models,
                    proc,
                    steps=steps_stage1,
                    disable_tiling=args.disable_tiling,
                    seed=seq_seed,
                    deterministic=bool(args.deterministic),
                    kv_cache=bool(args.kv_cache),
                )
                if bool(args.save_clean):
                    deblur.save(clean_path)
        else:
            deblur = run_stage1_deblur(
                models,
                proc,
                steps=steps_stage1,
                disable_tiling=args.disable_tiling,
                seed=seq_seed,
                deterministic=bool(args.deterministic),
                kv_cache=bool(args.kv_cache),
            )
            if bool(args.save_clean):
                deblur.save(clean_path)

        disp = compute_disparity(models, deblur)

        curve_val = float(focus_curve[min(idx, len(focus_curve) - 1)])
        if focus_mode == "percentile":
            # Convert smooth percentile curve into a per-frame focus disparity.
            # This helps when the depth estimator's absolute scale drifts across time.
            disp_vals = disp.reshape(-1)
            focus_disp = float(np.percentile(disp_vals, curve_val))
        else:
            focus_disp = curve_val

        # Default to a fixed seed across frames to reduce flicker.
        frame_seed = (seq_seed + idx) if bool(args.vary_seed_per_frame) else seq_seed

        out = run_stage2_bokeh(
            models=models,
            deblurred_img=deblur,
            disp=disp,
            focus_disp=focus_disp,
            steps=steps_stage2,
            k_value=args.k_value,
            latents=latents,
            disable_tiling=args.disable_tiling,
            seed=frame_seed,
            deterministic=bool(args.deterministic),
            kv_cache=bool(args.kv_cache),
        )

        out.save(final_path)

        if bool(args.profile_timing):
            dt = time.perf_counter() - t0
            print(f"  Frame {idx:05d}: {dt:.2f}s")

        if (idx + 1) % 10 == 0 or (idx + 1) == len(frame_paths):
            print(f"  Saved {idx + 1}/{len(frame_paths)}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process a dataset of video frame folders with Genfocus: refocus (deblur) then time-varying defocus blur (focus pull)."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Root folder containing sequence folders")
    parser.add_argument("--output_root", type=str, required=True, help="Root output folder for PNG sequences")

    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps per stage")
    parser.add_argument("--steps_stage1", type=int, default=None, help="Stage 1 (deblur) steps; default=--steps")
    parser.add_argument("--steps_stage2", type=int, default=None, help="Stage 2 (bokeh) steps; default=--steps")
    parser.add_argument("--k_value", type=float, default=20, help="Defocus strength")
    parser.add_argument("--resize_long_side", type=int, default=0, help="Resize long side (0 = original, aligned to 16)")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling optimization")
    parser.add_argument("--kv_cache", action="store_true", help="Enable KV cache inside FLUX wrapper (often faster, uses more VRAM)")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic CuDNN (slower; for strict reproducibility)")
    parser.add_argument("--skip_deblur", action="store_true", help="Skip stage-1 deblur (much faster; quality trade-off)")
    parser.add_argument(
        "--no_save_clean",
        dest="save_clean",
        action="store_false",
        help="Do not save/load stage-1 clean frames under clean/ (slightly faster, but can't resume stage-2)",
    )
    parser.set_defaults(save_clean=True)
    parser.add_argument("--profile_timing", action="store_true", help="Print per-frame wall time")

    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine", help="Focus pull curve over time")
    parser.add_argument(
        "--focus_mode",
        choices=["percentile", "disp"],
        default="percentile",
        help="How to parameterize focus pull across time. 'percentile' is smoother under depth-scale drift.",
    )
    parser.add_argument("--random_focus_pull", action="store_true", help="Randomize start/end focus depths per sequence")
    parser.add_argument(
        "--vary_seed_per_frame",
        action="store_true",
        help="Vary RNG seed per frame (can increase flicker; default keeps it fixed for stability)",
    )
    parser.add_argument("--seed", type=int, default=0, help="0=random, otherwise deterministic")

    parser.add_argument(
        "--process_existing",
        action="store_true",
        help="Recompute frames even if output PNGs already exist (default: skip existing)",
    )

    parser.add_argument("--min_frames", type=int, default=2, help="Minimum number of frames to consider a folder a sequence")

    args = parser.parse_args()

    if not os.path.isdir(args.input_root):
        raise SystemExit(f"Input root not found: {args.input_root}")

    os.makedirs(args.output_root, exist_ok=True)

    models = load_models()

    seq_dirs = find_sequence_dirs(args.input_root, min_frames=int(args.min_frames))
    if not seq_dirs:
        print("❌ No sequence folders found (need folders containing >=2 images).")
        return

    print(f"✅ Found {len(seq_dirs)} sequences under {args.input_root}")

    for seq_dir in seq_dirs:
        process_sequence(models, seq_dir, args.input_root, args.output_root, args)

    print("\n🎉 All sequences processed.")


if __name__ == "__main__":
    main()
