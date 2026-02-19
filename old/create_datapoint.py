import os
import cv2
import time
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

import depth_pro

# --- Model & Config ---

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEBLUR_LORA_PATH = "."
DEBLUR_WEIGHT_NAME = "deblurNet.safetensors"
BOKEH_LORA_DIR = "."
BOKEH_WEIGHT_NAME = "bokehNet.safetensors"

if not os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
    print(f"❌ Warning: {DEBLUR_WEIGHT_NAME} not found.")
if not os.path.exists(os.path.join(BOKEH_LORA_DIR, BOKEH_WEIGHT_NAME)):
    print(f"❌ Warning: {BOKEH_WEIGHT_NAME} not found.")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"🚀 Device detected: {device}")

# Global models
pipe_flux = None
depth_model = None
depth_transform = None
current_adapter = None

def load_models():
    global pipe_flux, depth_model, depth_transform, current_adapter

    print("🔄 Loading FLUX pipeline...")
    pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    current_adapter = None

    if device == "cuda":
        print("🚀 Moving FLUX to CUDA...")
        pipe_flux.to("cuda")

    print("🔄 Loading Depth Pro model...")
    try:
        depth_model, depth_transform = depth_pro.create_model_and_transforms()
        if device == "cuda":
            depth_model.eval().to("cuda")
        else:
            depth_model.eval()
        print("✅ Depth Pro loaded.")
    except Exception as e:
        print(f"❌ Failed to load Depth Pro: {e}")
        depth_model = None
        depth_transform = None

def switch_lora(target_mode):
    global pipe_flux, current_adapter
    if current_adapter == target_mode:
        return
    print(f"🔄 Switching LoRA to [{target_mode}]...")
    pipe_flux.unload_lora_weights()
    if target_mode == "deblur":
        try:
            pipe_flux.load_lora_weights(DEBLUR_LORA_PATH, weight_name=DEBLUR_WEIGHT_NAME, adapter_name="deblurring")
            pipe_flux.set_adapters(["deblurring"])
            current_adapter = "deblur"
        except Exception as e:
            print(f"❌ Failed to load Deblur LoRA: {e}")
    elif target_mode == "bokeh":
        try:
            pipe_flux.load_lora_weights(BOKEH_LORA_DIR, weight_name=BOKEH_WEIGHT_NAME, adapter_name="bokeh")
            pipe_flux.set_adapters(["bokeh"])
            current_adapter = "bokeh"
        except Exception as e:
            print(f"❌ Failed to load Bokeh LoRA: {e}")

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

def process_single_image(image_path, output_root, args):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(output_root, image_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing: {image_path}")
    print(f"Output Directory: {output_dir}")

    try:
        raw_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image {image_path}: {e}")
        return

    # Preprocessing
    clean_input_processed = resize_and_pad_image(raw_img, args.resize_long_side)
    w, h = clean_input_processed.size
    print(f"Image Size: {w}x{h}")

    # --- Stage 1: Deblur & Depth ---
    print("🚀 Stage 1: DeblurNet & Depth Pro")
    switch_lora("deblur")

    condition_0_img = Image.new("RGB", (w, h), (0, 0, 0))
    cond0 = Condition(condition_0_img, "deblurring", [0, 32], 1.0)
    cond1 = Condition(clean_input_processed, "deblurring", [0, 0], 1.0)
    
    # Check if tiling tricks should be disabled
    force_no_tile = min(w, h) < 512
    no_tiled_denoise = bool(args.disable_tiling) or force_no_tile

    seed_everything(42)  # For reproducibility
    
    with torch.no_grad():
       res_deblur = generate(
            pipe_flux,
            height=h,
            width=w,
            prompt="a sharp photo with everything in focus",
            num_inference_steps=args.steps,
            conditions=[cond0, cond1],
            NO_TILED_DENOISE=no_tiled_denoise,
        )
    deblurred_img = res_deblur.images[0]

    # Save Deblur Result (Reference) - optional
    deblurred_img.save(os.path.join(output_dir, "deblurred_ref.png"))

    # Estimate Depth
    try:
        img_t = depth_transform(deblurred_img)
        if device == "cuda":
            img_t = img_t.to("cuda")
        with torch.no_grad():
            pred = depth_model.infer(img_t, f_px=None)
        depth_map = pred["depth"].cpu().numpy().squeeze()
        safe_depth = np.where(depth_map > 0.0, depth_map, np.finfo(np.float32).max)
        disp_orig = 1.0 / safe_depth
        disp = cv2.resize(disp_orig, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Save Depth Map (Reference) - optional
        norm_disp = (disp - disp.min()) / (disp.max() - disp.min())
        Image.fromarray((norm_disp * 255).astype(np.uint8)).save(os.path.join(output_dir, "depth_ref.png"))
        
    except Exception as e:
        print(f"❌ Depth Error: {e}")
        return

    # --- Stage 2: Focal Sweep (Bokeh) ---
    print(f"🚀 Stage 2: BokehNet Sweep ({args.frames} frames)")
    switch_lora("bokeh")

    # Shared latents for consistency
    seed_everything(42)
    gen = torch.Generator(device=pipe_flux.device).manual_seed(1234)
    latents, _ = pipe_flux.prepare_latents(
        batch_size=1,
        num_channels_latents=16,
        height=h,
        width=w,
        dtype=pipe_flux.dtype,
        device=pipe_flux.device,
        generator=gen,
        latents=None,
    )

    disp_vals = disp.flatten()
    min_d = np.percentile(disp_vals, 1)
    max_d = np.percentile(disp_vals, 99)
    focus_values = np.linspace(min_d, max_d, int(args.frames))
    
    K_value = args.k_value

    for i, fd in enumerate(focus_values):
        print(f"   [{i+1}/{args.frames}] Focusing on disp={fd:.4f}")
        
        dmf = disp - np.float32(fd)
        defocus_abs = np.abs(K_value * dmf)
        MAX_COC = 100.0
        defocus_t = torch.from_numpy(defocus_abs).unsqueeze(0).float()
        cond_map = (defocus_t / MAX_COC).clamp(0, 1).repeat(3, 1, 1).unsqueeze(0)
        
        cond_img = Condition(deblurred_img, "bokeh")
        cond_dmf = Condition(cond_map, "bokeh", [0, 0], 1.0, No_preprocess=True)
        
        seed_everything(42)
        gen = torch.Generator(device=pipe_flux.device).manual_seed(1234)
        
        with torch.no_grad():
            res = generate(
                pipe_flux,
                height=h,
                width=w,
                prompt="an excellent photo with a large aperture",
                num_inference_steps=args.steps,
                conditions=[cond_img, cond_dmf],
                guidance_scale=1.0,
                kv_cache=False,
                generator=gen,
                latents=latents,
                NO_TILED_DENOISE=no_tiled_denoise,
            )
        
        out_filename = f"image_{i}.png"
        out_path = os.path.join(output_dir, out_filename)
        res.images[0].save(out_path)

    print(f"✅ Completed: {image_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Focal Sweep Datapoints (Batch Processing)")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing images")
    parser.add_argument("--output_root", type=str, default="./output", help="Root output directory")
    
    # Defaults set to High Quality
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps (Default: 50 for HQ)")
    parser.add_argument("--k_value", type=int, default=20, help="Blur Strength K (Default: 20)")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames for the sweep (Default: 20)")
    parser.add_argument("--resize_long_side", type=int, default=0, help="Resize long side (0 = Original Resolution)")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling optimization (Not recommended for HQ unless analyzing artifacts)")

    args = parser.parse_args()

    # 1. Load Models
    load_models()

    # 2. Get List of Images
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        exit(1)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not image_files:
        print("❌ No valid images found in input directory.")
        exit(1)

    print(f"Found {len(image_files)} images to process.")

    # 3. Process Each Image
    for img_path in image_files:
        process_single_image(img_path, args.output_root, args)
    
    print("\n🎉 All processing complete.")
