import cv2
import os
import argparse
import glob


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _natural_key(s: str):
    import re

    parts = []
    for token in re.split(r"(\d+)", s):
        if not token:
            continue
        if token.isdigit():
            parts.append((1, int(token)))
        else:
            parts.append((0, token.lower()))
    parts.append((2, s))
    return tuple(parts)


def _list_images(folder_path: str) -> list[str]:
    images = []
    for img in os.listdir(folder_path):
        ext = os.path.splitext(img)[1].lower()
        if ext in VALID_IMAGE_EXTS:
            images.append(img)
    images.sort(key=_natural_key)
    return images

def generate_video_from_pngs(folder_path, video_name, fps=30):
    images = _list_images(folder_path)
    if not images:
        print(f"No images found in {folder_path}")
        return

    first_frame_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Failed to load first frame: {first_frame_path}")
        return

    height, width, layers = frame.shape
    print(f"Video Resolution: {width}x{height}, FPS: {fps}")

    # Use mp4v codec for mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    count = 0
    for image in images:
        pth = os.path.join(folder_path, image)
        img_data = cv2.imread(pth)
        if img_data is None:
            print(f"Warning: could not read {pth}, skipping.")
            continue
        video.write(img_data)
        count += 1

    video.release()
    print(f"Successfully saved video to {video_name} ({count} frames)")


def _folders_with_images(input_root: str) -> list[str]:
    """Return subfolders (including input_root) that contain at least one image file."""
    folders = []
    if os.path.isdir(input_root) and _list_images(input_root):
        folders.append(input_root)

    if not os.path.isdir(input_root):
        return folders

    for name in sorted(os.listdir(input_root), key=_natural_key):
        p = os.path.join(input_root, name)
        if os.path.isdir(p) and _list_images(p):
            folders.append(p)

    return folders


def _is_mp4_path(path: str) -> bool:
    return str(path).lower().endswith(".mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image sequence to a video.")
    parser.add_argument("--input_folder", "-i", type=str, help="Path to the folder containing PNGs")
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="output_videos",
        help=(
            "Output path. If input_folder contains images directly, this is the output .mp4 file path. "
            "If input_folder contains subfolders of images, this is treated as an output directory and each "
            "subfolder is saved as <subfolder>.mp4 inside it."
        ),
    )
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")

    args = parser.parse_args()

    # If no input folder is provided, try to find the most recent sweep folder
    target_folder = args.input_folder
    if not target_folder:
        base_output_dir = "./output"
        if os.path.exists(base_output_dir):
            all_subdirs = [os.path.join(base_output_dir, d) for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]
            if all_subdirs:
                # Assuming folder names like sweep_123456... sort by creation time or name
                latest_subdir = max(all_subdirs, key=os.path.getmtime)
                print(f"No input folder specified. Using most recent: {latest_subdir}")
                target_folder = latest_subdir

    if not target_folder or not os.path.exists(target_folder):
        print("Please provide a valid input folder using --input_folder or ensure ./output/ exists.")
        raise SystemExit(1)

    folders = _folders_with_images(target_folder)
    if not folders:
        print(f"No image folders found under: {target_folder}")
        raise SystemExit(1)

    # Batch mode if we found at least one subfolder-with-images, or if output_file is a directory.
    has_subfolders = any(os.path.dirname(f) != os.path.abspath(target_folder) for f in folders if f != target_folder)

    if has_subfolders or os.path.isdir(args.output_file):
        out_dir = args.output_file
        # If user gave a file path like something.mp4, interpret as its parent directory.
        if _is_mp4_path(out_dir):
            out_dir = os.path.dirname(out_dir) or "."
        os.makedirs(out_dir, exist_ok=True)

        # Only process subfolders in batch mode (avoid duplicating the root folder unless it is itself a leaf sequence).
        for folder in folders:
            if folder == target_folder:
                # If target_folder itself is a leaf image sequence (no subfolders), we still want a video.
                continue
            base = os.path.basename(folder.rstrip(os.sep))
            out_path = os.path.join(out_dir, f"{base}.mp4")
            generate_video_from_pngs(folder, out_path, args.fps)

        # If input_root itself has images and there were no subfolder sequences, produce a video for it too.
        if target_folder in folders and not has_subfolders:
            base = os.path.basename(target_folder.rstrip(os.sep)) or "output_video"
            out_path = os.path.join(out_dir, f"{base}.mp4")
            generate_video_from_pngs(target_folder, out_path, args.fps)
    else:
        # Single sequence -> single output file
        generate_video_from_pngs(target_folder, args.output_file, args.fps)
