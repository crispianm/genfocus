import argparse
import os
from dataclasses import dataclass

import cv2


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
    images: list[str] = []
    try:
        names = os.listdir(folder_path)
    except FileNotFoundError:
        return []

    for name in names:
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_IMAGE_EXTS:
            images.append(name)
    images.sort(key=_natural_key)
    return images


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


def _center_square_crop(img, size: int):
    if size <= 0:
        raise ValueError("zoom size must be > 0")
    h, w = img.shape[:2]
    side = min(h, w)
    if size > side:
        raise ValueError(
            f"zoom size {size} exceeds min image side {side} for frame {w}x{h}"
        )

    x1 = (w - size) // 2
    y1 = (h - size) // 2
    return img[y1 : y1 + size, x1 : x1 + size]


@dataclass(frozen=True)
class SequencePair:
    name: str
    original_dir: str
    processed_dir: str


def _find_sequence_pairs(original_root: str, processed_root: str) -> list[SequencePair]:
    pairs: list[SequencePair] = []
    if not os.path.isdir(processed_root):
        return pairs

    for name in sorted(os.listdir(processed_root), key=_natural_key):
        processed_dir = os.path.join(processed_root, name)
        if not os.path.isdir(processed_dir):
            continue
        if not _list_images(processed_dir):
            continue

        original_dir = os.path.join(original_root, name)
        if not os.path.isdir(original_dir):
            print(f"Skipping {name}: missing original folder {original_dir}")
            continue
        if not _list_images(original_dir):
            print(f"Skipping {name}: no images in original folder {original_dir}")
            continue

        pairs.append(SequencePair(name=name, original_dir=original_dir, processed_dir=processed_dir))

    return pairs


def _build_frame_map(folder: str) -> dict[str, str]:
    """Map frame stem -> filename (first occurrence)."""
    m: dict[str, str] = {}
    for fname in _list_images(folder):
        st = _stem(fname)
        if st not in m:
            m[st] = fname
    return m


def _validate_frames_complete(original_dir: str, processed_dir: str) -> tuple[bool, list[str]]:
    """Return (ok, ordered_stems).

    ok is True only if every original frame stem exists in processed_dir.
    Ordered stems follow the original sequence order.
    """
    orig_files = _list_images(original_dir)
    if not orig_files:
        return False, []

    orig_stems = [_stem(f) for f in orig_files]
    proc_map = _build_frame_map(processed_dir)

    missing = [st for st in orig_stems if st not in proc_map]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        print(
            f"Skipping {os.path.basename(processed_dir)}: processed missing {len(missing)} frames: {preview}{suffix}"
        )
        return False, []

    extra = [st for st in proc_map.keys() if st not in set(orig_stems)]
    if extra:
        # Not fatal; just ignore extras.
        print(
            f"Warning {os.path.basename(processed_dir)}: processed has {len(extra)} extra frames; ignoring extras"
        )

    return True, orig_stems


def _write_side_by_side_video(
    original_dir: str,
    processed_dir: str,
    ordered_stems: list[str],
    out_path: str,
    fps: int,
    zoom: int | None,
    left_label: str,
    right_label: str,
    put_text: bool,
):
    orig_map = _build_frame_map(original_dir)
    proc_map = _build_frame_map(processed_dir)

    if not ordered_stems:
        raise ValueError("ordered_stems is empty")

    first_orig_path = os.path.join(original_dir, orig_map[ordered_stems[0]])
    first_proc_path = os.path.join(processed_dir, proc_map[ordered_stems[0]])

    orig0 = cv2.imread(first_orig_path)
    proc0 = cv2.imread(first_proc_path)
    if orig0 is None:
        raise RuntimeError(f"Failed to read {first_orig_path}")
    if proc0 is None:
        raise RuntimeError(f"Failed to read {first_proc_path}")

    if zoom is not None:
        orig0 = _center_square_crop(orig0, zoom)
        proc0 = _center_square_crop(proc0, zoom)

    # Match processed size to original size (after zoom crop, if any).
    target_h, target_w = orig0.shape[:2]
    if proc0.shape[:2] != (target_h, target_w):
        proc0 = cv2.resize(proc0, (target_w, target_h), interpolation=cv2.INTER_AREA)

    out_h, out_w = target_h, target_w * 2

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    def _annotate(frame, text: str):
        if not put_text:
            return frame
        # Simple shadow + white text for readability.
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    count = 0
    for st in ordered_stems:
        orig_path = os.path.join(original_dir, orig_map[st])
        proc_path = os.path.join(processed_dir, proc_map[st])

        orig = cv2.imread(orig_path)
        proc = cv2.imread(proc_path)
        if orig is None or proc is None:
            print(f"Warning: could not read frame {st}; skipping")
            continue

        if zoom is not None:
            orig = _center_square_crop(orig, zoom)
            proc = _center_square_crop(proc, zoom)

        if orig.shape[:2] != (target_h, target_w):
            orig = cv2.resize(orig, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if proc.shape[:2] != (target_h, target_w):
            proc = cv2.resize(proc, (target_w, target_h), interpolation=cv2.INTER_AREA)

        left = _annotate(orig, left_label)
        right = _annotate(proc, right_label)
        side_by_side = cv2.hconcat([left, right])
        writer.write(side_by_side)
        count += 1

    writer.release()
    print(f"Saved {out_path} ({count} frames)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side comparison videos between original DAVIS frames and processed output frames. "
            "Only generates a video when every original frame exists in the processed folder."
        )
    )
    parser.add_argument(
        "--original_root",
        type=str,
        default="./davis_test_data",
        help="Root directory with original sequences (e.g., ./davis_test_data/<seq>/00000.jpg)",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default="./output_davis",
        help="Root directory with processed sequences (e.g., ./output_davis/<seq>/00000.png)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_comparisons",
        help="Directory to write comparison videos into",
    )
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    parser.add_argument(
        "--zoom",
        type=int,
        default=None,
        help=(
            "Optional center square crop size (pixels). If set, each side is cropped to zoom x zoom before stacking."
        ),
    )
    parser.add_argument(
        "--left_label",
        type=str,
        default="original",
        help="Text label to draw on the left panel",
    )
    parser.add_argument(
        "--right_label",
        type=str,
        default="processed",
        help="Text label to draw on the right panel",
    )
    parser.add_argument(
        "--no_text",
        action="store_true",
        help="Disable drawing labels on frames",
    )

    args = parser.parse_args()

    pairs = _find_sequence_pairs(args.original_root, args.processed_root)
    if not pairs:
        print("No sequence pairs found.")
        raise SystemExit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    created = 0
    for pair in pairs:
        ok, ordered_stems = _validate_frames_complete(pair.original_dir, pair.processed_dir)
        if not ok:
            continue

        out_path = os.path.join(args.output_dir, f"{pair.name}.mp4")
        _write_side_by_side_video(
            original_dir=pair.original_dir,
            processed_dir=pair.processed_dir,
            ordered_stems=ordered_stems,
            out_path=out_path,
            fps=args.fps,
            zoom=args.zoom,
            left_label=args.left_label,
            right_label=args.right_label,
            put_text=not args.no_text,
        )
        created += 1

    print(f"Done. Created {created} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
