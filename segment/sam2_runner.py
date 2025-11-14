#!/usr/bin/env python3
"""
SAM2 segmentation runner that consumes robust prompts and produces binary masks.

Can be run as a standalone script or imported as a module.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

MaskGenerator = Callable[[np.ndarray, Dict[str, object]], np.ndarray]


@dataclass
class Prompt:
    frame_key: str
    bbox: Sequence[float]
    center: Sequence[float]
    metadata: Dict[str, object]


def _box_mask(image_shape, box: Sequence[float]) -> np.ndarray:
    h, w = image_shape
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, int(np.floor(x1))))
    x2 = max(0, min(w, int(np.ceil(x2))))
    y1 = max(0, min(h, int(np.floor(y1))))
    y2 = max(0, min(h, int(np.ceil(y2))))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def _guess_frame_path(frames_dir: Path, frame_key: str) -> Optional[Path]:
    """Best-effort resolver that tolerates keys with/without extensions/padding."""
    candidates = []
    key_path = Path(frame_key)

    # if caller already provided an extension, try the exact name first
    if key_path.suffix:
        candidates.append(key_path.name)
    base = key_path.stem if key_path.suffix else key_path.name

    if base.isdigit():
        z = base.zfill(6)
        candidates.extend([f"{z}.jpg", f"{z}.png", f"{z}.jpeg"])

    candidates.extend([f"{base}.jpg", f"{base}.png", f"{base}.jpeg"])

    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = frames_dir / name
        if candidate.exists():
            return candidate
    return None


class SAM2Runner:
    """
    Utility that consumes robust prompts and produces binary masks either via SAM2
    or via a fallback box rasterizer.
    """

    def __init__(self, generator: Optional[MaskGenerator] = None):
        self.generator = generator

    def run(
        self,
        prompts: Iterable[Dict[str, object]],
        frames_dir: Path,
        out_dir: Path,
        overwrite: bool = False,
        use_box_fallback: bool = False,
        verbose: bool = True,
        debug: bool = False,
    ):
        """
        Process prompts and generate segmentation masks.

        Args:
            prompts: Iterable of prompt dictionaries
            frames_dir: Directory containing frame images
            out_dir: Output directory for masks
            overwrite: Whether to overwrite existing masks
            use_box_fallback: Use simple box masks instead of SAM2
            verbose: Show progress bar
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        masks_by_frame: Dict[str, np.ndarray] = {}
        cached_key: Optional[str] = None
        cached_image: Optional[np.ndarray] = None
        frame_file_map: Dict[str, str] = {}

        prompt_iter = tqdm(prompts, desc="Generating masks") if verbose else prompts

        for prompt in prompt_iter:
            frame_key = str(prompt.get("frame_key"))
            frame_path = _guess_frame_path(frames_dir, frame_key)
            if frame_path is None:
                if verbose:
                    print(f"[warn] Frame not found for key={frame_key}")
                if debug:
                    candidates = []
                    key_path = Path(frame_key)
                    if key_path.suffix:
                        candidates.append(key_path.name)
                    base = key_path.stem if key_path.suffix else key_path.name
                    if base.isdigit():
                        z = base.zfill(6)
                        candidates.extend([f"{z}.jpg", f"{z}.png", f"{z}.jpeg"])
                    candidates.extend([f"{base}.jpg", f"{base}.png", f"{base}.jpeg"])
                    print(f"        frames_dir: {frames_dir}")
                    print(f"        tried: {candidates}")
                    try:
                        entries = sorted(p.name for p in frames_dir.iterdir())
                        print(f"        dir contains ({len(entries)} files): {entries[:20]}{' ...' if len(entries) > 20 else ''}")
                    except FileNotFoundError:
                        print(f"        frames_dir does not exist")
                continue
            frame_file_map.setdefault(frame_key, frame_path.name)
            if frame_key != cached_key:
                image_bgr = cv2.imread(str(frame_path))
                if image_bgr is None:
                    if verbose:
                        print(f"[warn] Failed to read {frame_path}")
                    continue
                cached_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                cached_key = frame_key
            image_rgb = cached_image
            if image_rgb is None:
                continue

            if use_box_fallback or self.generator is None:
                mask_arr = _box_mask(image_rgb.shape[:2], prompt["bbox"])
            else:
                mask_arr = self.generator(
                    image_rgb,
                    {"bbox": prompt["bbox"], "center": prompt.get("center"), "prompt": prompt},
                )
            mask_arr = np.asarray(mask_arr)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.squeeze()
            mask_arr = (mask_arr > 0).astype(np.uint8)

            prev = masks_by_frame.get(frame_key)
            masks_by_frame[frame_key] = mask_arr if prev is None else np.maximum(prev, mask_arr)

        for frame_key, mask in masks_by_frame.items():
            fname = frame_file_map.get(frame_key, f"{frame_key}.png")
            out_path = out_dir / f"{Path(fname).stem}_mask.png"
            if out_path.exists() and not overwrite:
                if verbose:
                    print(f"[skip] {out_path} exists")
                continue
            cv2.imwrite(str(out_path), mask * 255)

        if verbose:
            print(f"[save] {len(masks_by_frame)} masks saved to {out_dir}")


def parse_args():
    """Parse command-line arguments for standalone execution."""
    ap = argparse.ArgumentParser(
        description="Generate segmentation masks from robust prompt JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use box fallback (no SAM2 required)
  python -m segment.sam2_runner --prompts robust_prompts.json --out-dir masks/ --box-fallback

  # With SAM2 (requires custom generator implementation)
  python -m segment.sam2_runner --prompts robust_prompts.json --out-dir masks/
        """,
    )
    ap.add_argument("--prompts", "--prompts-json", dest="prompts_json", required=True,
                    help="Path to robust_prompts.json file")
    ap.add_argument("--out-dir", "--output", required=True,
                    help="Output directory for mask images")
    ap.add_argument("--frames-dir", default=None,
                    help="Optional override for frames directory (overrides JSON)")
    ap.add_argument("--box-fallback", "--box-mask-fallback", action="store_true",
                    help="Use simple box masks instead of SAM2 (no model required)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing mask files")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress progress output")
    ap.add_argument("--debug", action="store_true",
                    help="Print extra diagnostics when frames are missing")
    return ap.parse_args()


def main():
    """CLI entrypoint for standalone execution."""
    args = parse_args()

    with open(args.prompts_json, "r") as f:
        payload = json.load(f)

    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(payload["frames_dir"])
    prompts = payload.get("prompts", [])

    if not prompts:
        print("[info] No prompts found in JSON")
        return

    runner = SAM2Runner(generator=None)
    runner.run(
        prompts=prompts,
        frames_dir=frames_dir,
        out_dir=Path(args.out_dir),
        overwrite=args.overwrite,
        use_box_fallback=args.box_fallback,
        verbose=not args.quiet,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
