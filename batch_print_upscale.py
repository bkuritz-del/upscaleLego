#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

RESAMPLE_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def ensure_mode(img: Image.Image) -> Image.Image:
    """
    Convert image into a safe working mode while preserving alpha if present.
    """
    if img.mode in ("RGB", "RGBA"):
        return img

    if "A" in img.getbands():
        return img.convert("RGBA")

    return img.convert("RGB")


def resize_two_step(img: Image.Image, target_size: tuple[int, int], resample: int) -> Image.Image:
    """
    Upscale in 2x steps, then do the final resize.
    Helps reduce artifacts on very large enlargements.
    """
    out = img
    target_w, target_h = target_size

    while out.width * 2 <= target_w and out.height * 2 <= target_h:
        out = out.resize((out.width * 2, out.height * 2), resample=resample)

    if out.size != target_size:
        out = out.resize(target_size, resample=resample)

    return out


def quantize_color(pixel: tuple[int, int, int], step: int = 8) -> tuple[int, int, int]:
    """
    Bucket colors slightly so highlights/shadows don't overpower the dominant color.
    """
    r, g, b = pixel[:3]
    return (
        round(r / step) * step,
        round(g / step) * step,
        round(b / step) * step,
    )


def get_dominant_color(img: Image.Image) -> tuple[int, int, int]:
    """
    Find the most common non-extreme color in the image.
    Well suited to mostly solid-color LEGO tile images with shading.
    """
    rgb = img.convert("RGB")
    small = rgb.resize((80, 80), Image.Resampling.BICUBIC)
    pixels = list(small.getdata())

    filtered: list[tuple[int, int, int]] = []
    for r, g, b in pixels:
        # Ignore very bright highlights and very dark shadow extremes
        if max(r, g, b) > 245:
            continue
        if min(r, g, b) < 10:
            continue
        filtered.append(quantize_color((r, g, b), step=8))

    if not filtered:
        return rgb.resize((1, 1), Image.Resampling.BICUBIC).getpixel((0, 0))

    return Counter(filtered).most_common(1)[0][0]


def add_solid_bleed(img: Image.Image, bleed_px: int) -> Image.Image:
    """
    Add solid-color bleed around image using dominant image color.
    """
    width, height = img.size
    dominant = get_dominant_color(img)

    if img.mode == "RGBA":
        fill = dominant + (255,)
    else:
        fill = dominant

    canvas = Image.new(img.mode, (width + 2 * bleed_px, height + 2 * bleed_px), fill)
    canvas.paste(img, (bleed_px, bleed_px))
    return canvas


def save_with_dpi(img: Image.Image, out_path: Path, dpi: int, quality: int) -> None:
    """
    Save while preserving input extension behavior as much as possible.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    save_kwargs = {"dpi": (dpi, dpi)}

    if ext in (".jpg", ".jpeg"):
        if img.mode == "RGBA":
            # JPEG does not support transparency
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        save_kwargs.update({
            "quality": quality,
            "optimize": True,
            "progressive": True,
        })

    elif ext == ".png":
        save_kwargs.update({
            "optimize": True,
        })

    elif ext == ".webp":
        save_kwargs.update({
            "quality": quality,
            "method": 6,
        })

    img.save(out_path, **save_kwargs)


def process_image(
    src: Path,
    input_dir: Path,
    output_dir: Path,
    target_size: tuple[int, int],
    bleed_px: int,
    dpi: int,
    resample: int,
    two_step: bool,
    unsharp: tuple[float, int, int] | None,
    suffix: str,
    quality: int,
    overwrite: bool,
) -> str:
    rel = src.relative_to(input_dir)
    out_ext = src.suffix.lower()
    out_path = output_dir / rel.parent / f"{rel.stem}{suffix}{out_ext}"

    if out_path.exists() and not overwrite:
        return f"[SKIP] {src.name} -> already exists"

    with Image.open(src) as img:
        img.load()
        img = ensure_mode(img)

        if two_step:
            out = resize_two_step(img, target_size, resample)
        else:
            out = img.resize(target_size, resample=resample)

        if unsharp is not None:
            radius, percent, threshold = unsharp
            out = out.filter(
                ImageFilter.UnsharpMask(
                    radius=radius,
                    percent=percent,
                    threshold=threshold,
                )
            )

        out = add_solid_bleed(out, bleed_px)
        save_with_dpi(out, out_path, dpi=dpi, quality=quality)

    return f"[OK]   {src.name} -> {out_path.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch upscale images for print and add solid dominant-color bleed."
    )

    parser.add_argument("input_dir", type=Path, help="Folder containing source images")
    parser.add_argument("output_dir", type=Path, help="Folder where processed images will be written")

    parser.add_argument("--out_inches", type=float, default=12.0, help="Final artwork size before bleed, in inches")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI metadata and pixel math basis")
    parser.add_argument("--bleed_inches", type=float, default=1.0, help="Bleed size on each side, in inches")

    parser.add_argument(
        "--resample",
        choices=RESAMPLE_MAP.keys(),
        default="bicubic",
        help="Resize filter",
    )
    parser.add_argument(
        "--two_step",
        action="store_true",
        help="Upscale in repeated 2x steps before final resize",
    )
    parser.add_argument(
        "--unsharp",
        nargs=3,
        type=float,
        metavar=("RADIUS", "PERCENT", "THRESHOLD"),
        default=(0.8, 160, 2),
        help="Unsharp mask settings; default: 0.8 160 2",
    )

    parser.add_argument("--recursive", action="store_true", help="Process subfolders recursively")
    parser.add_argument("--suffix", default="_print", help="Suffix added to output filename")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--quality", type=int, default=92, help="JPEG/WebP quality")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input folder not found: {args.input_dir}")

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input path is not a folder: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_px = int(round(args.out_inches * args.dpi))
    bleed_px = int(round(args.bleed_inches * args.dpi))
    final_px = target_px + (2 * bleed_px)
    target_size = (target_px, target_px)

    resample = RESAMPLE_MAP[args.resample]

    if args.unsharp is None:
        unsharp = None
    else:
        radius, percent, threshold = args.unsharp
        unsharp = (float(radius), int(round(percent)), int(round(threshold)))

    total = 0
    saved = 0
    skipped = 0
    failed = 0

    for src in iter_images(args.input_dir, args.recursive):
        total += 1
        try:
            message = process_image(
                src=src,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                target_size=target_size,
                bleed_px=bleed_px,
                dpi=args.dpi,
                resample=resample,
                two_step=args.two_step,
                unsharp=unsharp,
                suffix=args.suffix,
                quality=args.quality,
                overwrite=args.overwrite,
            )
            print(message)
            if message.startswith("[OK]"):
                saved += 1
            elif message.startswith("[SKIP]"):
                skipped += 1

        except Exception as exc:
            failed += 1
            print(f"[FAIL] {src} -> {exc}")

    print("\nDone.")
    print(f"Total files found:      {total}")
    print(f"Saved:                  {saved}")
    print(f"Skipped existing:       {skipped}")
    print(f"Failed:                 {failed}")
    print(f"Artwork size:           {target_px} x {target_px} px")
    print(f"Bleed per side:         {bleed_px} px")
    print(f"Final output size:      {final_px} x {final_px} px")
    print(f"DPI metadata:           {args.dpi}")
    print(f"Output format behavior: same as input")


if __name__ == "__main__":
    main()
