#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from PIL import Image, ImageFilter

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

RESAMPLE_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def iter_images(input_dir: Path, recursive: bool):
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for p in it:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def ensure_mode(img: Image.Image) -> Image.Image:
    # Convert to RGB/RGBA in a safe way
    if img.mode in ("P", "LA"):
        return img.convert("RGBA")
    if img.mode not in ("RGB", "RGBA"):
        return img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return img


def resize_two_step(img: Image.Image, target_size: tuple[int, int], resample) -> Image.Image:
    out = img
    tw, th = target_size
    while out.width * 2 <= tw and out.height * 2 <= th:
        out = out.resize((out.width * 2, out.height * 2), resample=resample)
    if (out.width, out.height) != target_size:
        out = out.resize(target_size, resample=resample)
    return out


def quantize_color(pixel):
    # Softly bucket colors so highlights/shadows don't dominate
    if isinstance(pixel, int):
        return pixel
    if len(pixel) >= 3:
        r, g, b = pixel[:3]
        return (round(r / 8) * 8, round(g / 8) * 8, round(b / 8) * 8)
    return pixel


def get_dominant_color(img: Image.Image):
    """
    Find the most common non-extreme color.
    Good for mostly solid-color LEGO images with some shading/highlights.
    """
    rgb = img.convert("RGB")
    small = rgb.resize((80, 80), Image.Resampling.BICUBIC)
    pixels = list(small.getdata())

    # Ignore extreme highlights/shadows
    filtered = []
    for p in pixels:
        r, g, b = p
        if max(r, g, b) > 245:
            continue
        if min(r, g, b) < 10:
            continue
        filtered.append(quantize_color(p))

    if not filtered:
        # Fallback to average color
        return rgb.resize((1, 1), Image.Resampling.BICUBIC).getpixel((0, 0))

    return Counter(filtered).most_common(1)[0][0]


def add_bleed_with_dominant_color(img: Image.Image, bleed_px: int) -> Image.Image:
    w, h = img.size
    dominant = get_dominant_color(img)

    if img.mode == "RGBA":
        fill = dominant + (255,)
    else:
        fill = dominant

    canvas = Image.new(img.mode, (w + 2 * bleed_px, h + 2 * bleed_px), fill)
    canvas.paste(img, (bleed_px, bleed_px))
    return canvas


def save_with_dpi(img: Image.Image, out_path: Path, dpi: int, quality: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    save_kwargs = {"dpi": (dpi, dpi)}

    if ext in (".jpg", ".jpeg"):
        if img.mode == "RGBA":
            img = img.convert("RGB")
        save_kwargs.update({"quality": quality, "optimize": True, "progressive": True})
    elif ext == ".png":
        save_kwargs.update({"optimize": True})
    elif ext == ".webp":
        save_kwargs.update({"quality": quality, "method": 6})

    img.save(out_path, **save_kwargs)


def main():
    ap = argparse.ArgumentParser(
        description="Batch upscale images to print size and add 1-inch dominant-color bleed."
    )
    ap.add_argument("input_dir", type=Path, help="Folder containing source images")
    ap.add_argument("output_dir", type=Path, help="Folder for output images")

    ap.add_argument("--out_inches", type=float, default=12.0, help="Final art size before bleed, in inches. Default 12.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI. Default 300.")
    ap.add_argument("--resample", choices=RESAMPLE_MAP.keys(), default="bicubic")
    ap.add_argument("--two_step", action="store_true", help="Use 2x stepping for large upscales")
    ap.add_argument("--unsharp", nargs=3, type=float, metavar=("RADIUS", "PERCENT", "THRESHOLD"),
                    default=(0.8, 160, 2),
                    help="UnsharpMask settings. Default: 0.8 160 2")

    ap.add_argument("--bleed_inches", type=float, default=1.0, help="Bleed on each side in inches. Default 1.0")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--suffix", default="_print")
    ap.add_argument("--format", choices=["png", "jpg", "webp", "tif"], default="png")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quality", type=int, default=92)

    args = ap.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input folder not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_px = int(round(args.out_inches * args.dpi))         # 12in * 300 = 3600
    bleed_px = int(round(args.bleed_inches * args.dpi))       # 1in * 300 = 300
    target_size = (target_px, target_px)
    resample = RESAMPLE_MAP[args.resample]

    r, p, t = args.unsharp
    unsharp = (float(r), int(round(p)), int(round(t)))

    total = saved = skipped = failed = 0

    for src in iter_images(args.input_dir, args.recursive):
        total += 1
        try:
            rel = src.relative_to(args.input_dir)
            out_ext = "." + args.format.lower()
            out_path = args.output_dir / rel.parent / f"{rel.stem}{args.suffix}{out_ext}"

            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            with Image.open(src) as img:
                img = ensure_mode(img)

                if args.two_step:
                    out = resize_two_step(img, target_size, resample)
                else:
                    out = img.resize(target_size, resample=resample)

                if unsharp:
                    out = out.filter(
                        ImageFilter.UnsharpMask(
                            radius=unsharp[0],
                            percent=unsharp[1],
                            threshold=unsharp[2],
                        )
                    )

                out = add_bleed_with_dominant_color(out, bleed_px)

                save_with_dpi(out, out_path, dpi=args.dpi, quality=args.quality)

            saved += 1
            print(f"[OK] {src.name} -> {out_path.name}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {src} -> {e}")

    final_px = target_px + 2 * bleed_px
    print()
    print(f"Total:   {total}")
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    print(f"Art size before bleed: {target_px} x {target_px} px")
    print(f"Bleed per side:        {bleed_px} px")
    print(f"Final file size:       {final_px} x {final_px} px @ {args.dpi} DPI")


if __name__ == "__main__":
    main()
