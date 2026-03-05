#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    # Keep alpha if present
    if img.mode in ("P", "LA"):
        return img.convert("RGBA")
    if img.mode not in ("RGB", "RGBA"):
        return img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return img

def upscale_to_target(
    img: Image.Image,
    target_size: tuple[int, int],
    resample,
    two_step: bool,
    unsharp: tuple[float, int, int] | None,
) -> Image.Image:
    img = ensure_mode(img)

    if not two_step:
        out = img.resize(target_size, resample=resample)
    else:
        # 2x stepping tends to reduce some ringing on large jumps
        out = img
        tw, th = target_size
        while out.width * 2 <= tw and out.height * 2 <= th:
            out = out.resize((out.width * 2, out.height * 2), resample=resample)
        if (out.width, out.height) != target_size:
            out = out.resize(target_size, resample=resample)

    if unsharp:
        r, pct, thr = unsharp
        out = out.filter(ImageFilter.UnsharpMask(radius=r, percent=pct, threshold=thr))

    return out

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
    ap = argparse.ArgumentParser(description="Batch upscale to a physical print size and set DPI metadata.")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)

    ap.add_argument("--out_inches", type=float, default=12.0, help="Output size in inches (square). Default 12in (1ft).")
    ap.add_argument("--dpi", type=int, default=300, help="DPI metadata + pixel target. Default 300.")
    ap.add_argument("--resample", choices=RESAMPLE_MAP.keys(), default="lanczos")
    ap.add_argument("--two_step", action="store_true")

    # Texture pipeline sharpen (optional)
    ap.add_argument("--unsharp", nargs=3, type=float, metavar=("RADIUS", "PERCENT", "THRESHOLD"),
                    default=None, help="Example: --unsharp 1.2 140 3")

    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--suffix", default="_12in_300dpi")
    ap.add_argument("--format", choices=["png", "jpg", "webp", "tif"], default="png", help="Output format/extension")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quality", type=int, default=92)

    args = ap.parse_args()

    target_px = int(round(args.out_inches * args.dpi))  # 12in * 300 = 3600
    target_size = (target_px, target_px)

    resample = RESAMPLE_MAP[args.resample]
    unsharp = None
    if args.unsharp is not None:
        r, p, t = args.unsharp
        unsharp = (float(r), int(round(p)), int(round(t)))

    fmt_ext = "." + args.format.lower()

    total = saved = skipped = failed = 0

    for src in iter_images(args.input_dir, args.recursive):
        total += 1
        try:
            rel = src.relative_to(args.input_dir)
            out_name = rel.stem + args.suffix + fmt_ext
            out_path = args.output_dir / rel.parent / out_name

            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            with Image.open(src) as img:
                out_img = upscale_to_target(
                    img=img,
                    target_size=target_size,
                    resample=resample,
                    two_step=args.two_step,
                    unsharp=unsharp,
                )
                save_with_dpi(out_img, out_path, dpi=args.dpi, quality=args.quality)

            saved += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {src} -> {e}")

    print(f"Total: {total} | Saved: {saved} | Skipped: {skipped} | Failed: {failed}")
    print(f"Target: {target_size[0]}x{target_size[1]} px @ {args.dpi} dpi")

if __name__ == "__main__":
    main()
