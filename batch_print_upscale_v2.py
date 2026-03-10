#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from upscale_core import UpscaleSettings, process_batch


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be in R,G,B format")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must contain integers") from exc
    if any(c < 0 or c > 255 for c in rgb):
        raise argparse.ArgumentTypeError("Color values must be between 0 and 255")
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch upscale images for print with bleed.")

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)

    parser.add_argument("--out_width_inches", type=float, default=12.0)
    parser.add_argument("--out_height_inches", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bleed_inches", type=float, default=1.0)

    parser.add_argument("--bleed_mode", choices=["dominant", "edge", "mirror", "custom"], default="dominant")
    parser.add_argument("--custom_bleed_color", type=parse_rgb, default=None)

    parser.add_argument("--resample", choices=["nearest", "bilinear", "bicubic", "lanczos"], default="bicubic")
    parser.add_argument("--two_step", action="store_true")
    parser.add_argument("--no_sharpen", action="store_true")
    parser.add_argument("--unsharp", nargs=3, type=float, metavar=("RADIUS", "PERCENT", "THRESHOLD"), default=(0.8, 160, 2))

    parser.add_argument("--pad_to_square", action="store_true")
    parser.add_argument("--background_mode", choices=["auto", "transparent", "white", "black", "custom"], default="auto")
    parser.add_argument("--custom_background_color", type=parse_rgb, default=None)

    parser.add_argument("--output_format_mode", choices=["same", "png", "jpg", "webp", "tif"], default="same")
    parser.add_argument("--suffix", default="_print")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quality", type=int, default=92)

    args = parser.parse_args()

    radius, percent, threshold = args.unsharp

    settings = UpscaleSettings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        out_width_inches=args.out_width_inches,
        out_height_inches=args.out_height_inches,
        dpi=args.dpi,
        bleed_inches=args.bleed_inches,
        bleed_mode=args.bleed_mode,
        custom_bleed_color=args.custom_bleed_color,
        resample=args.resample,
        two_step=args.two_step,
        sharpen_enabled=not args.no_sharpen,
        unsharp_radius=float(radius),
        unsharp_percent=int(round(percent)),
        unsharp_threshold=int(round(threshold)),
        pad_to_square=args.pad_to_square,
        background_mode=args.background_mode,
        custom_background_color=args.custom_background_color,
        output_format_mode=args.output_format_mode,
        suffix=args.suffix,
        recursive=args.recursive,
        overwrite=args.overwrite,
        quality=args.quality,
    )

    def progress(idx: int, total: int, message: str) -> None:
        print(f"[{idx}/{total}] {message}")

    summary = process_batch(settings, progress_callback=progress)

    print("\nDone.")
    print(f"Total files found:      {summary['total']}")
    print(f"Saved:                  {summary['saved']}")
    print(f"Skipped existing:       {summary['skipped']}")
    print(f"Failed:                 {summary['failed']}")
    print(f"Artwork size:           {summary['art_width_px']} x {summary['art_height_px']} px")
    print(f"Bleed per side:         {summary['bleed_px']} px")
    print(f"Final output size:      {summary['final_width_px']} x {summary['final_height_px']} px")
    print(f"DPI metadata:           {args.dpi}")
    print(f"Output format mode:     {args.output_format_mode}")


if __name__ == "__main__":
    main()
