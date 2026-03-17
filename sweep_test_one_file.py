#!/usr/bin/env python3
from __future__ import annotations

from itertools import product
from pathlib import Path

from core import UpscaleSettings, process_one_image

# =========================
# EDIT THESE TWO PATHS
# =========================
SOURCE_FILE = Path(r"C:\path\to\your\test_file.tif")
OUTPUT_DIR = Path(r"C:\path\to\your\output_tests")

# =========================
# 24-output high-probability sweep
# =========================
SCALE_PERCENTS = [1027.636]
RESAMPLES = ["lanczos", "bicubic"]
TWO_STEP_OPTIONS = [True]
SHARPEN_OPTIONS = [True, False]
UNSHARP_RADII = [0.8]
UNSHARP_PERCENTS = [140, 160, 180]
UNSHARP_THRESHOLDS = [2]

# Fixed settings
DPI = 300
BLEED_INCHES = 1.0
BLEED_MODE = "edge_miter"
PAD_TO_SQUARE = True
BACKGROUND_MODE = "auto"
OUTPUT_FORMAT_MODE = "same"
QUALITY = 92
OVERWRITE = True


def bool_tag(value: bool) -> str:
    return "Y" if value else "N"


def clean_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = list(
        product(
            SCALE_PERCENTS,
            RESAMPLES,
            TWO_STEP_OPTIONS,
            SHARPEN_OPTIONS,
            UNSHARP_RADII,
            UNSHARP_PERCENTS,
            UNSHARP_THRESHOLDS,
        )
    )

    print(f"Source: {SOURCE_FILE}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Total combinations: {len(combos)}")
    print("")

    total = 0
    saved = 0
    failed = 0

    for (
        scale_percent,
        resample,
        two_step,
        sharpen_enabled,
        unsharp_radius,
        unsharp_percent,
        unsharp_threshold,
    ) in combos:
        total += 1

        suffix = (
            f"_sc{clean_float(scale_percent)}"
            f"_rs-{resample}"
            f"_2s-{bool_tag(two_step)}"
            f"_sh-{bool_tag(sharpen_enabled)}"
            f"_r-{clean_float(unsharp_radius)}"
            f"_p-{unsharp_percent}"
            f"_t-{unsharp_threshold}"
        )

        settings = UpscaleSettings(
            input_dir=SOURCE_FILE.parent,
            output_dir=OUTPUT_DIR,
            scale_percent=scale_percent,
            dpi=DPI,
            bleed_inches=BLEED_INCHES,
            bleed_mode=BLEED_MODE,
            custom_bleed_color=None,
            resample=resample,
            two_step=two_step,
            sharpen_enabled=sharpen_enabled,
            unsharp_radius=unsharp_radius,
            unsharp_percent=unsharp_percent,
            unsharp_threshold=unsharp_threshold,
            pad_to_square=PAD_TO_SQUARE,
            background_mode=BACKGROUND_MODE,
            custom_background_color=None,
            output_format_mode=OUTPUT_FORMAT_MODE,
            suffix=suffix,
            recursive=False,
            overwrite=OVERWRITE,
            quality=QUALITY,
        )

        try:
            did_save, message = process_one_image(SOURCE_FILE, settings)
            if did_save:
                saved += 1
                print(f"[{total:02d}/{len(combos):02d}] {message}")
            else:
                print(f"[{total:02d}/{len(combos):02d}] SKIPPED {message}")
        except Exception as exc:
            failed += 1
            print(f"[{total:02d}/{len(combos):02d}] FAIL {suffix} -> {exc}")

    print("")
    print("Done.")
    print(f"Total:  {total}")
    print(f"Saved:  {saved}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
