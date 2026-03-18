#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from upscale_core_v2 import UpscaleSettings, process_one_image


SOURCE_FILE = Path("0011__block.tif")
OUTPUT_DIR = SOURCE_FILE.parent / "V2_TEST_OUTPUT"

settings = UpscaleSettings(
    input_dir=SOURCE_FILE.parent,
    output_dir=OUTPUT_DIR,
    scale_percent=1027.636,
    dpi=300,
    bleed_inches=1.0,
    bleed_mode="edge_miter",
    custom_bleed_color=None,
    resample="v2_edge_guided",
    two_step=True,
    sharpen_enabled=True,
    unsharp_radius=0.8,
    unsharp_percent=160,
    unsharp_threshold=2,
    pad_to_square=True,
    background_mode="auto",
    custom_background_color=None,
    output_format_mode="same",
    suffix="_v2test",
    recursive=False,
    overwrite=True,
    quality=92,
)


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    did_save, message = process_one_image(SOURCE_FILE, settings)
    print(message)

    if did_save:
        print("Done.")
        print(f"Output folder: {OUTPUT_DIR}")
    else:
        print("No file saved.")


if __name__ == "__main__":
    main()
