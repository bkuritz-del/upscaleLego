#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageFilter, ImageOps

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

RESAMPLE_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass
class UpscaleSettings:
    input_dir: Path
    output_dir: Path
    out_width_inches: float = 12.0
    out_height_inches: float = 12.0
    dpi: int = 300
    bleed_inches: float = 1.0
    bleed_mode: str = "dominant"  # dominant, edge, mirror, custom
    custom_bleed_color: tuple[int, int, int] | None = None
    resample: str = "bicubic"
    two_step: bool = True
    sharpen_enabled: bool = True
    unsharp_radius: float = 0.8
    unsharp_percent: int = 160
    unsharp_threshold: int = 2
    pad_to_square: bool = True
    background_mode: str = "auto"  # auto, transparent, white, black, custom
    custom_background_color: tuple[int, int, int] | None = None
    output_format_mode: str = "same"  # same, png, jpg, webp, tif
    suffix: str = "_print"
    recursive: bool = False
    overwrite: bool = False
    quality: int = 92


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def ensure_mode(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        return img
    if "A" in img.getbands():
        return img.convert("RGBA")
    return img.convert("RGB")


def quantize_color(pixel: tuple[int, int, int], step: int = 8) -> tuple[int, int, int]:
    r, g, b = pixel[:3]
    return (
        round(r / step) * step,
        round(g / step) * step,
        round(b / step) * step,
    )


def get_dominant_color(img: Image.Image) -> tuple[int, int, int]:
    rgb = img.convert("RGB")
    small = rgb.resize((80, 80), Image.Resampling.BICUBIC)
    pixels = list(small.getdata())

    filtered: list[tuple[int, int, int]] = []
    for r, g, b in pixels:
        if max(r, g, b) > 245:
            continue
        if min(r, g, b) < 10:
            continue
        filtered.append(quantize_color((r, g, b), step=8))

    if not filtered:
        return rgb.resize((1, 1), Image.Resampling.BICUBIC).getpixel((0, 0))

    return Counter(filtered).most_common(1)[0][0]


def get_auto_background_fill(img: Image.Image) -> tuple[int, int, int] | tuple[int, int, int, int]:
    dominant = get_dominant_color(img)
    if img.mode == "RGBA":
        return dominant + (0,)
    return dominant


def get_background_fill(
    img: Image.Image,
    background_mode: str,
    custom_color: tuple[int, int, int] | None,
) -> tuple[int, int, int] | tuple[int, int, int, int]:
    if background_mode == "transparent":
        return (0, 0, 0, 0) if img.mode == "RGBA" else (255, 255, 255)
    if background_mode == "white":
        return (255, 255, 255, 255) if img.mode == "RGBA" else (255, 255, 255)
    if background_mode == "black":
        return (0, 0, 0, 255) if img.mode == "RGBA" else (0, 0, 0)
    if background_mode == "custom" and custom_color is not None:
        return custom_color + ((255,) if img.mode == "RGBA" else ())
    return get_auto_background_fill(img)


def pad_to_square(
    img: Image.Image,
    background_mode: str = "auto",
    custom_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    width, height = img.size
    if width == height:
        return img

    size = max(width, height)
    fill = get_background_fill(img, background_mode, custom_color)
    canvas = Image.new(img.mode, (size, size), fill)

    x = (size - width) // 2
    y = (size - height) // 2

    if img.mode == "RGBA":
        canvas.paste(img, (x, y), img)
    else:
        canvas.paste(img, (x, y))
    return canvas


def resize_two_step(img: Image.Image, target_size: tuple[int, int], resample: int) -> Image.Image:
    out = img
    target_w, target_h = target_size

    while out.width * 2 <= target_w and out.height * 2 <= target_h:
        out = out.resize((out.width * 2, out.height * 2), resample=resample)

    if out.size != target_size:
        out = out.resize(target_size, resample=resample)

    return out


def add_bleed_dominant(img: Image.Image, bleed_px: int, custom_color: tuple[int, int, int] | None = None) -> Image.Image:
    width, height = img.size
    dominant = custom_color if custom_color is not None else get_dominant_color(img)

    if img.mode == "RGBA":
        fill = dominant + (255,)
    else:
        fill = dominant

    canvas = Image.new(img.mode, (width + 2 * bleed_px, height + 2 * bleed_px), fill)
    if img.mode == "RGBA":
        canvas.paste(img, (bleed_px, bleed_px), img)
    else:
        canvas.paste(img, (bleed_px, bleed_px))
    return canvas


def add_bleed_edge_extend(img: Image.Image, bleed_px: int) -> Image.Image:
    width, height = img.size
    canvas = Image.new(img.mode, (width + 2 * bleed_px, height + 2 * bleed_px))
    if img.mode == "RGBA":
        canvas.paste(img, (bleed_px, bleed_px), img)
    else:
        canvas.paste(img, (bleed_px, bleed_px))

    top = img.crop((0, 0, width, 1)).resize((width, bleed_px))
    bottom = img.crop((0, height - 1, width, height)).resize((width, bleed_px))
    left = img.crop((0, 0, 1, height)).resize((bleed_px, height))
    right = img.crop((width - 1, 0, width, height)).resize((bleed_px, height))

    canvas.paste(top, (bleed_px, 0))
    canvas.paste(bottom, (bleed_px, height + bleed_px))
    canvas.paste(left, (0, bleed_px))
    canvas.paste(right, (width + bleed_px, bleed_px))

    tl = img.crop((0, 0, 1, 1)).resize((bleed_px, bleed_px))
    tr = img.crop((width - 1, 0, width, 1)).resize((bleed_px, bleed_px))
    bl = img.crop((0, height - 1, 1, height)).resize((bleed_px, bleed_px))
    br = img.crop((width - 1, height - 1, width, height)).resize((bleed_px, bleed_px))

    canvas.paste(tl, (0, 0))
    canvas.paste(tr, (width + bleed_px, 0))
    canvas.paste(bl, (0, height + bleed_px))
    canvas.paste(br, (width + bleed_px, height + bleed_px))
    return canvas


def add_bleed_mirror(img: Image.Image, bleed_px: int) -> Image.Image:
    width, height = img.size
    expanded = ImageOps.expand(img, border=bleed_px)
    core = expanded.crop((bleed_px, bleed_px, bleed_px + width, bleed_px + height))
    expanded.paste(core, (bleed_px, bleed_px))

    left = img.crop((0, 0, min(bleed_px, width), height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    right = img.crop((max(0, width - bleed_px), 0, width, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    top = img.crop((0, 0, width, min(bleed_px, height))).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    bottom = img.crop((0, max(0, height - bleed_px), width, height)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if left.width < bleed_px:
        left = left.resize((bleed_px, height))
    if right.width < bleed_px:
        right = right.resize((bleed_px, height))
    if top.height < bleed_px:
        top = top.resize((width, bleed_px))
    if bottom.height < bleed_px:
        bottom = bottom.resize((width, bleed_px))

    expanded.paste(left, (0, bleed_px))
    expanded.paste(right, (bleed_px + width, bleed_px))
    expanded.paste(top, (bleed_px, 0))
    expanded.paste(bottom, (bleed_px, bleed_px + height))

    expanded.paste(top.crop((0, 0, bleed_px, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (0, 0))
    expanded.paste(top.crop((width - bleed_px, 0, width, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (bleed_px + width, 0))
    expanded.paste(bottom.crop((0, 0, bleed_px, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (0, bleed_px + height))
    expanded.paste(bottom.crop((width - bleed_px, 0, width, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (bleed_px + width, bleed_px + height))
    return expanded


def add_bleed(img: Image.Image, bleed_px: int, mode: str, custom_color: tuple[int, int, int] | None = None) -> Image.Image:
    if bleed_px <= 0:
        return img
    if mode == "edge":
        return add_bleed_edge_extend(img, bleed_px)
    if mode == "mirror":
        return add_bleed_mirror(img, bleed_px)
    if mode == "custom":
        return add_bleed_dominant(img, bleed_px, custom_color=custom_color)
    return add_bleed_dominant(img, bleed_px)


def get_output_extension(src: Path, output_format_mode: str) -> str:
    if output_format_mode == "same":
        return src.suffix.lower()
    mapping = {
        "png": ".png",
        "jpg": ".jpg",
        "webp": ".webp",
        "tif": ".tif",
    }
    return mapping[output_format_mode]


def save_with_dpi(img: Image.Image, out_path: Path, dpi: int, quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    save_kwargs = {"dpi": (dpi, dpi)}

    if ext in (".jpg", ".jpeg"):
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        save_kwargs.update({"quality": quality, "optimize": True, "progressive": True})

    elif ext == ".png":
        save_kwargs.update({"optimize": True})

    elif ext == ".webp":
        save_kwargs.update({"quality": quality, "method": 6})

    img.save(out_path, **save_kwargs)


def process_one_image(src: Path, settings: UpscaleSettings) -> tuple[bool, str]:
    rel = src.relative_to(settings.input_dir)
    out_ext = get_output_extension(src, settings.output_format_mode)
    out_path = settings.output_dir / rel.parent / f"{rel.stem}{settings.suffix}{out_ext}"

    if out_path.exists() and not settings.overwrite:
        return False, f"[SKIP] {src.name} -> already exists"

    target_width_px = int(round(settings.out_width_inches * settings.dpi))
    target_height_px = int(round(settings.out_height_inches * settings.dpi))
    bleed_px = int(round(settings.bleed_inches * settings.dpi))
    target_size = (target_width_px, target_height_px)

    with Image.open(src) as img:
        img.load()
        img = ensure_mode(img)

        if settings.pad_to_square:
            img = pad_to_square(img, settings.background_mode, settings.custom_background_color)

        resample = RESAMPLE_MAP[settings.resample]
        if settings.two_step:
            out = resize_two_step(img, target_size, resample)
        else:
            out = img.resize(target_size, resample=resample)

        if settings.sharpen_enabled:
            out = out.filter(
                ImageFilter.UnsharpMask(
                    radius=settings.unsharp_radius,
                    percent=settings.unsharp_percent,
                    threshold=settings.unsharp_threshold,
                )
            )

        out = add_bleed(out, bleed_px, settings.bleed_mode, settings.custom_bleed_color)
        save_with_dpi(out, out_path, settings.dpi, settings.quality)

    return True, f"[OK]   {src.name} -> {out_path.name}"


def process_batch(
    settings: UpscaleSettings,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    if not settings.input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {settings.input_dir}")
    if not settings.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {settings.input_dir}")

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_images(settings.input_dir, settings.recursive))
    total = len(files)
    saved = skipped = failed = 0

    for idx, src in enumerate(files, start=1):
        try:
            did_save, message = process_one_image(src, settings)
            if did_save:
                saved += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            message = f"[FAIL] {src.name} -> {exc}"

        if progress_callback:
            progress_callback(idx, total, message)

    return {
        "total": total,
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
        "art_width_px": int(round(settings.out_width_inches * settings.dpi)),
        "art_height_px": int(round(settings.out_height_inches * settings.dpi)),
        "bleed_px": int(round(settings.bleed_inches * settings.dpi)),
        "final_width_px": int(round(settings.out_width_inches * settings.dpi)) + 2 * int(round(settings.bleed_inches * settings.dpi)),
        "final_height_px": int(round(settings.out_height_inches * settings.dpi)) + 2 * int(round(settings.bleed_inches * settings.dpi)),
    }
