#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
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
    scale_percent: float = 1027.636
    dpi: int = 300
    bleed_inches: float = 1.0
    bleed_mode: str = "dominant"  # dominant, edge, edge_miter, mirror, custom
    custom_bleed_color: tuple[int, int, int] | None = None
    resample: str = "bicubic"  # nearest, bilinear, bicubic, lanczos, edge_directed
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
    """
    Preserve CMYK when present.
    Only convert modes that are awkward for processing.
    """
    if img.mode in ("RGB", "RGBA", "CMYK"):
        return img

    if img.mode == "P":
        return img.convert("RGB")

    if img.mode == "L":
        return img.convert("RGB")

    if "A" in img.getbands():
        return img.convert("RGBA")

    return img.convert("RGB")


def rgb_to_cmyk(rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """
    Simple RGB -> CMYK conversion for fills/bleeds.
    This is not ICC-managed, but is sufficient for internal fill colors.
    """
    r, g, b = [x / 255.0 for x in rgb]

    k = 1.0 - max(r, g, b)
    if k >= 1.0:
        return (0, 0, 0, 255)

    c = (1.0 - r - k) / (1.0 - k)
    m = (1.0 - g - k) / (1.0 - k)
    y = (1.0 - b - k) / (1.0 - k)

    return (
        int(round(c * 255)),
        int(round(m * 255)),
        int(round(y * 255)),
        int(round(k * 255)),
    )


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


def get_auto_background_fill(
    img: Image.Image,
) -> tuple[int, int, int] | tuple[int, int, int, int]:
    dominant = get_dominant_color(img)

    if img.mode == "RGBA":
        return dominant + (0,)
    if img.mode == "CMYK":
        return rgb_to_cmyk(dominant)
    return dominant


def get_background_fill(
    img: Image.Image,
    background_mode: str,
    custom_color: tuple[int, int, int] | None,
) -> tuple[int, int, int] | tuple[int, int, int, int]:
    if img.mode == "CMYK":
        if background_mode == "transparent":
            # CMYK has no alpha; safest fallback is white
            return (0, 0, 0, 0)
        if background_mode == "white":
            return (0, 0, 0, 0)
        if background_mode == "black":
            return (0, 0, 0, 255)
        if background_mode == "custom" and custom_color is not None:
            return rgb_to_cmyk(custom_color)
        return get_auto_background_fill(img)

    if background_mode == "transparent":
        return (0, 0, 0, 0) if img.mode == "RGBA" else (255, 255, 255)
    if background_mode == "white":
        return (255, 255, 255, 255) if img.mode == "RGBA" else (255, 255, 255)
    if background_mode == "black":
        return (0, 0, 0, 255) if img.mode == "RGBA" else (0, 0, 0)
    if background_mode == "custom" and custom_color is not None:
        return custom_color + ((255,) if img.mode == "RGBA" else ())
    return get_auto_background_fill(img)


def pad_to_square_canvas(
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


def _pil_to_luma_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _edge_map_from_luma(luma: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(luma, dtype=np.float32)
    gy = np.zeros_like(luma, dtype=np.float32)

    gx[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
    gy[1:-1, :] = luma[2:, :] - luma[:-2, :]

    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def _array_to_pil_gray(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _blend_channel(base: np.ndarray, sharp: np.ndarray, weight: np.ndarray) -> np.ndarray:
    out = base * (1.0 - weight) + sharp * weight
    return np.clip(out, 0, 255)


def resize_edge_directed(
    img: Image.Image,
    target_size: tuple[int, int],
    two_step: bool = True,
    base_resample: int = Image.Resampling.LANCZOS,
    edge_strength: float = 0.35,
    edge_gamma: float = 0.8,
) -> Image.Image:
    """
    Edge-aware upscale:
    1. upscale normally
    2. build edge map from original luminance
    3. upscale edge map
    4. use it to apply local edge crisping only where edges are strong
    """
    if img.size == target_size:
        return img.copy()

    if two_step:
        base = resize_two_step(img, target_size, base_resample)
    else:
        base = img.resize(target_size, resample=base_resample)

    orig_luma = _pil_to_luma_array(img)
    edge_map = _edge_map_from_luma(orig_luma)

    edge_img = _array_to_pil_gray(edge_map)
    edge_up = edge_img.resize(target_size, resample=Image.Resampling.BICUBIC)
    edge_up_arr = np.asarray(edge_up, dtype=np.float32) / 255.0

    edge_weight = np.power(np.clip(edge_up_arr, 0.0, 1.0), edge_gamma)
    edge_weight = np.clip(edge_weight * edge_strength, 0.0, 1.0)

    original_mode = base.mode
    has_alpha = original_mode == "RGBA"

    alpha = None
    if has_alpha:
        alpha = base.getchannel("A")
        work = base.convert("RGB")
    elif original_mode == "CMYK":
        work = base.convert("RGB")
    elif original_mode != "RGB":
        work = base.convert("RGB")
    else:
        work = base

    work_arr = np.asarray(work, dtype=np.float32)

    sharp_img = work.filter(
        ImageFilter.Kernel(
            size=(3, 3),
            kernel=[
                 0, -1,  0,
                -1,  5, -1,
                 0, -1,  0,
            ],
            scale=None,
            offset=0,
        )
    )
    sharp_arr = np.asarray(sharp_img, dtype=np.float32)

    weight_3 = edge_weight[:, :, None]
    blended = _blend_channel(work_arr, sharp_arr, weight_3)
    out_rgb = Image.fromarray(blended.astype(np.uint8), mode="RGB")

    if original_mode == "CMYK":
        out = out_rgb.convert("CMYK")
    elif has_alpha:
        out = out_rgb.convert("RGBA")
        out.putalpha(alpha)
    else:
        out = out_rgb.convert(original_mode) if original_mode != "RGB" else out_rgb

    return out


def add_bleed_dominant(
    img: Image.Image,
    bleed_px: int,
    custom_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    width, height = img.size
    dominant = custom_color if custom_color is not None else get_dominant_color(img)

    if img.mode == "RGBA":
        fill = dominant + (255,)
    elif img.mode == "CMYK":
        fill = rgb_to_cmyk(dominant)
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

    expanded.paste(
        top.crop((0, 0, bleed_px, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        (0, 0),
    )
    expanded.paste(
        top.crop((width - bleed_px, 0, width, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        (bleed_px + width, 0),
    )
    expanded.paste(
        bottom.crop((0, 0, bleed_px, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        (0, bleed_px + height),
    )
    expanded.paste(
        bottom.crop((width - bleed_px, 0, width, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        (bleed_px + width, bleed_px + height),
    )
    return expanded


def add_bleed_edge_extend_miter(img: Image.Image, bleed_px: int) -> Image.Image:
    width, height = img.size
    if bleed_px <= 0:
        return img

    canvas = Image.new(img.mode, (width + 2 * bleed_px, height + 2 * bleed_px))

    if img.mode == "RGBA":
        canvas.paste(img, (bleed_px, bleed_px), img)
    else:
        canvas.paste(img, (bleed_px, bleed_px))

    sample = max(2, min(bleed_px // 6, 16, width, height))

    top = img.crop((0, 0, width, sample)).resize(
        (width, bleed_px), resample=Image.Resampling.BICUBIC
    )
    bottom = img.crop((0, height - sample, width, height)).resize(
        (width, bleed_px), resample=Image.Resampling.BICUBIC
    )
    left = img.crop((0, 0, sample, height)).resize(
        (bleed_px, height), resample=Image.Resampling.BICUBIC
    )
    right = img.crop((width - sample, 0, width, height)).resize(
        (bleed_px, height), resample=Image.Resampling.BICUBIC
    )

    canvas.paste(top, (bleed_px, 0))
    canvas.paste(bottom, (bleed_px, height + bleed_px))
    canvas.paste(left, (0, bleed_px))
    canvas.paste(right, (width + bleed_px, bleed_px))

    def make_corner_from_edge_runs(
        horiz_strip: Image.Image,
        vert_strip: Image.Image,
        horiz_x: int,
        vert_y: int,
        corner_name: str,
    ) -> Image.Image:
        corner = Image.new(img.mode, (bleed_px, bleed_px))

        horiz_x = max(0, min(horiz_strip.width - 1, horiz_x))
        vert_y = max(0, min(vert_strip.height - 1, vert_y))

        horiz_patch = horiz_strip.crop((horiz_x, 0, horiz_x + 1, bleed_px)).resize(
            (bleed_px, bleed_px), resample=Image.Resampling.BICUBIC
        )

        vert_patch = vert_strip.crop((0, vert_y, bleed_px, vert_y + 1)).resize(
            (bleed_px, bleed_px), resample=Image.Resampling.BICUBIC
        )

        h_pixels = horiz_patch.load()
        v_pixels = vert_patch.load()
        c_pixels = corner.load()

        for y in range(bleed_px):
            for x in range(bleed_px):
                if corner_name == "tl":
                    use_h = y <= x
                elif corner_name == "tr":
                    use_h = y <= (bleed_px - 1 - x)
                elif corner_name == "bl":
                    use_h = y >= (bleed_px - 1 - x)
                else:  # br
                    use_h = y >= x

                c_pixels[x, y] = h_pixels[x, y] if use_h else v_pixels[x, y]

        return corner

    inset = max(
        1,
        min(
            bleed_px,
            sample * 2,
            width // 8 if width >= 8 else 1,
            height // 8 if height >= 8 else 1,
        ),
    )

    top_left_corner = make_corner_from_edge_runs(
        top, left,
        horiz_x=inset,
        vert_y=inset,
        corner_name="tl",
    )
    top_right_corner = make_corner_from_edge_runs(
        top, right,
        horiz_x=width - 1 - inset,
        vert_y=inset,
        corner_name="tr",
    )
    bottom_left_corner = make_corner_from_edge_runs(
        bottom, left,
        horiz_x=inset,
        vert_y=height - 1 - inset,
        corner_name="bl",
    )
    bottom_right_corner = make_corner_from_edge_runs(
        bottom, right,
        horiz_x=width - 1 - inset,
        vert_y=height - 1 - inset,
        corner_name="br",
    )

    canvas.paste(top_left_corner, (0, 0))
    canvas.paste(top_right_corner, (width + bleed_px, 0))
    canvas.paste(bottom_left_corner, (0, height + bleed_px))
    canvas.paste(bottom_right_corner, (width + bleed_px, height + bleed_px))

    return canvas


def add_bleed(
    img: Image.Image,
    bleed_px: int,
    mode: str,
    custom_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    if bleed_px <= 0:
        return img
    if mode == "edge":
        return add_bleed_edge_extend(img, bleed_px)
    if mode == "edge_miter":
        return add_bleed_edge_extend_miter(img, bleed_px)
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

    icc_profile = img.info.get("icc_profile")
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    if ext in (".jpg", ".jpeg"):
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        elif img.mode not in ("RGB", "CMYK"):
            img = img.convert("RGB")

        save_kwargs.update({"quality": quality, "optimize": True, "progressive": True})

    elif ext == ".png":
        if img.mode == "CMYK":
            img = img.convert("RGB")
            save_kwargs.pop("icc_profile", None)
        save_kwargs.update({"optimize": True})

    elif ext == ".webp":
        if img.mode == "CMYK":
            img = img.convert("RGB")
            save_kwargs.pop("icc_profile", None)
        save_kwargs.update({"quality": quality, "method": 6})

    elif ext in (".tif", ".tiff"):
        if img.mode == "RGBA":
            img = img.convert("RGB")

        save_kwargs.update({
            "compression": "tiff_lzw",
            "format": "TIFF",
        })

    img.save(out_path, **save_kwargs)


def process_one_image(src: Path, settings: UpscaleSettings) -> tuple[bool, str]:
    rel = src.relative_to(settings.input_dir)
    out_ext = get_output_extension(src, settings.output_format_mode)
    out_path = settings.output_dir / rel.parent / f"{rel.stem}{settings.suffix}{out_ext}"

    if out_path.exists() and not settings.overwrite:
        return False, f"[SKIP] {src.name} -> already exists"

    bleed_px = int(round(settings.bleed_inches * settings.dpi))
    scale_factor = settings.scale_percent / 100.0

    with Image.open(src) as img:
        img.load()
        img = ensure_mode(img)

        if settings.pad_to_square:
            img = pad_to_square_canvas(img, settings.background_mode, settings.custom_background_color)

        original_width, original_height = img.size
        target_width_px = max(1, int(round(original_width * scale_factor)))
        target_height_px = max(1, int(round(original_height * scale_factor)))
        target_size = (target_width_px, target_height_px)

        if settings.resample == "edge_directed":
            out = resize_edge_directed(
                img,
                target_size,
                two_step=settings.two_step,
                base_resample=Image.Resampling.LANCZOS,
                edge_strength=0.35,
                edge_gamma=0.8,
            )
        else:
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
) -> dict[str, int | float]:
    if not settings.input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {settings.input_dir}")
    if not settings.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {settings.input_dir}")

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_images(settings.input_dir, settings.recursive))
    total = len(files)
    saved = 0
    skipped = 0
    failed = 0

    preview_width_px = 0
    preview_height_px = 0
    final_width_px = 0
    final_height_px = 0
    bleed_px = int(round(settings.bleed_inches * settings.dpi))
    scale_factor = settings.scale_percent / 100.0

    if files:
        with Image.open(files[0]) as preview_img:
            preview_img.load()
            preview_img = ensure_mode(preview_img)
            if settings.pad_to_square:
                preview_img = pad_to_square_canvas(
                    preview_img,
                    settings.background_mode,
                    settings.custom_background_color,
                )

            preview_width_px = max(1, int(round(preview_img.width * scale_factor)))
            preview_height_px = max(1, int(round(preview_img.height * scale_factor)))
            final_width_px = preview_width_px + (2 * bleed_px)
            final_height_px = preview_height_px + (2 * bleed_px)

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
        "scale_percent": settings.scale_percent,
        "art_width_px": preview_width_px,
        "art_height_px": preview_height_px,
        "bleed_px": bleed_px,
        "final_width_px": final_width_px,
        "final_height_px": final_height_px,
    }
