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
    resample: str = "v3_progressive_edge"
    two_step: bool = True
    sharpen_enabled: bool = True
    unsharp_radius: float = 0.7
    unsharp_percent: int = 120
    unsharp_threshold: int = 2
    pad_to_square: bool = True
    background_mode: str = "auto"  # auto, transparent, white, black, custom
    custom_background_color: tuple[int, int, int] | None = None
    output_format_mode: str = "same"  # same, png, jpg, webp, tif
    suffix: str = "_print_v3"
    recursive: bool = False
    overwrite: bool = False
    quality: int = 92


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def ensure_mode(img: Image.Image) -> Image.Image:
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


# ----------------------------
# NUMPY / IMAGE MATH
# ----------------------------

def pil_rgb_to_float01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return arr / 255.0


def float01_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGB")


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)


def convolve1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")

    out = np.zeros_like(arr, dtype=np.float32)
    for i, w in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + arr.shape[axis])
        out += w * padded[tuple(sl)]
    return out


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr.astype(np.float32, copy=True)
    kernel = gaussian_kernel1d(sigma)
    tmp = convolve1d_reflect(arr.astype(np.float32), kernel, axis=1)
    out = convolve1d_reflect(tmp, kernel, axis=0)
    return out


def compute_structure_tensor(
    luma: np.ndarray,
    sigma_d: float = 0.9,
    sigma_i: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = gaussian_blur(luma, sigma_d)

    lx = np.zeros_like(base, dtype=np.float32)
    ly = np.zeros_like(base, dtype=np.float32)

    lx[:, 1:-1] = (base[:, 2:] - base[:, :-2]) * 0.5
    ly[1:-1, :] = (base[2:, :] - base[:-2, :]) * 0.5

    jxx = gaussian_blur(lx * lx, sigma_i)
    jxy = gaussian_blur(lx * ly, sigma_i)
    jyy = gaussian_blur(ly * ly, sigma_i)

    return jxx, jxy, jyy


def tensor_to_coherence_angle(
    jxx: np.ndarray,
    jxy: np.ndarray,
    jyy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    lambda1 = 0.5 * (jxx + jyy + d)
    lambda2 = 0.5 * (jxx + jyy - d)

    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)) ** 2
    coherence = np.clip(coherence, 0.0, 1.0)

    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-12)
    return coherence, angle


def resize_np_channel(channel: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    ch_img = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8), mode="L")
    out = ch_img.resize(size, resample=resample)
    return np.asarray(out, dtype=np.float32) / 255.0


def image_sharp_kernel_rgb(arr_rgb: np.ndarray) -> np.ndarray:
    img = float01_to_pil_rgb(arr_rgb)
    sharp = img.filter(
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
    return pil_rgb_to_float01(sharp)


def sobel_like_gradients(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(luma, dtype=np.float32)
    gy = np.zeros_like(luma, dtype=np.float32)

    gx[:, 1:-1] = (luma[:, 2:] - luma[:, :-2]) * 0.5
    gy[1:-1, :] = (luma[2:, :] - luma[:-2, :]) * 0.5
    return gx, gy


def laplacian_map(luma: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(luma, dtype=np.float32)
    lap[1:-1, 1:-1] = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
    )
    return lap


def build_masks(
    luma: np.ndarray,
    coherence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx, gy = sobel_like_gradients(luma)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()

    edge_mask = np.clip((coherence ** 0.70) * 1.15, 0.0, 1.0)
    edge_mask = np.clip(edge_mask * (0.35 + 0.65 * grad_mag), 0.0, 1.0)

    lap = np.abs(laplacian_map(luma))
    if lap.max() > 0:
        lap = lap / lap.max()

    curvature_mask = np.clip(1.0 - lap, 0.0, 1.0)
    straight_mask = np.clip(edge_mask * (0.55 + 0.45 * curvature_mask), 0.0, 1.0)
    ridge_mask = np.clip((grad_mag ** 0.9) * (0.5 + 0.5 * edge_mask), 0.0, 1.0)

    return edge_mask, straight_mask, ridge_mask


def preserve_luma(rgb: np.ndarray, target_luma: np.ndarray) -> np.ndarray:
    current_luma = rgb_to_luma(rgb)
    scale = target_luma / np.maximum(current_luma, 1e-6)
    scale = np.clip(scale, 0.70, 1.45)
    return np.clip(rgb * scale[..., None], 0.0, 1.0)


def progressive_pass_size(
    current_size: tuple[int, int],
    target_size: tuple[int, int],
    pass_factor: float = 1.40,
) -> tuple[int, int]:
    cw, ch = current_size
    tw, th = target_size

    nw = min(tw, max(cw + 1, int(round(cw * pass_factor))))
    nh = min(th, max(ch + 1, int(round(ch * pass_factor))))
    return nw, nh


def upscale_rgb_progressive_v3(
    rgb_src_img: Image.Image,
    target_size: tuple[int, int],
    sharpen_enabled: bool,
    unsharp_radius: float,
    unsharp_percent: int,
    unsharp_threshold: int,
) -> Image.Image:
    current_img = rgb_src_img.convert("RGB")

    if current_img.size == target_size:
        return current_img.copy()

    while current_img.size != target_size:
        next_size = progressive_pass_size(current_img.size, target_size, pass_factor=1.40)

        base_up = current_img.resize(next_size, resample=Image.Resampling.LANCZOS)
        base_rgb = pil_rgb_to_float01(base_up)

        cur_rgb = pil_rgb_to_float01(current_img)
        cur_luma = rgb_to_luma(cur_rgb)

        jxx, jxy, jyy = compute_structure_tensor(cur_luma, sigma_d=0.9, sigma_i=1.8)
        coherence, _angle = tensor_to_coherence_angle(jxx, jxy, jyy)

        edge_mask_src, straight_mask_src, ridge_mask_src = build_masks(cur_luma, coherence)

        edge_mask = resize_np_channel(edge_mask_src, next_size, Image.Resampling.BICUBIC)
        straight_mask = resize_np_channel(straight_mask_src, next_size, Image.Resampling.BICUBIC)
        ridge_mask = resize_np_channel(ridge_mask_src, next_size, Image.Resampling.BICUBIC)

        # edge-guided structural crisping
        kernel_sharp_rgb = image_sharp_kernel_rgb(base_rgb)
        mixed_rgb = np.clip(
            base_rgb * (1.0 - edge_mask[..., None]) + kernel_sharp_rgb * edge_mask[..., None],
            0.0,
            1.0,
        )

        # lightness-only adaptive sharpening
        mixed_luma = rgb_to_luma(mixed_rgb)
        luma_img = Image.fromarray(np.clip(mixed_luma * 255.0, 0, 255).astype(np.uint8), mode="L")

        if sharpen_enabled:
            sharp_luma_img = luma_img.filter(
                ImageFilter.UnsharpMask(
                    radius=unsharp_radius,
                    percent=unsharp_percent,
                    threshold=unsharp_threshold,
                )
            )
            sharp_luma = np.asarray(sharp_luma_img, dtype=np.float32) / 255.0
        else:
            sharp_luma = mixed_luma.copy()

        target_luma = np.clip(
            mixed_luma * (1.0 - straight_mask) + sharp_luma * straight_mask,
            0.0,
            1.0,
        )
        mixed_rgb = preserve_luma(mixed_rgb, target_luma)

        # ridge softening to reduce swollen halo look on embossed edges
        blur_rgb = np.empty_like(mixed_rgb, dtype=np.float32)
        for c in range(3):
            blur_rgb[..., c] = gaussian_blur(mixed_rgb[..., c], sigma=0.55)

        ridge_reduce = np.clip(ridge_mask * (1.0 - straight_mask), 0.0, 0.65)
        mixed_rgb = np.clip(
            mixed_rgb * (1.0 - ridge_reduce[..., None]) + blur_rgb * ridge_reduce[..., None],
            0.0,
            1.0,
        )

        # very mild micro-contrast restore on stable straight edges
        final_luma = rgb_to_luma(mixed_rgb)
        stable_boost = np.clip(straight_mask * 0.10, 0.0, 0.10)
        boosted_luma = np.clip(final_luma + stable_boost * (final_luma - gaussian_blur(final_luma, 0.8)), 0.0, 1.0)
        mixed_rgb = preserve_luma(mixed_rgb, boosted_luma)

        current_img = float01_to_pil_rgb(mixed_rgb)

    return current_img


# ----------------------------
# BLEED TOOLS
# ----------------------------

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

    expanded.paste(top.crop((0, 0, bleed_px, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (0, 0))
    expanded.paste(top.crop((width - bleed_px, 0, width, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (bleed_px + width, 0))
    expanded.paste(bottom.crop((0, 0, bleed_px, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (0, bleed_px + height))
    expanded.paste(bottom.crop((width - bleed_px, 0, width, bleed_px)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (bleed_px + width, bleed_px + height))
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

    top = img.crop((0, 0, width, sample)).resize((width, bleed_px), resample=Image.Resampling.BICUBIC)
    bottom = img.crop((0, height - sample, width, height)).resize((width, bleed_px), resample=Image.Resampling.BICUBIC)
    left = img.crop((0, 0, sample, height)).resize((bleed_px, height), resample=Image.Resampling.BICUBIC)
    right = img.crop((width - sample, 0, width, height)).resize((bleed_px, height), resample=Image.Resampling.BICUBIC)

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
                else:
                    use_h = y >= x
                c_pixels[x, y] = h_pixels[x, y] if use_h else v_pixels[x, y]

        return corner

    inset = max(1, min(bleed_px, sample * 2, width // 8 if width >= 8 else 1, height // 8 if height >= 8 else 1))

    canvas.paste(make_corner_from_edge_runs(top, left, inset, inset, "tl"), (0, 0))
    canvas.paste(make_corner_from_edge_runs(top, right, width - 1 - inset, inset, "tr"), (width + bleed_px, 0))
    canvas.paste(make_corner_from_edge_runs(bottom, left, inset, height - 1 - inset, "bl"), (0, height + bleed_px))
    canvas.paste(make_corner_from_edge_runs(bottom, right, width - 1 - inset, height - 1 - inset, "br"), (width + bleed_px, height + bleed_px))

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


# ----------------------------
# OUTPUT
# ----------------------------

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


# ----------------------------
# MAIN PROCESSING
# ----------------------------

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
        icc_profile = img.info.get("icc_profile")
        original_mode = img.mode

        img = ensure_mode(img)

        if settings.pad_to_square:
            img = pad_to_square_canvas(img, settings.background_mode, settings.custom_background_color)

        original_width, original_height = img.size
        target_width_px = max(1, int(round(original_width * scale_factor)))
        target_height_px = max(1, int(round(original_height * scale_factor)))
        target_size = (target_width_px, target_height_px)

        work_rgb = img.convert("RGB")
        out_rgb = upscale_rgb_progressive_v3(
            work_rgb,
            target_size,
            sharpen_enabled=settings.sharpen_enabled,
            unsharp_radius=settings.unsharp_radius,
            unsharp_percent=settings.unsharp_percent,
            unsharp_threshold=settings.unsharp_threshold,
        )

        if original_mode == "CMYK":
            out = out_rgb.convert("CMYK")
        elif original_mode == "RGBA":
            out = out_rgb.convert("RGBA")
        else:
            out = out_rgb

        if icc_profile:
            out.info["icc_profile"] = icc_profile

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
