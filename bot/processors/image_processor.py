import io
import random
import logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


def _detect_format(img: Image.Image, original_format: str | None) -> str:
    fmt = (original_format or "JPEG").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in {"JPEG", "PNG", "WEBP"}:
        fmt = "JPEG"
    return fmt


def step1_destroy_metadata(img: Image.Image) -> Image.Image:
    data = list(img.getdata())
    mode = img.mode
    size = img.size
    clean = Image.new(mode, size)
    clean.putdata(data)
    return clean


def step2_pixel_grid_shift(img: Image.Image) -> Image.Image:
    w, h = img.size
    border = 2
    img_with_border = Image.new(img.mode, (w + border * 2, h + border * 2))

    if img.mode in ("RGB", "RGBA"):
        arr = np.array(img)
        top_row = arr[0:1, :, :]
        bottom_row = arr[-1:, :, :]
        left_col = arr[:, 0:1, :]
        right_col = arr[:, -1:, :]

        top_pad = np.tile(top_row, (border, 1, 1))
        bottom_pad = np.tile(bottom_row, (border, 1, 1))
        left_pad = np.tile(left_col, (1, border, 1))
        right_pad = np.tile(right_col, (1, border, 1))

        padded = np.pad(arr, ((border, border), (border, border), (0, 0)), mode='edge')
        img_with_border = Image.fromarray(padded.astype(np.uint8), mode=img.mode)
    else:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)
        padded = np.pad(arr, ((border, border), (border, border), (0, 0)), mode='edge')
        img_with_border = Image.fromarray(padded.astype(np.uint8), mode="RGB")
        if img.mode != "RGB":
            img_with_border = img_with_border.convert(img.mode)

    crop_top = random.randint(1, 3)
    crop_bottom = random.randint(1, 3)
    crop_left = random.randint(1, 3)
    crop_right = random.randint(1, 3)

    bw, bh = img_with_border.size
    img_cropped = img_with_border.crop((
        crop_left, crop_top,
        bw - crop_right, bh - crop_bottom
    ))

    delta_w = random.randint(-2, 2)
    delta_h = random.randint(-2, 2)
    new_w = max(w + delta_w, 2)
    new_h = max(h + delta_h, 2)
    img_resized = img_cropped.resize((new_w, new_h), Image.LANCZOS)

    return img_resized


def step3_subpixel_noise(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    arr = np.array(img, dtype=np.float32)
    noise = np.random.uniform(-1.5, 1.5, arr.shape).astype(np.float32)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def step4_color_space_micro_shift(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
    else:
        r, g, b = img.split()

    def shift_channel(ch):
        arr = np.array(ch, dtype=np.float32)
        delta = random.uniform(-1.0, 1.0)
        arr = arr + delta
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    r = shift_channel(r)
    g = shift_channel(g)
    b = shift_channel(b)

    if has_alpha:
        return Image.merge("RGBA", (r, g, b, a))
    return Image.merge("RGB", (r, g, b))


def step5_jpeg_ghost(img: Image.Image, fmt: str) -> Image.Image:
    if fmt == "JPEG":
        buf = io.BytesIO()
        quality = random.randint(92, 97)
        img_save = img
        if img.mode == "RGBA":
            img_save = img.convert("RGB")
        img_save.save(buf, format="JPEG", quality=quality, subsampling=0)
        buf.seek(0)
        return Image.open(buf).copy()
    return img


def step6_micro_rotation(img: Image.Image) -> Image.Image:
    angle = random.uniform(-0.3, 0.3)
    if img.mode == "RGBA":
        rotated = img.rotate(angle, expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    else:
        bg_color = (255, 255, 255) if img.mode == "RGB" else 255
        rotated = img.rotate(angle, expand=False, resample=Image.BICUBIC, fillcolor=bg_color)
    return rotated


def step7_sharpness_micro_adjust(img: Image.Image) -> Image.Image:
    factor = random.uniform(0.97, 1.03)
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def step8_brightness_contrast(img: Image.Image) -> Image.Image:
    b_factor = random.uniform(0.997, 1.003)
    c_factor = random.uniform(0.997, 1.003)
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    return img


def step9_final_scale(img: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    target_w = original_size[0] + random.randint(-1, 1)
    target_h = original_size[1] + random.randint(-1, 1)
    target_w = max(target_w, 2)
    target_h = max(target_h, 2)
    return img.resize((target_w, target_h), Image.LANCZOS)


def process_image(image_bytes: bytes, original_filename: str) -> tuple[bytes, str]:
    buf = io.BytesIO(image_bytes)
    img = Image.open(buf)
    original_format = img.format
    original_size = img.size

    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    fmt = _detect_format(img, original_format)

    logger.info(f"Processing image: {original_filename}, format={fmt}, size={original_size}, mode={img.mode}")

    img = step1_destroy_metadata(img)
    img = step2_pixel_grid_shift(img)
    img = step3_subpixel_noise(img)
    img = step4_color_space_micro_shift(img)
    img = step5_jpeg_ghost(img, fmt)
    img = step6_micro_rotation(img)
    img = step7_sharpness_micro_adjust(img)
    img = step8_brightness_contrast(img)
    img = step9_final_scale(img, original_size)

    out_buf = io.BytesIO()
    img_out = img
    if fmt == "JPEG" and img.mode == "RGBA":
        img_out = img.convert("RGB")

    save_kwargs: dict = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = random.randint(93, 96)
        save_kwargs["subsampling"] = 0
    elif fmt == "PNG":
        save_kwargs["optimize"] = True

    img_out.save(out_buf, **save_kwargs)
    out_buf.seek(0)

    ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    out_ext = ext_map.get(fmt, ".jpg")
    out_filename = f"unique_{original_filename.rsplit('.', 1)[0]}{out_ext}"

    logger.info(f"Image processed successfully: {out_filename}")
    return out_buf.read(), out_filename
