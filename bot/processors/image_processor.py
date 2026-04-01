import io
import random
import logging
import datetime
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


def _sample_pixels(img: Image.Image, label: str) -> None:
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    sample = arr[h // 2, w // 2]
    logger.debug(f"[{label}] size={img.size} center_pixel={sample.tolist()}")


def step1_destroy_metadata(img: Image.Image) -> Image.Image:
    data = list(img.getdata())
    mode = img.mode
    size = img.size
    clean = Image.new(mode, size)
    clean.putdata(data)
    _sample_pixels(clean, "after_meta_strip")
    return clean


def step2_pad_and_asymmetric_crop(img: Image.Image) -> Image.Image:
    _sample_pixels(img, "before_pad_crop")

    arr = np.array(img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img)
    padded = np.pad(arr, ((4, 4), (4, 4), (0, 0)), mode="edge")
    padded_img = Image.fromarray(padded.astype(np.uint8), mode="RGB" if arr.ndim == 3 else img.mode)

    if img.mode == "RGBA":
        alpha = np.array(img.split()[3])
        alpha_pad = np.pad(alpha, ((4, 4), (4, 4)), mode="edge")
        r, g, b = padded_img.split()
        padded_img = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_pad, "L")))

    pw, ph = padded_img.size

    crop_left = random.randint(2, 5)
    crop_right = random.randint(2, 5)
    crop_top = random.randint(2, 5)
    crop_bottom = random.randint(2, 5)

    cropped = padded_img.crop((
        crop_left,
        crop_top,
        pw - crop_right,
        ph - crop_bottom,
    ))

    orig_w, orig_h = img.size
    delta_w = random.randint(2, 4)
    delta_h = random.randint(2, 4)
    sign_w = random.choice([-1, 1])
    sign_h = random.choice([-1, 1])
    new_w = max(orig_w + sign_w * delta_w, 4)
    new_h = max(orig_h + sign_h * delta_h, 4)
    if new_w % 2 != 0:
        new_w += 1
    if new_h % 2 != 0:
        new_h += 1

    result = cropped.resize((new_w, new_h), Image.LANCZOS)

    logger.info(f"[pad_crop] orig={img.size} crop=({crop_left},{crop_top},{crop_right},{crop_bottom}) new={result.size}")
    _sample_pixels(result, "after_pad_crop")
    return result


def step3_rotation(img: Image.Image) -> Image.Image:
    _sample_pixels(img, "before_rotation")

    angle = random.uniform(0.4, 0.8) * random.choice([-1, 1])
    w, h = img.size

    if img.mode == "RGBA":
        rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    else:
        bg = (255, 255, 255) if img.mode == "RGB" else 255
        rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=bg)

    rw, rh = rotated.size
    border_w = (rw - w) // 2 + 1
    border_h = (rh - h) // 2 + 1
    border_w = max(border_w, 0)
    border_h = max(border_h, 0)

    result = rotated.crop((
        border_w,
        border_h,
        rw - border_w,
        rh - border_h,
    ))
    result = result.resize((w, h), Image.LANCZOS)

    logger.info(f"[rotation] angle={angle:.3f}° orig={img.size} result={result.size}")
    _sample_pixels(result, "after_rotation")
    return result


def step4_zoom_crop(img: Image.Image) -> Image.Image:
    _sample_pixels(img, "before_zoom")

    w, h = img.size
    pct = random.uniform(0.02, 0.04)
    left = int(w * pct)
    top = int(h * pct)
    right = w - int(w * pct)
    bottom = h - int(h * pct)

    left = max(left, 1)
    top = max(top, 1)
    right = min(right, w - 1)
    bottom = min(bottom, h - 1)

    cropped = img.crop((left, top, right, bottom))
    result = cropped.resize((w, h), Image.LANCZOS)

    logger.info(f"[zoom] pct={pct:.3f} crop=({left},{top},{right},{bottom}) result={result.size}")
    _sample_pixels(result, "after_zoom")
    return result


def step5_color_shifts(img: Image.Image) -> Image.Image:
    _sample_pixels(img, "before_color_shifts")

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # Slightly darken (-2 to -5%), increase contrast (4-8%), desaturate (88-96%)
    b_factor = 1.0 + random.uniform(-5, -2) / 100.0
    c_factor = random.uniform(1.04, 1.08)
    s_factor = random.uniform(0.88, 0.96)

    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = ImageEnhance.Color(rgb).enhance(s_factor)
        r2, g2, b2 = rgb.split()
        img = Image.merge("RGBA", (r2, g2, b2, a))
    else:
        img = ImageEnhance.Color(img).enhance(s_factor)

    hue_shift = random.uniform(3, 6) * random.choice([-1, 1])
    try:
        img_hsv = img.convert("RGB") if img.mode == "RGBA" else img
        arr = np.array(img_hsv, dtype=np.float32) / 255.0
        hue_rad = hue_shift * np.pi / 180.0
        cos_h = np.cos(hue_rad)
        sin_h = np.sin(hue_rad)
        m = np.array([
            [cos_h + (1 - cos_h) / 3, (1 - cos_h) / 3 - np.sqrt(1/3) * sin_h, (1 - cos_h) / 3 + np.sqrt(1/3) * sin_h],
            [(1 - cos_h) / 3 + np.sqrt(1/3) * sin_h, cos_h + (1 - cos_h) / 3, (1 - cos_h) / 3 - np.sqrt(1/3) * sin_h],
            [(1 - cos_h) / 3 - np.sqrt(1/3) * sin_h, (1 - cos_h) / 3 + np.sqrt(1/3) * sin_h, cos_h + (1 - cos_h) / 3],
        ])
        arr = arr @ m.T
        arr = np.clip(arr, 0, 1)
        hue_shifted = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
        if has_alpha:
            _, _, _, a = img.split()
            r, g, b = hue_shifted.split()
            img = Image.merge("RGBA", (r, g, b, a))
        else:
            img = hue_shifted
    except Exception as e:
        logger.warning(f"Hue shift failed: {e}")

    logger.info(f"[color_shifts] brightness_delta={b_factor:.4f} contrast={c_factor:.4f} sat={s_factor:.4f} hue={hue_shift:.2f}°")
    _sample_pixels(img, "after_color_shifts")
    return img


def step6_gaussian_noise(img: Image.Image, fmt: str = "JPEG") -> Image.Image:
    _sample_pixels(img, "before_noise")

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = img

    arr = np.array(rgb, dtype=np.float32)
    # PNG is lossless — high noise destroys compression and inflates file 3-5x.
    # For PNG use tiny sigma; geometric transforms already ensure pHash distance.
    if fmt == "PNG":
        sigma = random.uniform(0.5, 1.2)
    else:
        sigma = random.uniform(3.0, 6.0)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    rgb_out = Image.fromarray(arr, mode="RGB")

    if has_alpha:
        r2, g2, b2 = rgb_out.split()
        img = Image.merge("RGBA", (r2, g2, b2, a))
    else:
        img = rgb_out

    logger.info(f"[noise] sigma={sigma:.2f} fmt={fmt}")
    _sample_pixels(img, "after_noise")
    return img


def step_chroma_shift(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
    else:
        r, g, b = img.split()

    channel_name = random.choice(["r", "g", "b"])
    direction = random.choice([0, 1])
    shift = random.choice([-1, 1])

    channels = {"r": np.array(r), "g": np.array(g), "b": np.array(b)}
    channels[channel_name] = np.roll(channels[channel_name], shift, axis=direction)

    merged = (
        Image.merge("RGBA", (
            Image.fromarray(channels["r"], "L"),
            Image.fromarray(channels["g"], "L"),
            Image.fromarray(channels["b"], "L"),
            a,
        )) if has_alpha else
        Image.merge("RGB", (
            Image.fromarray(channels["r"], "L"),
            Image.fromarray(channels["g"], "L"),
            Image.fromarray(channels["b"], "L"),
        ))
    )

    logger.info(f"[chroma_shift] channel={channel_name} axis={direction} shift={shift}px")
    return merged


def step_downscale_upscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = random.uniform(0.90, 0.95)
    ds_w = max(int(w * scale) & ~1, 4)
    ds_h = max(int(h * scale) & ~1, 4)
    downscaled = img.resize((ds_w, ds_h), Image.BICUBIC)
    result = downscaled.resize((w, h), Image.LANCZOS)
    logger.info(f"[downscale_upscale] orig={w}x{h} temp={ds_w}x{ds_h} scale={scale:.3f}")
    return result


def step_vignette(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    has_alpha = img.mode == "RGBA"
    w, h = img.size
    arr = np.array(img, dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.ogrid[:h, :w]
    dist_norm = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2) / np.sqrt(2)
    strength = random.uniform(0.02, 0.04)
    mask = np.clip(1.0 - strength * (dist_norm ** 2), 0, 1).astype(np.float32)
    if has_alpha:
        arr[:, :, :3] = np.clip(arr[:, :, :3] * mask[:, :, np.newaxis], 0, 255)
    else:
        arr = np.clip(arr * mask[:, :, np.newaxis], 0, 255)
    result = Image.fromarray(arr.astype(np.uint8), mode=img.mode)
    logger.info(f"[vignette] strength={strength:.3f}")
    return result


def step_unsharp_mask(img: Image.Image) -> Image.Image:
    radius = round(random.uniform(0.3, 0.6), 2)
    percent = random.randint(80, 130)
    result = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=0))
    logger.info(f"[unsharp_mask] radius={radius} percent={percent}")
    return result


def _inject_fake_exif(jpeg_bytes: bytes) -> bytes:
    try:
        import piexif
        cameras = [
            ("Canon", "Canon EOS R6"),
            ("Nikon", "NIKON Z 6_2"),
            ("Sony", "ILCE-7M4"),
            ("Apple", "iPhone 15 Pro"),
            ("Samsung", "SM-S918B"),
        ]
        make, model = random.choice(cameras)
        days_ago = random.randint(1, 365)
        rand_dt = datetime.datetime.now() - datetime.timedelta(
            days=days_ago,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        dt_str = rand_dt.strftime("%Y:%m:%d %H:%M:%S").encode()
        software_options = [
            b"Adobe Photoshop 25.5.0",
            b"Adobe Photoshop 26.1.0",
            b"Adobe Lightroom 13.3",
            b"Adobe Lightroom Classic 13.4",
        ]
        software = random.choice(software_options)
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: make.encode(),
                piexif.ImageIFD.Model: model.encode(),
                piexif.ImageIFD.Software: software,
                piexif.ImageIFD.DateTime: dt_str,
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: dt_str,
                piexif.ExifIFD.DateTimeDigitized: dt_str,
            },
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        exif_bytes = piexif.dump(exif_dict)
        result = piexif.insert(exif_bytes, jpeg_bytes)
        logger.info(f"[fake_exif] make={make} model={model} software={software.decode()} date={dt_str.decode()}")
        return result
    except Exception as e:
        logger.warning(f"[fake_exif] Failed: {e}")
        return jpeg_bytes


def _save_jpeg(img: Image.Image, quality: int, subsampling: int = 0, progressive: bool = False) -> bytes:
    buf = io.BytesIO()
    img_save = img.convert("RGB") if img.mode == "RGBA" else img
    kwargs: dict = {"format": "JPEG", "quality": quality, "subsampling": subsampling}
    if progressive:
        kwargs["progressive"] = True
    img_save.save(buf, **kwargs)
    return buf.getvalue()


def _find_jpeg_quality_for_size(
    img: Image.Image,
    target_bytes: int,
    subsampling: int = 0,
    progressive: bool = False,
) -> int:
    lo, hi = 5, 95
    best_quality = 85
    best_diff = float("inf")

    for _ in range(10):
        mid = (lo + hi) // 2
        data = _save_jpeg(img, mid, subsampling=subsampling, progressive=progressive)
        size = len(data)
        diff = abs(size - target_bytes)
        if diff < best_diff:
            best_diff = diff
            best_quality = mid
        if size < target_bytes:
            lo = mid + 1
        elif size > target_bytes:
            hi = mid - 1
        else:
            break
        if lo > hi:
            break

    return best_quality


def _save_png(img: Image.Image, compress_level: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=compress_level, optimize=True)
    return buf.getvalue()


def _find_png_compress_for_size(img: Image.Image, target_bytes: int) -> int:
    best_level = 6
    best_diff = float("inf")
    for level in range(0, 10):
        data = _save_png(img, level)
        diff = abs(len(data) - target_bytes)
        if diff < best_diff:
            best_diff = diff
            best_level = level
    return best_level


def step7_reencode(img: Image.Image, fmt: str) -> Image.Image:
    if fmt != "JPEG":
        return img

    img_save = img.convert("RGB") if img.mode == "RGBA" else img
    quality = random.randint(82, 88)
    data = _save_jpeg(img_save, quality)
    buf = io.BytesIO(data)
    result = Image.open(buf).copy()
    logger.info(f"[reencode] quality={quality} size={len(data)/1024:.1f}KB")
    _sample_pixels(result, "after_reencode")
    return result


def process_image(image_bytes: bytes, original_filename: str) -> tuple[bytes, str]:
    original_file_size = len(image_bytes)

    buf = io.BytesIO(image_bytes)
    img = Image.open(buf)
    original_format = img.format
    original_size = img.size

    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    fmt = _detect_format(img, original_format)

    logger.info(
        f"Processing image: {original_filename}, format={fmt}, "
        f"size={original_size}, mode={img.mode}, file_size={original_file_size/1024:.1f}KB"
    )
    _sample_pixels(img, "original")

    img = step1_destroy_metadata(img)
    img = step2_pad_and_asymmetric_crop(img)
    img = step3_rotation(img)
    img = step4_zoom_crop(img)
    img = step_downscale_upscale(img)
    img = step5_color_shifts(img)
    img = step6_gaussian_noise(img, fmt)
    img = step_chroma_shift(img)
    img = step_vignette(img)
    img = step_unsharp_mask(img)
    img = step7_reencode(img, fmt)

    img_out = img
    if fmt == "JPEG" and img.mode == "RGBA":
        img_out = img.convert("RGB")

    # Random JPEG subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0 — changes DCT block structure
    jpeg_subsampling = random.choice([0, 1, 2])
    # Random progressive encoding — changes file byte structure
    jpeg_progressive = random.choice([True, False])
    logger.info(f"[jpeg] subsampling={jpeg_subsampling} progressive={jpeg_progressive}")

    quality = 85
    if fmt == "JPEG":
        quality = _find_jpeg_quality_for_size(
            img_out, original_file_size,
            subsampling=jpeg_subsampling, progressive=jpeg_progressive,
        )
        jitter = random.randint(-2, 2)
        quality = max(40, min(95, quality + jitter))
        out_data = _save_jpeg(img_out, quality, subsampling=jpeg_subsampling, progressive=jpeg_progressive)
        # Inject fake EXIF (camera brand, model, software, random shoot date)
        out_data = _inject_fake_exif(out_data)
    elif fmt == "PNG":
        img_out = img_out.filter(ImageFilter.GaussianBlur(radius=0.4))
        compress_level = _find_png_compress_for_size(img_out, original_file_size)
        out_data = _save_png(img_out, compress_level)
    else:
        out_buf = io.BytesIO()
        img_out.save(out_buf, format=fmt)
        out_data = out_buf.getvalue()

    ratio = len(out_data) / original_file_size
    logger.info(
        f"Image processing complete: {original_filename}, "
        f"orig={original_file_size/1024:.1f}KB out={len(out_data)/1024:.1f}KB "
        f"ratio={ratio:.2f} quality={quality if fmt == 'JPEG' else 'N/A'}"
    )

    ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    out_ext = ext_map.get(fmt, ".jpg")
    out_filename = f"unique_{original_filename.rsplit('.', 1)[0]}{out_ext}"

    return out_data, out_filename
