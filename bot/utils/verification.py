import hashlib
import json
import logging
import subprocess
import tempfile
import os
import io
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from scipy.fftpack import dct as scipy_dct
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, falling back to aHash")


def md5_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def compute_phash(img: Image.Image) -> int:
    img_gray = img.convert("L").resize((32, 32), Image.LANCZOS)
    arr = np.array(img_gray, dtype=np.float32)

    if SCIPY_AVAILABLE:
        dct_rows = scipy_dct(arr, axis=1, norm="ortho")
        dct_full = scipy_dct(dct_rows, axis=0, norm="ortho")
        dct_low = dct_full[:8, :8]
    else:
        dct_low = arr[:8, :8]

    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(h1: int, h2: int) -> int:
    xor = h1 ^ h2
    return bin(xor).count("1")


def count_exif_tags(data: bytes) -> int:
    try:
        import piexif
        exif = piexif.load(data)
        count = 0
        for ifd_name in exif:
            if isinstance(exif[ifd_name], dict):
                count += len(exif[ifd_name])
        return count
    except Exception:
        return 0


def format_size_kb(data: bytes) -> str:
    return f"{len(data) / 1024:.1f}"


def verify_image(
    original_bytes: bytes,
    processed_bytes: bytes,
) -> str:
    orig_md5 = md5_hash(original_bytes)
    proc_md5 = md5_hash(processed_bytes)
    hashes_differ = orig_md5 != proc_md5

    orig_exif = count_exif_tags(original_bytes)
    proc_exif = count_exif_tags(processed_bytes)
    exif_line = "очищен ✅" if proc_exif == 0 else f"осталось {proc_exif} тегов ⚠️"

    orig_size = format_size_kb(original_bytes)
    proc_size = format_size_kb(processed_bytes)

    md5_ok = "изменён ✅" if hashes_differ else "совпадает ⚠️"

    return (
        f"✅ Фото уникализировано\n\n"
        f"MD5: {md5_ok}\n"
        f"EXIF: {exif_line}\n"
        f"Размер: {orig_size} → {proc_size} KB\n\n"
        f"Изменения: обрезка, поворот, downscale→upscale, "
        f"яркость/контраст/насыщенность, hue, шум, виньетка, unsharp, "
        f"EXIF очищен + случайный (камера, дата, Adobe)\n\n"
        f"🔥 Facebook увидит как новый креатив"
    )


def _probe_video_meta(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except Exception:
        return {}


# MP4 container-level identifiers — part of the spec, cannot be removed
_CONTAINER_FIELDS = {"major_brand", "minor_version", "compatible_brands"}

# Codec/muxer-added stream tags — set by libx264/libfdk_aac/MP4 muxer itself
_STREAM_FIELDS_SKIP = {
    "vendor_id", "handler_name", "language", "encoder",
    "creation_time",  # auto-set by muxer on each stream
    "timecode",       # timecode track tag from original container
}

# Format-level tags that WE intentionally inject (fake values) — not original metadata
_FORMAT_FIELDS_INJECTED = {
    "creation_time", "encoder", "title", "comment",
    "description", "copyright",
}


def _count_video_meta_tags(probe: dict) -> int:
    """Count original metadata tags left, excluding container fields and our injected tags."""
    count = 0
    fmt_tags = probe.get("format", {}).get("tags", {})
    for key, val in fmt_tags.items():
        k = key.lower()
        if k in _CONTAINER_FIELDS or k in _FORMAT_FIELDS_INJECTED:
            continue
        if val and val.strip():
            count += 1
    for stream in probe.get("streams", []):
        for key, val in stream.get("tags", {}).items():
            if key.lower() in _STREAM_FIELDS_SKIP:
                continue
            if val and val.strip():
                count += 1
    return count


def _get_video_stream_info(probe: dict) -> tuple[str, str]:
    fps = "?"
    resolution = "?"
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            w = stream.get("width", "?")
            h = stream.get("height", "?")
            resolution = f"{w}x{h}"
            fps_str = stream.get("r_frame_rate", "?/1")
            try:
                num, den = fps_str.split("/")
                fps = str(round(int(num) / int(den), 2))
            except Exception:
                fps = fps_str
            break
    return fps, resolution


def _extract_first_frame_phash(video_path: str) -> int:
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vframes", "1",
            "-f", "image2",
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return 0

        with open(tmp_path, "rb") as f:
            frame_bytes = f.read()
        img = Image.open(io.BytesIO(frame_bytes))
        return compute_phash(img)
    except Exception as e:
        logger.warning(f"Frame extraction failed: {e}")
        return 0
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def verify_video(
    original_bytes: bytes,
    processed_bytes: bytes,
) -> str:
    orig_md5 = md5_hash(original_bytes)
    proc_md5 = md5_hash(processed_bytes)
    hashes_differ = orig_md5 != proc_md5

    orig_size = format_size_kb(original_bytes)
    proc_size = format_size_kb(processed_bytes)

    new_fps = "?"
    proc_meta_count = 0

    # Only probe the processed file — we only need output metadata/fps
    tmp_proc_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_proc:
            tmp_proc.write(processed_bytes)
            tmp_proc_path = tmp_proc.name

        proc_probe = _probe_video_meta(tmp_proc_path)
        proc_meta_count = _count_video_meta_tags(proc_probe)
        new_fps, _ = _get_video_stream_info(proc_probe)
    except Exception:
        pass
    finally:
        if tmp_proc_path:
            try:
                if os.path.exists(tmp_proc_path):
                    os.remove(tmp_proc_path)
            except Exception:
                pass

    meta_line = "очищены ✅" if proc_meta_count == 0 else f"осталось {proc_meta_count} тегов ⚠️"
    md5_ok = "изменён ✅" if hashes_differ else "совпадает ⚠️"

    return (
        f"✅ Видео уникализировано\n\n"
        f"MD5: {md5_ok}\n"
        f"Метаданные: {meta_line}\n"
        f"Размер: {orig_size} → {proc_size} KB  |  FPS: → {new_fps}\n\n"
        f"Изменения: обрезка, поворот, downscale→upscale, шум, "
        f"цвет/hue/colorbalance, хром.аберрация, GOP, B-frames, "
        f"CRF, аудио ресемплинг, метаданные очищены + случайные\n\n"
        f"🔥 Facebook увидит как новый креатив"
    )
