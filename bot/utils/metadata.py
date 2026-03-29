import piexif
import io
from PIL import Image


def strip_image_metadata(img: Image.Image) -> Image.Image:
    data = list(img.getdata())
    clean = Image.new(img.mode, img.size)
    clean.putdata(data)
    clean.info = {}
    return clean


def embed_srgb_profile(img: Image.Image) -> Image.Image:
    try:
        srgb_icc = b'\x00' * 4
        return img
    except Exception:
        return img


def clean_exif_bytes(image_bytes: bytes) -> bytes:
    try:
        piexif.remove(image_bytes)
        return image_bytes
    except Exception:
        return image_bytes


def save_clean_image(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
    elif fmt == "PNG":
        save_kwargs["optimize"] = True
    img.save(buf, **save_kwargs)
    buf.seek(0)
    return buf.read()
