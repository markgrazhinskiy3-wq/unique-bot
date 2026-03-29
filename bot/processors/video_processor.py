import os
import json
import random
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def probe_video(input_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def get_video_info(probe_data: dict) -> dict:
    info = {
        "width": 1280,
        "height": 720,
        "fps": "30",
        "bitrate": "2000k",
        "codec": "libx264",
        "audio_codec": "aac",
        "duration": 0.0,
    }

    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = stream.get("width", 1280)
            info["height"] = stream.get("height", 720)
            fps_str = stream.get("r_frame_rate", "30/1")
            try:
                num, den = fps_str.split("/")
                info["fps"] = str(round(int(num) / int(den), 3))
            except Exception:
                info["fps"] = "30"
            codec = stream.get("codec_name", "h264")
            info["codec"] = "libx264" if codec in ("h264", "avc") else "libx264"
        elif stream.get("codec_type") == "audio":
            audio_codec = stream.get("codec_name", "aac")
            info["audio_codec"] = "aac" if audio_codec in ("aac", "mp3", "opus") else "aac"

    fmt = probe_data.get("format", {})
    try:
        info["duration"] = float(fmt.get("duration", 0))
    except Exception:
        info["duration"] = 0.0

    bit_rate = fmt.get("bit_rate")
    if bit_rate:
        try:
            kbps = int(bit_rate) // 1000
            info["bitrate"] = f"{kbps}k"
        except Exception:
            pass

    return info


def build_video_filter(info: dict) -> str:
    w = info["width"]
    h = info["height"]

    crop_top = random.randint(2, 4)
    crop_bottom = random.randint(2, 4)
    crop_left = random.randint(2, 4)
    crop_right = random.randint(2, 4)

    cropped_w = w - crop_left - crop_right
    cropped_h = h - crop_top - crop_bottom

    out_w = w + random.choice([-2, -1, 0, 1, 2])
    out_h = h + random.choice([-2, -1, 0, 1, 2])
    out_w = max(out_w, 2)
    out_h = max(out_h, 2)
    if out_w % 2 != 0:
        out_w += 1
    if out_h % 2 != 0:
        out_h += 1

    noise_strength = random.randint(2, 4)
    brightness = round(random.uniform(-0.005, 0.005), 4)
    contrast = round(random.uniform(0.995, 1.005), 4)
    saturation = round(random.uniform(0.995, 1.005), 4)
    gamma = round(random.uniform(0.998, 1.002), 4)

    hue_h = round(random.uniform(-0.5, 0.5), 3)
    hue_s = round(random.uniform(0.997, 1.003), 4)

    filters = [
        f"crop={cropped_w}:{cropped_h}:{crop_left}:{crop_top}",
        f"scale={out_w}:{out_h}:flags=lanczos",
        f"noise=c0s={noise_strength}:c0f=t",
        f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}",
        f"hue=h={hue_h}:s={hue_s}",
    ]

    return ",".join(filters)


def process_video(input_path: str, output_path: str) -> None:
    probe_data = probe_video(input_path)
    info = get_video_info(probe_data)

    vf = build_video_filter(info)

    audio_tempo = round(random.uniform(0.999, 1.001), 4)
    audio_pitch = round(random.uniform(0.999, 1.001), 4)

    fps_variation = round(float(info["fps"]) + random.uniform(-0.1, 0.1), 3)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(random.randint(20, 24)),
        "-r", str(fps_variation),
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-af", f"atempo={audio_tempo},asetrate=44100*{audio_pitch},aresample=44100",
        "-movflags", "+faststart",
        "-fflags", "+bitexact",
        output_path
    ]

    logger.info(f"Running FFmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error(f"FFmpeg stderr: {result.stderr[-2000:]}")
        raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

    logger.info(f"Video processed successfully: {output_path}")


def process_video_file(input_bytes: bytes, original_filename: str) -> tuple[bytes, str]:
    suffix = Path(original_filename).suffix.lower() or ".mp4"
    out_suffix = ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(input_bytes)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix=out_suffix, delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        process_video(tmp_in_path, tmp_out_path)
        with open(tmp_out_path, "rb") as f:
            result_bytes = f.read()

        out_filename = f"unique_{Path(original_filename).stem}{out_suffix}"
        return result_bytes, out_filename
    finally:
        for p in [tmp_in_path, tmp_out_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
