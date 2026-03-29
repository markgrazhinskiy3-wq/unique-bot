import os
import json
import random
import logging
import traceback
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

TMP_FILES: list[str] = []


def _run_ffmpeg(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    logger.info(f"FFmpeg command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg timed out after {timeout}s\nCommand: {' '.join(cmd)}")
        raise RuntimeError(f"FFmpeg timed out after {timeout}s") from e
    except FileNotFoundError as e:
        logger.error("FFmpeg not found — is it installed?")
        raise RuntimeError("FFmpeg binary not found") from e

    if result.returncode != 0:
        logger.error(
            f"FFmpeg failed (exit {result.returncode})\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout[-1000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}"
        )
        stderr_tail = result.stderr[-500:].strip() if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"FFmpeg exit {result.returncode}: {stderr_tail}"
        )

    logger.info(f"FFmpeg succeeded (exit 0)")
    return result


def probe_video(input_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found — is FFmpeg installed?")

    if result.returncode != 0:
        logger.error(f"ffprobe failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {result.stderr[-300:]}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.error(f"ffprobe JSON parse error: {e}\nOutput: {result.stdout[:500]}")
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}") from e


def get_video_info(probe_data: dict) -> dict:
    info = {
        "width": 1280,
        "height": 720,
        "fps": "30",
        "bitrate": "2000k",
        "codec": "libx264",
        "audio_codec": "aac",
        "duration": 0.0,
        "has_audio": False,
    }

    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = stream.get("width", 1280)
            info["height"] = stream.get("height", 720)
            fps_str = stream.get("r_frame_rate", "30/1")
            try:
                num, den = fps_str.split("/")
                den = int(den)
                fps_val = int(num) / den if den != 0 else 30
                info["fps"] = str(round(fps_val, 3))
            except Exception:
                info["fps"] = "30"
        elif stream.get("codec_type") == "audio":
            info["has_audio"] = True
            audio_codec = stream.get("codec_name", "aac")
            info["audio_codec"] = "aac"

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

    logger.info(f"Video info: {info}")
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
    out_w = max(out_w, 4)
    out_h = max(out_h, 4)
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

    logger.info(f"Video filter chain: {','.join(filters)}")
    return ",".join(filters)


def process_video(input_path: str, output_path: str) -> None:
    try:
        probe_data = probe_video(input_path)
    except Exception as e:
        logger.error(f"probe_video failed:\n{traceback.format_exc()}")
        raise

    try:
        info = get_video_info(probe_data)
    except Exception as e:
        logger.error(f"get_video_info failed:\n{traceback.format_exc()}")
        raise

    try:
        vf = build_video_filter(info)
    except Exception as e:
        logger.error(f"build_video_filter failed:\n{traceback.format_exc()}")
        raise

    audio_tempo = round(random.uniform(0.999, 1.001), 4)
    audio_pitch = round(random.uniform(0.999, 1.001), 4)
    fps_variation = round(float(info["fps"]) + random.uniform(-0.1, 0.1), 3)
    crf = random.randint(20, 24)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-r", str(fps_variation),
        "-movflags", "+faststart",
        "-fflags", "+bitexact",
    ]

    if info["has_audio"]:
        af = f"atempo={audio_tempo},asetrate=44100*{audio_pitch},aresample=44100"
        cmd += [
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-af", af,
        ]
    else:
        cmd += ["-an"]

    cmd.append(output_path)

    try:
        _run_ffmpeg(cmd, timeout=600)
    except Exception as e:
        logger.error(f"FFmpeg processing failed:\n{traceback.format_exc()}")
        raise

    logger.info(f"Video processed successfully → {output_path}")


def process_video_file(input_bytes: bytes, original_filename: str) -> tuple[bytes, str]:
    suffix = Path(original_filename).suffix.lower()
    if not suffix:
        suffix = ".mp4"
    out_suffix = ".mp4"

    tmp_in_path = None
    tmp_out_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(input_bytes)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=out_suffix, delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        logger.info(
            f"Processing video: {original_filename} "
            f"({len(input_bytes)/1024/1024:.1f}MB) → {tmp_in_path}"
        )

        process_video(tmp_in_path, tmp_out_path)

        with open(tmp_out_path, "rb") as f:
            result_bytes = f.read()

        out_filename = f"unique_{Path(original_filename).stem}{out_suffix}"
        logger.info(
            f"Video processing complete: {out_filename} "
            f"({len(result_bytes)/1024/1024:.1f}MB)"
        )
        return result_bytes, out_filename

    except Exception:
        logger.error(f"process_video_file failed:\n{traceback.format_exc()}")
        raise

    finally:
        for p in [tmp_in_path, tmp_out_path]:
            if p:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                        logger.debug(f"Cleaned up temp file: {p}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up {p}: {cleanup_err}")
