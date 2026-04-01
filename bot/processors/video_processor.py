import os
import json
import random
import logging
import traceback
import subprocess
import tempfile
import datetime
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
        "bitrate_kbps": 0,
        "codec": "libx264",
        "audio_codec": "aac",
        "audio_sample_rate": 44100,
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
            info["audio_codec"] = "aac"
            try:
                info["audio_sample_rate"] = int(stream.get("sample_rate", 44100))
            except Exception:
                info["audio_sample_rate"] = 44100

    fmt = probe_data.get("format", {})
    try:
        info["duration"] = float(fmt.get("duration", 0))
    except Exception:
        info["duration"] = 0.0

    bit_rate = fmt.get("bit_rate")
    if bit_rate:
        try:
            info["bitrate_kbps"] = max(int(bit_rate) // 1000, 100)
        except Exception:
            pass

    logger.info(f"Video info: {info}")
    return info


def build_video_filter(info: dict) -> str:
    w = info["width"]
    h = info["height"]

    # Crop 1–2% per side — invisible to human eye but changes pHash
    crop_left   = max(int(w * random.uniform(0.01, 0.02)), 2)
    crop_right  = max(int(w * random.uniform(0.01, 0.02)), 2)
    crop_top    = max(int(h * random.uniform(0.01, 0.02)), 2)
    crop_bottom = max(int(h * random.uniform(0.01, 0.02)), 2)

    cropped_w = w - crop_left - crop_right
    cropped_h = h - crop_top - crop_bottom
    if cropped_w % 2 != 0:
        cropped_w -= 1
    if cropped_h % 2 != 0:
        cropped_h -= 1

    # Scale back to original (lanczos resampling changes hash)
    out_w = w
    out_h = h
    if out_w % 2 != 0:
        out_w += 1
    if out_h % 2 != 0:
        out_h += 1

    # Very small rotation (0.1–0.3°) — barely visible, creates interpolation artifacts
    angle_deg = round(random.uniform(0.1, 0.3) * random.choice([-1, 1]), 3)
    angle_rad = round(angle_deg * 3.14159265 / 180, 6)

    # Subtle adjustments — not visible to naked eye
    noise_strength = random.randint(1, 2)
    brightness = round(random.uniform(-0.005, 0.005), 4)
    contrast   = round(random.uniform(0.996, 1.004), 4)
    saturation = round(random.uniform(0.997, 1.003), 4)
    gamma      = round(random.uniform(0.998, 1.002), 4)
    hue_h      = round(random.uniform(-0.5, 0.5), 3)
    hue_s      = round(random.uniform(0.998, 1.002), 4)

    # Subtle color channel shift
    cb_rs = round(random.uniform(0.005, 0.015), 3)
    cb_gs = round(random.uniform(-0.010, -0.002), 3)
    cb_bs = round(random.uniform(0.002, 0.010), 3)

    filters = [
        f"crop={cropped_w}:{cropped_h}:{crop_left}:{crop_top}",
        f"scale={out_w}:{out_h}:flags=lanczos",
        f"rotate={angle_rad}:c=black:ow=iw:oh=ih",
        f"noise=c0s={noise_strength}:c0f=t",
        f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}",
        f"hue=h={hue_h}:s={hue_s}",
        f"colorbalance=rs={cb_rs}:gs={cb_gs}:bs={cb_bs}",
    ]

    logger.info(
        f"Video filter: crop=({crop_left},{crop_top},{crop_right},{crop_bottom}) "
        f"scale={out_w}x{out_h} rotate={angle_deg}° noise={noise_strength} "
        f"colorbalance=rs={cb_rs}:gs={cb_gs}:bs={cb_bs}"
    )
    return ",".join(filters)


def process_video(input_path: str, output_path: str) -> None:
    try:
        probe_data = probe_video(input_path)
    except Exception:
        logger.error(f"probe_video failed:\n{traceback.format_exc()}")
        raise

    try:
        info = get_video_info(probe_data)
    except Exception:
        logger.error(f"get_video_info failed:\n{traceback.format_exc()}")
        raise

    try:
        vf = build_video_filter(info)
    except Exception:
        logger.error(f"build_video_filter failed:\n{traceback.format_exc()}")
        raise

    # Very slight tempo shift (0.1% max) — safe range for atempo is 0.5–2.0
    audio_tempo = round(random.uniform(0.9995, 1.0005), 4)
    fps_variation = round(float(info["fps"]) + random.uniform(-0.05, 0.05), 3)

    # ── 1. GOP randomization ──────────────────────────────────────────────────
    # Default GOP is typically 2×fps. Randomizing changes keyframe structure,
    # which is a strong signal Facebook uses for video fingerprinting.
    fps_num = float(info["fps"])
    gop_size = random.randint(int(fps_num * 1.5), int(fps_num * 4))
    gop_size = max(gop_size, 1)
    keyint_min = max(1, gop_size // 4)
    logger.info(f"GOP: gop={gop_size} keyint_min={keyint_min}")

    # ── 2. Fake metadata injection ────────────────────────────────────────────
    # Randomize creation_time (random day in last 2 years) and encoder string.
    # Facebook reads container metadata — different values = different fingerprint.
    days_ago = random.randint(1, 730)
    rand_dt = datetime.datetime.now() - datetime.timedelta(
        days=days_ago,
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    creation_time = rand_dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    encoder_tags = [
        "Lavf58.76.100", "Lavf59.16.100", "Lavf60.3.100",
        "Lavf58.45.100", "Lavf59.27.100",
    ]
    fake_encoder = random.choice(encoder_tags)
    logger.info(f"Metadata: creation_time={creation_time} encoder={fake_encoder}")

    # ── 3. Audio resampling (±1 Hz) ───────────────────────────────────────────
    # 44099 or 44101 instead of 44100 changes every DCT/FFT block in the audio
    # stream, defeating audio fingerprinting entirely.
    orig_sample_rate = info["audio_sample_rate"]
    audio_resample_rate = orig_sample_rate + random.choice([-1, 1])
    logger.info(f"Audio resample: {orig_sample_rate} → {audio_resample_rate} Hz")

    # Target original bitrate to preserve file size.
    orig_kbps = info["bitrate_kbps"]
    if orig_kbps > 0:
        video_bitrate_args = [
            "-b:v", f"{orig_kbps}k",
            "-maxrate", f"{int(orig_kbps * 1.3)}k",
            "-bufsize", f"{int(orig_kbps * 2)}k",
        ]
        logger.info(f"Using target bitrate: {orig_kbps}k (original)")
    else:
        video_bitrate_args = ["-crf", "18"]
        logger.info("Bitrate unknown, using CRF 18")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        # Strip ALL original metadata
        "-map_metadata", "-1",
        "-map_metadata:s:v", "-1",
        "-map_metadata:s:a", "-1",
        "-map_chapters", "-1",
        # Inject randomized metadata (different fingerprint each run)
        "-metadata", f"creation_time={creation_time}",
        "-metadata", f"encoder={fake_encoder}",
        "-metadata", "title=",
        "-metadata", "comment=",
        "-metadata", "description=",
        "-metadata", "copyright=",
        # Clear stream handler names
        "-metadata:s:v", "handler_name=",
        "-metadata:s:v", "vendor_id=",
        "-metadata:s:v", "language=",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        *video_bitrate_args,
        "-r", str(fps_variation),
        # GOP randomization — changes keyframe structure
        "-g", str(gop_size),
        "-keyint_min", str(keyint_min),
        "-movflags", "+faststart",
        "-fflags", "+bitexact",
        "-flags:v", "+bitexact",
        "-flags:a", "+bitexact",
    ]

    if info["has_audio"]:
        cmd += [
            "-c:a", "aac",
            "-b:a", "128k",
            # Resample to ±1 Hz — changes every audio block's hash
            "-ar", str(audio_resample_rate),
            "-af", f"atempo={audio_tempo}",
            "-metadata:s:a", "handler_name=",
            "-metadata:s:a", "language=",
        ]
    else:
        cmd += ["-an"]

    cmd.append(output_path)

    try:
        _run_ffmpeg(cmd, timeout=600)
    except Exception:
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
