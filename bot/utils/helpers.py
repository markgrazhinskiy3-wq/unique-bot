import os
import tempfile
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_temp_file(suffix: str = "") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def cleanup(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.warning(f"Failed to cleanup {p}: {e}")


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_image(filename: str) -> bool:
    return get_file_extension(filename) in {".jpg", ".jpeg", ".png", ".webp"}


def is_video(filename: str) -> bool:
    return get_file_extension(filename) in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


async def run_with_timeout(coro, timeout: float = 300):
    return await asyncio.wait_for(coro, timeout=timeout)
