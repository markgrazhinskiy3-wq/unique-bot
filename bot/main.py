import os
import io
import asyncio
import logging
import traceback
from pathlib import Path

import httpx

from telegram import Update, Document, PhotoSize
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from processors.image_processor import process_image
from processors.video_processor import process_video_file
from utils.helpers import is_image, is_video
from utils.verification import verify_image, verify_video

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# If LOCAL_BOT_API_URL is set — use local server (up to 2 GB).
# Otherwise — use standard Telegram API (up to 20 MB).
LOCAL_API_URL = os.environ.get("LOCAL_BOT_API_URL", "").rstrip("/")
USE_LOCAL_API = bool(LOCAL_API_URL)

TELEGRAM_FILE_SIZE_LIMIT = 2000 * 1024 * 1024 if USE_LOCAL_API else 20 * 1024 * 1024
SIZE_LIMIT_LABEL = "2 ГБ" if USE_LOCAL_API else "20 МБ"

DEBUG_ERRORS = os.environ.get("DEBUG_ERRORS", "true").lower() == "true"
PROCESSING_TIMEOUT = 300

MSG_START = (
    "👋 Привет! Я бот для уникализации креативов.\n\n"
    "Отправь мне фото или видео, и я сделаю его уникальным для Facebook.\n\n"
    "📸 Фото — отправляй как файл (документ) для лучшего качества.\n"
    "🎬 Видео — отправляй как файл (документ).\n\n"
    "Можно отправлять и обычным способом (сжатым), но качество будет хуже.\n\n"
    f"⚠️ Максимальный размер файла: {SIZE_LIMIT_LABEL}."
)
MSG_PROCESSING = "⏳ Обрабатываю файл, подожди немного..."
MSG_WRONG_FORMAT = "⚠️ Неподдерживаемый формат. Отправьте фото или видео."
MSG_PROCESSING_ERROR = "❌ Ошибка обработки. Попробуйте ещё раз."
MSG_TIMEOUT = "⏱ Слишком долгая обработка. Попробуйте файл поменьше."
MSG_TOO_LARGE = (
    f"⚠️ Файл слишком большой (лимит {SIZE_LIMIT_LABEL}).\n"
    "Отправьте файл меньшего размера."
)


def _error_msg(label: str, e: Exception) -> str:
    if DEBUG_ERRORS:
        short = str(e)[:300]
        return f"❌ Ошибка обработки {label}: {short}"
    return MSG_PROCESSING_ERROR


def _check_file_size(file_size: int | None) -> bool:
    if file_size is None:
        return True
    return file_size <= TELEGRAM_FILE_SIZE_LIMIT


async def _download_file(file_id: str, token: str, context: ContextTypes.DEFAULT_TYPE) -> bytes:
    if USE_LOCAL_API:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{LOCAL_API_URL}/bot{token}/getFile",
                json={"file_id": file_id},
            )
            resp.raise_for_status()
            file_path = resp.json()["result"]["file_path"]

            # In --local mode the server returns an absolute filesystem path like:
            # /var/lib/telegram-bot-api/{token}/documents/file_0.png
            # Correct download URL: {LOCAL_API_URL}/file/bot{token}/documents/file_0.png
            # We must strip everything up to and including /{token}/
            token_marker = f"/{token}/"
            if token_marker in file_path:
                relative_path = file_path.split(token_marker, 1)[1]
            else:
                # Fallback for non-local mode (leading slash only)
                relative_path = file_path.lstrip("/")

            download_url = f"{LOCAL_API_URL}/file/bot{token}/{relative_path}"
            logger.info(f"Downloading via local API: {download_url}")
            file_resp = await client.get(download_url)
            file_resp.raise_for_status()
            return file_resp.content
    else:
        # Standard Telegram API (up to 20 MB)
        tg_file = await context.bot.get_file(file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        return buf.getvalue()


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(MSG_START)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc: Document = update.message.document
    if not doc:
        await update.message.reply_text(MSG_WRONG_FORMAT)
        return

    if not _check_file_size(doc.file_size):
        await update.message.reply_text(MSG_TOO_LARGE)
        return

    filename = doc.file_name or "file"
    ext = Path(filename).suffix.lower()

    if ext in SUPPORTED_IMAGE_EXTS:
        await process_image_message(update, context, doc.file_id, filename)
    elif ext in SUPPORTED_VIDEO_EXTS:
        await process_video_message(update, context, doc.file_id, filename)
    else:
        await update.message.reply_text(MSG_WRONG_FORMAT)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photos = update.message.photo
    if not photos:
        await update.message.reply_text(MSG_WRONG_FORMAT)
        return
    best = max(photos, key=lambda p: p.file_size or 0)
    if not _check_file_size(best.file_size):
        await update.message.reply_text(MSG_TOO_LARGE)
        return
    await process_image_message(update, context, best.file_id, "photo.jpg")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    video = update.message.video
    if not video:
        await update.message.reply_text(MSG_WRONG_FORMAT)
        return
    if not _check_file_size(video.file_size):
        await update.message.reply_text(MSG_TOO_LARGE)
        return
    filename = video.file_name or "video.mp4"
    await process_video_message(update, context, video.file_id, filename)


async def process_image_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    filename: str,
) -> None:
    status_msg = await update.message.reply_text(MSG_PROCESSING)
    try:
        async def do_process():
            image_bytes = await _download_file(file_id, context.bot.token, context)

            loop = asyncio.get_event_loop()
            result_bytes, out_filename = await loop.run_in_executor(
                None, process_image, image_bytes, filename
            )
            caption = await loop.run_in_executor(
                None, verify_image, image_bytes, result_bytes
            )
            return result_bytes, out_filename, caption

        result_bytes, out_filename, caption = await asyncio.wait_for(
            do_process(), timeout=PROCESSING_TIMEOUT
        )

        await status_msg.delete()
        await update.message.reply_document(
            document=io.BytesIO(result_bytes),
            filename=out_filename,
            caption=caption,
            disable_content_type_detection=True,
        )
    except asyncio.TimeoutError:
        await status_msg.edit_text(MSG_TIMEOUT)
    except Exception as e:
        logger.error(f"Image processing error:\n{traceback.format_exc()}")
        await status_msg.edit_text(_error_msg("фото", e))


async def process_video_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    filename: str,
) -> None:
    status_msg = await update.message.reply_text(MSG_PROCESSING)
    try:
        async def do_process():
            video_bytes = await _download_file(file_id, context.bot.token, context)

            logger.info(
                f"Downloaded video: {filename} "
                f"({len(video_bytes)/1024/1024:.1f}MB)"
            )

            loop = asyncio.get_event_loop()
            result_bytes, out_filename = await loop.run_in_executor(
                None, process_video_file, video_bytes, filename
            )
            caption = await loop.run_in_executor(
                None, verify_video, video_bytes, result_bytes
            )
            return result_bytes, out_filename, caption

        result_bytes, out_filename, caption = await asyncio.wait_for(
            do_process(), timeout=PROCESSING_TIMEOUT
        )

        await status_msg.delete()
        await update.message.reply_document(
            document=io.BytesIO(result_bytes),
            filename=out_filename,
            caption=caption,
            disable_content_type_detection=True,
        )
    except asyncio.TimeoutError:
        logger.error(f"Video processing timeout for {filename}")
        await status_msg.edit_text(MSG_TIMEOUT)
    except Exception as e:
        logger.error(f"Video processing error for {filename}:\n{traceback.format_exc()}")
        await status_msg.edit_text(_error_msg("видео", e))


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")

    logger.info(f"Starting bot. Local API: {'YES → ' + LOCAL_API_URL if USE_LOCAL_API else 'NO (standard API)'}")

    builder = Application.builder().token(token)
    if USE_LOCAL_API:
        builder = builder.base_url(f"{LOCAL_API_URL}/bot").base_file_url(f"{LOCAL_API_URL}/file/bot")

    app = builder.build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    logger.info("Bot started. Polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
