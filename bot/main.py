import os
import io
import asyncio
import logging
import tempfile
from pathlib import Path

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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MSG_START = (
    "👋 Привет! Я бот для уникализации креативов.\n\n"
    "Отправь мне фото или видео, и я сделаю его уникальным для Facebook.\n\n"
    "📸 Фото — отправляй как файл (документ) для лучшего качества.\n"
    "🎬 Видео — отправляй как файл (документ).\n\n"
    "Можно отправлять и обычным способом (сжатым), но качество будет хуже."
)
MSG_PROCESSING = "⏳ Обрабатываю файл, подожди немного..."
MSG_WRONG_FORMAT = "⚠️ Неподдерживаемый формат. Отправьте фото или видео."
MSG_PROCESSING_ERROR = "❌ Ошибка обработки. Попробуйте ещё раз."
MSG_TIMEOUT = "⏱ Слишком долгая обработка. Попробуйте файл поменьше."

PROCESSING_TIMEOUT = 300


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(MSG_START)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc: Document = update.message.document
    if not doc:
        await update.message.reply_text(MSG_WRONG_FORMAT)
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
    await process_image_message(update, context, best.file_id, "photo.jpg")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    video = update.message.video
    if not video:
        await update.message.reply_text(MSG_WRONG_FORMAT)
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
            tg_file = await context.bot.get_file(file_id)
            buf = io.BytesIO()
            await tg_file.download_to_memory(buf)
            image_bytes = buf.getvalue()

            loop = asyncio.get_event_loop()
            result_bytes, out_filename = await loop.run_in_executor(
                None, process_image, image_bytes, filename
            )
            return result_bytes, out_filename

        result_bytes, out_filename = await asyncio.wait_for(
            do_process(), timeout=PROCESSING_TIMEOUT
        )

        await status_msg.delete()
        await update.message.reply_document(
            document=io.BytesIO(result_bytes),
            filename=out_filename,
            caption="✅ Готово! Фото уникализировано.",
        )
    except asyncio.TimeoutError:
        await status_msg.edit_text(MSG_TIMEOUT)
    except Exception as e:
        logger.error(f"Image processing error: {e}", exc_info=True)
        await status_msg.edit_text(MSG_PROCESSING_ERROR)


async def process_video_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    filename: str,
) -> None:
    status_msg = await update.message.reply_text(MSG_PROCESSING)
    try:
        async def do_process():
            tg_file = await context.bot.get_file(file_id)
            buf = io.BytesIO()
            await tg_file.download_to_memory(buf)
            video_bytes = buf.getvalue()

            loop = asyncio.get_event_loop()
            result_bytes, out_filename = await loop.run_in_executor(
                None, process_video_file, video_bytes, filename
            )
            return result_bytes, out_filename

        result_bytes, out_filename = await asyncio.wait_for(
            do_process(), timeout=PROCESSING_TIMEOUT
        )

        await status_msg.delete()
        await update.message.reply_document(
            document=io.BytesIO(result_bytes),
            filename=out_filename,
            caption="✅ Готово! Видео уникализировано.",
        )
    except asyncio.TimeoutError:
        await status_msg.edit_text(MSG_TIMEOUT)
    except Exception as e:
        logger.error(f"Video processing error: {e}", exc_info=True)
        await status_msg.edit_text(MSG_PROCESSING_ERROR)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    logger.info("Bot started. Polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
