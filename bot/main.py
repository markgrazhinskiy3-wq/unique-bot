import os
import io
import json
import asyncio
import logging
import traceback
from pathlib import Path

import httpx

from telegram import Update, Document, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
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

LOCAL_API_URL = os.environ.get("LOCAL_BOT_API_URL", "").rstrip("/")
USE_LOCAL_API = bool(LOCAL_API_URL)

TELEGRAM_FILE_SIZE_LIMIT = 2000 * 1024 * 1024 if USE_LOCAL_API else 20 * 1024 * 1024
USER_FILE_SIZE_LIMIT = 75 * 1024 * 1024
SIZE_LIMIT_LABEL_ADMIN = "2 ГБ" if USE_LOCAL_API else "20 МБ"
SIZE_LIMIT_LABEL_USER = "75 МБ"

ADMIN_USER_ID = int(os.environ.get("ADMIN_USER_ID", "437752097"))
WHITELIST_PATH = Path(os.environ.get("WHITELIST_PATH", "/data/whitelist.json"))

DEBUG_ERRORS = os.environ.get("DEBUG_ERRORS", "true").lower() == "true"
PROCESSING_TIMEOUT = 300


def _load_whitelist() -> set[int]:
    try:
        if WHITELIST_PATH.exists():
            data = json.loads(WHITELIST_PATH.read_text())
            return set(int(x) for x in data.get("users", []))
    except Exception as e:
        logger.warning(f"Failed to load whitelist: {e}")
    return set()


def _save_whitelist(whitelist: set[int]) -> None:
    try:
        WHITELIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        WHITELIST_PATH.write_text(json.dumps({"users": list(whitelist)}, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save whitelist: {e}")


whitelist: set[int] = _load_whitelist()


def _is_admin(user_id: int) -> bool:
    return user_id == ADMIN_USER_ID


def _is_allowed(user_id: int) -> bool:
    return _is_admin(user_id) or user_id in whitelist


def _file_limit_for(user_id: int) -> int:
    return TELEGRAM_FILE_SIZE_LIMIT if _is_admin(user_id) else USER_FILE_SIZE_LIMIT


def _size_label_for(user_id: int) -> str:
    return SIZE_LIMIT_LABEL_ADMIN if _is_admin(user_id) else SIZE_LIMIT_LABEL_USER


def _error_msg(label: str, e: Exception) -> str:
    if DEBUG_ERRORS:
        short = str(e)[:300]
        return f"❌ Ошибка обработки {label}: {short}"
    return "❌ Ошибка обработки. Попробуйте ещё раз."


def _check_file_size(file_size: int | None, user_id: int) -> bool:
    if file_size is None:
        return True
    return file_size <= _file_limit_for(user_id)


async def _download_file(file_id: str, token: str, context: ContextTypes.DEFAULT_TYPE) -> bytes:
    if USE_LOCAL_API:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{LOCAL_API_URL}/bot{token}/getFile",
                json={"file_id": file_id},
            )
            resp.raise_for_status()
            file_path = resp.json()["result"]["file_path"]

            local_path = Path(file_path)
            if local_path.exists():
                logger.info(f"Reading file directly from disk: {file_path}")
                return local_path.read_bytes()

            token_marker = f"/{token}/"
            if token_marker in file_path:
                relative_path = file_path.split(token_marker, 1)[1]
            else:
                relative_path = file_path.lstrip("/")

            download_url = f"{LOCAL_API_URL}/file/bot{token}/{relative_path}"
            logger.info(f"Downloading via local API: {download_url}")
            file_resp = await client.get(download_url)
            file_resp.raise_for_status()
            return file_resp.content
    else:
        tg_file = await context.bot.get_file(file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        return buf.getvalue()


async def _notify_admin_new_user(context: ContextTypes.DEFAULT_TYPE, user: object) -> None:
    name = user.full_name or "Без имени"
    username = f"@{user.username}" if user.username else "нет"
    uid = user.id
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Одобрить", callback_data=f"approve:{uid}"),
            InlineKeyboardButton("❌ Отклонить", callback_data=f"reject:{uid}"),
        ]
    ])
    text = (
        f"🔔 Новый пользователь просит доступ:\n\n"
        f"👤 Имя: {name}\n"
        f"🔗 Username: {username}\n"
        f"🆔 ID: {uid}"
    )
    await context.bot.send_message(chat_id=ADMIN_USER_ID, text=text, reply_markup=keyboard)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    uid = user.id

    if _is_admin(uid):
        msg = (
            "👋 Привет, админ!\n\n"
            "Отправь мне фото или видео, и я сделаю его уникальным для Facebook.\n\n"
            "📸 Фото — отправляй как файл (документ) для лучшего качества.\n"
            "🎬 Видео — отправляй как файл (документ).\n\n"
            f"⚠️ Максимальный размер файла: {SIZE_LIMIT_LABEL_ADMIN}.\n\n"
            "Команды:\n"
            "/users — список пользователей\n"
            "/kick <ID> — удалить пользователя"
        )
        await update.message.reply_text(msg)
        return

    if uid in whitelist:
        msg = (
            "👋 Привет! Я бот для уникализации креативов.\n\n"
            "Отправь мне фото или видео, и я сделаю его уникальным для Facebook.\n\n"
            "📸 Фото — отправляй как файл (документ) для лучшего качества.\n"
            "🎬 Видео — отправляй как файл (документ).\n\n"
            f"⚠️ Максимальный размер файла: {SIZE_LIMIT_LABEL_USER}."
        )
        await update.message.reply_text(msg)
        return

    await update.message.reply_text(
        "🔐 Доступ к боту закрыт.\n\n"
        "Ваш запрос отправлен администратору. Ожидайте одобрения."
    )
    await _notify_admin_new_user(context, user)


async def users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not _is_admin(uid):
        return

    if not whitelist:
        await update.message.reply_text("📋 Список пользователей пуст.")
        return

    lines = [f"📋 Пользователи ({len(whitelist)}):"]
    for user_id in whitelist:
        lines.append(f"• {user_id}")
    lines.append("\nДля удаления: /kick <ID>")
    await update.message.reply_text("\n".join(lines))


async def kick_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not _is_admin(uid):
        return

    args = context.args
    if not args:
        await update.message.reply_text("Использование: /kick <user_id>")
        return

    try:
        target_id = int(args[0])
    except ValueError:
        await update.message.reply_text("❌ Неверный ID.")
        return

    if target_id in whitelist:
        whitelist.discard(target_id)
        _save_whitelist(whitelist)
        await update.message.reply_text(f"✅ Пользователь {target_id} удалён.")
        try:
            await context.bot.send_message(
                chat_id=target_id,
                text="🚫 Ваш доступ к боту был отозван администратором."
            )
        except Exception:
            pass
    else:
        await update.message.reply_text(f"❌ Пользователь {target_id} не найден.")


async def approve_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if not _is_admin(query.from_user.id):
        return

    action, target_str = query.data.split(":", 1)
    target_id = int(target_str)

    if action == "approve":
        whitelist.add(target_id)
        _save_whitelist(whitelist)
        await query.edit_message_text(
            query.message.text + f"\n\n✅ Одобрен"
        )
        try:
            await context.bot.send_message(
                chat_id=target_id,
                text=(
                    "✅ Ваш доступ одобрен администратором!\n\n"
                    "Отправьте /start чтобы начать."
                )
            )
        except Exception:
            pass
    elif action == "reject":
        await query.edit_message_text(
            query.message.text + f"\n\n❌ Отклонён"
        )
        try:
            await context.bot.send_message(
                chat_id=target_id,
                text="❌ Ваш запрос на доступ отклонён администратором."
            )
        except Exception:
            pass


async def _access_denied(update: Update) -> None:
    await update.message.reply_text(
        "🔐 У вас нет доступа к боту.\n"
        "Напишите /start чтобы запросить доступ."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(user.id):
        await _access_denied(update)
        return

    doc: Document = update.message.document
    if not doc:
        await update.message.reply_text("⚠️ Неподдерживаемый формат. Отправьте фото или видео.")
        return

    if not _check_file_size(doc.file_size, user.id):
        limit = _size_label_for(user.id)
        await update.message.reply_text(f"⚠️ Файл слишком большой (лимит {limit}).\nОтправьте файл меньшего размера.")
        return

    filename = doc.file_name or "file"
    ext = Path(filename).suffix.lower()

    if ext in SUPPORTED_IMAGE_EXTS:
        await process_image_message(update, context, doc.file_id, filename)
    elif ext in SUPPORTED_VIDEO_EXTS:
        await process_video_message(update, context, doc.file_id, filename)
    else:
        await update.message.reply_text("⚠️ Неподдерживаемый формат. Отправьте фото или видео.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(user.id):
        await _access_denied(update)
        return

    photos = update.message.photo
    if not photos:
        await update.message.reply_text("⚠️ Неподдерживаемый формат.")
        return
    best = max(photos, key=lambda p: p.file_size or 0)
    if not _check_file_size(best.file_size, user.id):
        limit = _size_label_for(user.id)
        await update.message.reply_text(f"⚠️ Файл слишком большой (лимит {limit}).")
        return
    await process_image_message(update, context, best.file_id, "photo.jpg")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(user.id):
        await _access_denied(update)
        return

    video = update.message.video
    if not video:
        await update.message.reply_text("⚠️ Неподдерживаемый формат.")
        return
    if not _check_file_size(video.file_size, user.id):
        limit = _size_label_for(user.id)
        await update.message.reply_text(f"⚠️ Файл слишком большой (лимит {limit}).")
        return
    filename = video.file_name or "video.mp4"
    await process_video_message(update, context, video.file_id, filename)


async def process_image_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    filename: str,
) -> None:
    status_msg = await update.message.reply_text("⏳ Обрабатываю файл, подожди немного...")
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
        await status_msg.edit_text("⏱ Слишком долгая обработка. Попробуйте файл поменьше.")
    except Exception as e:
        logger.error(f"Image processing error:\n{traceback.format_exc()}")
        await status_msg.edit_text(_error_msg("фото", e))


async def process_video_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_id: str,
    filename: str,
) -> None:
    status_msg = await update.message.reply_text("⏳ Обрабатываю файл, подожди немного...")
    try:
        async def do_process():
            video_bytes = await _download_file(file_id, context.bot.token, context)
            logger.info(f"Downloaded video: {filename} ({len(video_bytes)/1024/1024:.1f}MB)")
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
        await status_msg.edit_text("⏱ Слишком долгая обработка. Попробуйте файл поменьше.")
    except Exception as e:
        logger.error(f"Video processing error for {filename}:\n{traceback.format_exc()}")
        await status_msg.edit_text(_error_msg("видео", e))


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")

    logger.info(f"Starting bot. Local API: {'YES → ' + LOCAL_API_URL if USE_LOCAL_API else 'NO (standard API)'}")
    logger.info(f"Admin ID: {ADMIN_USER_ID} | Whitelist: {len(whitelist)} users")

    builder = Application.builder().token(token)
    if USE_LOCAL_API:
        builder = builder.base_url(f"{LOCAL_API_URL}/bot").base_file_url(f"{LOCAL_API_URL}/file/bot")

    app = builder.build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("users", users_handler))
    app.add_handler(CommandHandler("kick", kick_handler))
    app.add_handler(CallbackQueryHandler(approve_callback, pattern=r"^(approve|reject):\d+$"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))

    logger.info("Bot started. Polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
