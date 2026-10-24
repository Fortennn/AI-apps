"""Підготовка зображення до надсилання моделі.

Перевірка коректності, розмірів, EXIF-орієнтації та масштабування.
"""

import io
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання."""
    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    """Перевірити байти, що надійшли, і підготувати зображення для моделі."""
    if not content:
        raise ImageError("Файл порожній.")

    if len(content) > UPLOAD_MAX_BYTES:
        max_mb = UPLOAD_MAX_BYTES / (1024 * 1024)
        actual_mb = len(content) / (1024 * 1024)
        raise ImageError(f"Розмір файлу ({actual_mb:.1f} МБ) перевищує ліміт {max_mb:.0f} МБ.")

    try:
        # Запобіжник від decompression bomb (40 мегапікселів)
        Image.MAX_IMAGE_PIXELS = 40_000_000
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception as exc:
        raise ImageError(f"Файл не є придатним або підтримуваним зображенням: {exc}")

    orig_w, orig_h = img.size
    if orig_w < 50 or orig_h < 50:
        raise ImageError(f"Роздільність зображення ({orig_w}x{orig_h}) замала для розпізнавання документа.")

    # Враховуємо орієнтацію з EXIF (актуально для знімків з телефону)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Конвертуємо в RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    cur_w, cur_h = img.size
    max_side = max(cur_w, cur_h)

    # Зменшуємо до IMAGE_MAX_SIDE зі збереженням пропорцій
    if max_side > IMAGE_MAX_SIDE:
        scale = IMAGE_MAX_SIDE / max_side
        new_w = max(1, int(cur_w * scale))
        new_h = max(1, int(cur_h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    # PNG без втрат зберігає чіткість дрібного тексту та цифр
    img.save(buf, format="PNG")
    sent_bytes = buf.getvalue()

    return PreparedImage(
        data=sent_bytes,
        mime="image/png",
        original={"width": orig_w, "height": orig_h, "bytes": len(content)},
        sent={"width": img.width, "height": img.height, "bytes": len(sent_bytes)},
    )
