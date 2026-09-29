"""Веб-рівень застосунку: сторінка діалогу і JSON-ендпоінт.

Цей файл не знає про провайдера моделі чи складання списку повідомлень —
усе це лишається в `app/llm.py` і `app/schema.py`. Тут вирішується: що
застосунок приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr4:
    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Turn(BaseModel):
    """Одна репліка розмови: `user` — клієнт, `assistant` — помічник."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Те, що надсилає сторінка: нове повідомлення й розмову до нього."""

    message: str
    history: list[Turn] = []


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель.

    Контекст — це дані застосунку, а не знання моделі. Він живе окремим
    файлом і передається в запит разом з історією та зверненням.
    """
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку діалогу."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    """Повернути структуровану відповідь помічника у форматі JSON.

    Сторінка очікує обʼєкт із полями `result` (перевірена відповідь моделі;
    текст для клієнта — у `result.reply`), `model`, `elapsed` і `usage`.
    """
    clean_message = payload.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=400,
            detail="Повідомлення клієнта не може бути порожнім."
        )

    # Захист від підробленої або завеликої історії від клієнта
    # Валідуємо ролі реплік в історії
    history = []
    for turn in payload.history:
        role = turn.role.strip().lower()
        if role not in ("user", "assistant"):
            continue
        history.append({"role": role, "content": turn.content})

    try:
        context = load_context()
        response_data = llm.ask(clean_message, history, context)
        return response_data
    except llm.LLMError as exc:
        logger.error(f"Помилка LLM ({exc.status_code}): {exc}")
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception as exc:
        logger.exception("Непередбачена помилка під час обробки діалогу")
        raise HTTPException(
            status_code=500,
            detail="Внутрішня помилка сервера при обробці запиту."
        )
