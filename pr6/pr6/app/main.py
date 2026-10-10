"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import index, keyword, rag, llm

app = FastAPI(title="Помічник за базою знань — ПР6")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Те, що надсилає сторінка."""
    question: str
    filters: dict = {}


def hit_to_dict(hit: index.Hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


def answer_to_dict(result: rag.Answer) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "answer": result.text,
        "found": result.found,
        "sources": [
            {"ref": s.ref, "score": s.score, "source": s.chunk.source,
             "metadata": s.chunk.metadata, "text": s.chunk.text}
            for s in result.sources
        ],
        "retrieved": [hit_to_dict(h) for h in result.retrieved],
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Прочитати збудований індекс."""
    app.state.index = None
    app.state.keyword_index = None
    try:
        app.state.index = index.load()
    except Exception as exc:  # noqa: BLE001 — старт не має падати без індексу
        print(f"Індекс не завантажено: {type(exc).__name__}: {exc}")
        return
    app.state.keyword_index = keyword.build(app.state.index.chunks)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку помічника."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Стан індексу: чи збудовано, скільки фрагментів, якою моделлю."""
    idx = getattr(app.state, "index", None)
    if idx is None:
        return {"ready": False, "hint": "індекс не збудовано — виконайте python ingest.py"}
    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """Відповісти на питання й повернути відповідь із джерелами."""
    if not payload.question or not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Запитання не може бути порожнім."
        )

    idx = getattr(app.state, "index", None)
    if idx is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Індекс бази знань не завантажено. Виконай python ingest.py."
        )

    try:
        result = rag.answer(
            payload.question.strip(),
            app.state.index,
            app.state.keyword_index,
            filters=payload.filters or None,
        )
        return answer_to_dict(result)
    except llm.LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Помилка сервісу мовної моделі: {exc}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутрішня помилка сервера: {exc}"
        )
