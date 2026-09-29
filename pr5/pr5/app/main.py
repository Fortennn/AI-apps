"""Веб-рівень застосунку: сторінка пошуку і JSON-ендпоінти.

Цей файл не знає ані якою моделлю отримано вектори, ані як влаштований
індекс, ані як ранжує пошук за словами — усе це лишається в модулях
`app/embeddings.py`, `app/index.py`, `app/keyword.py`. Тут вирішується
інше: що застосунок приймає від сторінки, що віддає їй і з яким
HTTP-статусом.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import embeddings, index, keyword

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


def init_indexes(app_instance: FastAPI) -> None:
    """Прочитати збудований індекс і зібрати індекс за словами з тих
    самих фрагментів.
    """
    app_instance.state.index = None
    app_instance.state.keyword_index = None
    try:
        loaded_idx = index.load()
        app_instance.state.index = loaded_idx
        app_instance.state.keyword_index = keyword.build(loaded_idx.chunks)
    except Exception as exc:  # noqa: BLE001 — старт не має падати без індексу
        print(f"Індекс не завантажено: {type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    init_indexes(app_instance)
    yield


app = FastAPI(title="Пошук у базі знань — ПР5", lifespan=lifespan)
# Ініціалізація полів стану за замовчуванням
app.state.index = None
app.state.keyword_index = None
init_indexes(app)


class SearchRequest(BaseModel):
    """Те, що надсилає сторінка.

    `mode` — `semantic`, `keyword` або `both`. `filters` — умови на
    метадані фрагментів, наприклад `{"category": "інструкція"}`; порожній
    словник означає «без фільтрів». `threshold` — поріг схожості для
    семантичного пошуку; `None` — узяти значення з конфігурації.
    """

    query: str
    mode: str = "both"
    top_k: int = Field(default=index.DEFAULT_TOP_K, ge=1, le=100)
    filters: dict = {}
    threshold: float | None = None


def hit_to_dict(hit: index.Hit) -> dict:
    """Перетворити влучення на те, що піде на сторінку."""
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку пошуку."""
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


@app.post("/api/search")
def api_search(payload: SearchRequest) -> dict:
    """Виконати пошук і повернути влучення обох способів."""
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=400,
            detail="Пошуковий запит не може бути порожнім."
        )

    if payload.mode not in ("semantic", "keyword", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"Невідомий режим пошуку: '{payload.mode}'. Допустимі значення: 'semantic', 'keyword', 'both'."
        )

    idx = getattr(app.state, "index", None)
    if idx is None:
        raise HTTPException(
            status_code=503,
            detail="Індекс не знайдено на диску. Запустіть 'python ingest.py' для його побудови."
        )

    # Очищуємо порожні фільтри
    clean_filters = {
        k: v for k, v in payload.filters.items()
        if v is not None and str(v).strip() != ""
    }

    result: dict = {
        "query": clean_query,
        "semantic": None,
        "keyword": None,
        "elapsed": {}
    }

    # 1. Семантичний пошук
    if payload.mode in ("semantic", "both"):
        started = time.perf_counter()
        try:
            vector = embeddings.embed_query(clean_query)
            hits = index.search(
                idx,
                vector,
                top_k=payload.top_k,
                filters=clean_filters or None,
                threshold=payload.threshold if payload.threshold is not None else index.SIMILARITY_THRESHOLD,
            )
            result["elapsed"]["semantic"] = round(time.perf_counter() - started, 4)
            result["semantic"] = [hit_to_dict(h) for h in hits]
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Помилка семантичного пошуку: {exc}"
            ) from exc

    # 2. Пошук за ключовими словами (BM25)
    if payload.mode in ("keyword", "both"):
        started = time.perf_counter()
        try:
            hits = keyword.search(
                app.state.keyword_index,
                clean_query,
                top_k=payload.top_k,
                filters=clean_filters or None,
            )
            result["elapsed"]["keyword"] = round(time.perf_counter() - started, 4)
            result["keyword"] = [hit_to_dict(h) for h in hits]
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Помилка пошуку за ключовими словами: {exc}"
            ) from exc

    return result
