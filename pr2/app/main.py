"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт.

Цей файл не знає про `ultralytics`, ваги моделі й формат її виводу — усе це
лишається в `app/detector.py`. Тут вирішується: що застосунок віддає
клієнтові та з яким HTTP-статусом.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from . import detector

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Модель завантажується в пам'ять один раз під час старту сервера."""
    try:
        detector.load_model()
        print("[INFO] Модель YOLOv8 успішно завантажена в пам'ять.")
    except Exception as err:
        print(f"[WARN] Помилка завантаження моделі: {err}")
    yield


app = FastAPI(title="Детекція обʼєктів — ПР2", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати мінімалістичну сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    image: UploadFile = File(...),
    confidence: Optional[float] = Form(None),
    conf_query: Optional[float] = Query(None, alias="confidence"),
):
    """Повернути знайдені на зображенні обʼєкти у форматі JSON."""
    conf_val = confidence if confidence is not None else conf_query
    if conf_val is None or not (0.01 <= conf_val <= 1.0):
        conf_val = detector.DEFAULT_CONFIDENCE

    try:
        content = await image.read()
        return detector.detect(content, confidence=conf_val)
    except detector.InvalidImageError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except detector.ModelInferenceError as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
    except detector.DetectionError as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
