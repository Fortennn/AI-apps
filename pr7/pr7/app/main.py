"""Веб-рівень застосунку: сторінка розбору рахунків і JSON-ендпоінти."""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import extraction, images, llm

app = FastAPI(title="Розбір рахунків — ПР7")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def result_to_dict(result: extraction.Result) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "document": result.document,
        "issues": [asdict(issue) for issue in result.issues],
        "image": result.image,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/samples")
def api_samples() -> dict:
    """Перелік зразків за папками: `clean`, `degraded` і, якщо є, `own`."""
    groups = {}
    for folder in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            groups[folder.name] = files
    return groups


@app.post("/api/extract")
async def api_extract(image: UploadFile = File(...)) -> dict:
    """Розібрати документ і повернути поля, проблеми й рішення."""
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Файл не було завантажено."
        )

    try:
        content = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не вдалося прочитати байти файлу: {exc}"
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Завантажений файл порожній (0 байтів)."
        )

    try:
        result = extraction.process(content)
        return result_to_dict(result)
    except images.ImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
    except llm.LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутрішня помилка під час обробки: {exc}"
        )
