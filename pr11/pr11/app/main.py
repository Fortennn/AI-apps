"""Веб-рівень застосунку: сторінка помічника оператора і JSON-ендпоінти.
"""

import json
import logging
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import budget, cache, costs, drafting, gateway, jobs, limits, samples, telemetry

telemetry.setup()
log = logging.getLogger("app.http")

app = FastAPI(title="Помічник оператора — ПР11")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


class DraftIn(BaseModel):
    """Замовлення чернетки зі сторінки."""

    request_id: str
    client: str = "anonymous"
    fresh: bool = False


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Ідентифікатор запиту в контексті й у відповіді; запис про запит у журнал."""
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _REQUEST_ID.match(incoming) else telemetry.new_request_id()
    token = telemetry.request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        log.exception("http.unhandled", extra={"method": request.method, "path": request.url.path})
        raise
    finally:
        log.info("http.request", extra={
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        telemetry.request_id_var.reset(token)


def draft_to_dict(draft: jobs.Draft) -> dict:
    """Знімок чернетки для сторінки — без журналу подій."""
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    """Чинні налаштування."""
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
        "cache": cache.settings(),
        "budget": budget.settings(),
        "prompt_version": getattr(drafting, "PROMPT_VERSION", "drafts-v1"),
        "log_file": telemetry.settings()["log_file"],
    }


@app.get("/api/usage")
def api_usage() -> dict:
    """Стан витрат і кешу — для панелі на сторінці."""
    return {
        "budget": budget.status(),
        "cache": cache.stats(),
        "pricing": costs.settings(),
    }


@app.get("/api/requests")
def api_requests() -> list[dict]:
    """Зразки звернень для вибору на сторінці."""
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> JSONResponse:
    """Замовити чернетку."""
    current_request_id = telemetry.request_id_var.get() or telemetry.new_request_id()
    sample = samples.get(payload.request_id)
    if not sample:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "kind": "not_found",
                    "message": f"Звернення з номером '{payload.request_id}' не знайдено",
                    "retryable": False,
                    "request_id": current_request_id,
                }
            },
        )

    try:
        limits.acquire(payload.client)
    except limits.RateLimited as rl:
        retry_secs = int(math.ceil(rl.retry_after))
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_secs)},
            content={
                "error": {
                    "kind": "rate_limit",
                    "message": f"Перевищено ліміт запитів. Повторіть спробу через {retry_secs} с.",
                    "retryable": True,
                    "retry_after": rl.retry_after,
                    "request_id": current_request_id,
                }
            },
        )

    draft = jobs.start(sample, payload.client, fresh=payload.fresh)
    return JSONResponse(status_code=202, content=draft_to_dict(draft))


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    """Останні чернетки, нові першими."""
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> JSONResponse:
    """Знімок однієї чернетки."""
    draft = jobs.get(draft_id)
    if not draft:
        current_request_id = telemetry.request_id_var.get() or telemetry.new_request_id()
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "kind": "not_found",
                    "message": f"Чернетку '{draft_id}' не знайдено",
                    "retryable": False,
                    "request_id": current_request_id,
                }
            },
        )
    return JSONResponse(content=draft_to_dict(draft))


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request):
    """Потік подій чернетки у форматі server-sent events."""
    draft = jobs.get(draft_id)
    if not draft:
        current_request_id = telemetry.request_id_var.get() or telemetry.new_request_id()
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "kind": "not_found",
                    "message": f"Чернетку '{draft_id}' не знайдено",
                    "retryable": False,
                    "request_id": current_request_id,
                }
            },
        )

    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> JSONResponse:
    """Скасувати генерацію чернетки."""
    draft = jobs.get(draft_id)
    current_request_id = telemetry.request_id_var.get() or telemetry.new_request_id()
    if not draft:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "kind": "not_found",
                    "message": f"Чернетку '{draft_id}' не знайдено",
                    "retryable": False,
                    "request_id": current_request_id,
                }
            },
        )

    try:
        updated_draft = jobs.cancel(draft_id)
        return JSONResponse(content=draft_to_dict(updated_draft))
    except ValueError as ve:
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "kind": "conflict",
                    "message": str(ve),
                    "retryable": False,
                    "request_id": current_request_id,
                },
                "draft": draft_to_dict(draft),
            },
        )
