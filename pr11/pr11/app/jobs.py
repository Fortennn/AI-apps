"""Фонові задачі: чернетка генерується не в HTTP-запиті, а окремо.
"""

import contextvars
import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway, telemetry
from .providers.base import ProviderError

log = logging.getLogger("app.jobs")

TERMINAL = {"done", "failed", "cancelled"}
MAX_HISTORY = 200


@dataclass
class Draft:
    """Стан однієї чернетки — те, що зберігається в реєстрі і йде на
    сторінку як знімок."""

    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    cached: bool = False
    cost: float | None = None
    cost_estimated: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_cancel_events: dict[str, threading.Event] = {}
_lock = threading.Lock()
_changed = threading.Condition(_lock)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_draft(request_id: str, client: str) -> Draft:
    """Створити чернетку зі статусом `queued` і зареєструвати її."""
    draft = Draft(
        draft_id=uuid.uuid4().hex[:8],
        request_id=request_id,
        client=client,
        created_at=_now(),
        updated_at=_now(),
    )
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        if len(_order) > MAX_HISTORY:
            stale_ids = _order[MAX_HISTORY:]
            _order[:] = _order[:MAX_HISTORY]
            for sid in stale_ids:
                _drafts.pop(sid, None)
                _cancel_events.pop(sid, None)
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    """Чернетки, нові першими."""
    with _lock:
        return [_drafts[i] for i in _order[:limit] if i in _drafts]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    """Додати подію до журналу чернетки й розбудити підписників."""
    with _changed:
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
        draft.updated_at = _now()
        _changed.notify_all()


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    """Віддавати події чернетки, починаючи з номера `since + 1`."""
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def _run_draft_job(
    draft: Draft,
    sample: dict,
    cancel_event: threading.Event,
    fresh: bool,
    http_request_id: str,
) -> None:
    """Фоновий процес генерації чернетки в окремому потоці."""
    telemetry.request_id_var.set(http_request_id)

    if cancel_event.is_set():
        with _lock:
            draft.status = "cancelled"
            draft.updated_at = _now()
        emit(draft, "cancelled", {})
        log.info("draft.cancelled", extra={
            "draft_id": draft.draft_id,
            "request_id": http_request_id,
            "client": draft.client,
        })
        return

    with _lock:
        draft.status = "running"
        draft.updated_at = _now()

    emit(draft, "status", {
        "status": "running",
        "attempt": 1,
        "provider": gateway.PROVIDER,
        "fallback": False,
    })

    req = drafting.build_request(sample)
    req.metadata["draft_id"] = draft.draft_id
    req.metadata["request_id_header"] = http_request_id
    req.metadata["client"] = draft.client
    req.metadata["fresh"] = fresh

    started_at = time.time()
    first_token_time: float | None = None

    def on_delta(token: str) -> None:
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time()
        with _lock:
            draft.text += token
            draft.status = "running"
            draft.updated_at = _now()
        emit(draft, "delta", {"text": token})

    def on_status(data: dict) -> None:
        with _lock:
            if data.get("reset"):
                draft.text = ""
            if "attempt" in data:
                draft.attempts = data["attempt"]
            if "provider" in data:
                draft.provider = data["provider"]
            if "model" in data:
                draft.model = data["model"]
            if "fallback" in data:
                draft.fallback = data["fallback"]
            if "cached" in data:
                draft.cached = data["cached"]
            draft.updated_at = _now()
        emit(draft, "status", data)

    def is_cancelled() -> bool:
        return cancel_event.is_set()

    try:
        res = gateway.generate(req, on_delta, on_status, is_cancelled)

        if cancel_event.is_set():
            with _lock:
                draft.status = "cancelled"
                draft.updated_at = _now()
            emit(draft, "cancelled", {})
            log.info("draft.cancelled", extra={
                "draft_id": draft.draft_id,
                "request_id": http_request_id,
                "client": draft.client,
            })
            return

        total_elapsed = time.time() - started_at
        first_elapsed = (first_token_time - started_at) if first_token_time is not None else total_elapsed
        final_text = res.text if res.text else draft.text
        warnings = drafting.check(final_text, sample)

        with _lock:
            draft.status = "done"
            draft.text = final_text
            draft.provider = res.provider
            draft.model = res.model
            draft.cached = res.cached
            draft.cost = res.cost
            draft.cost_estimated = res.cost_estimated
            draft.usage = res.usage
            draft.elapsed = {
                "first_token": round(first_elapsed, 2),
                "total": round(total_elapsed, 2),
            }
            draft.warnings = warnings
            draft.updated_at = _now()

        emit(draft, "done", {
            "text": draft.text,
            "provider": draft.provider,
            "model": draft.model,
            "usage": draft.usage,
            "elapsed": draft.elapsed,
            "attempts": draft.attempts or 1,
            "fallback": draft.fallback,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
            "warnings": draft.warnings,
        })

        log.info("draft.done", extra={
            "draft_id": draft.draft_id,
            "request_id": http_request_id,
            "client": draft.client,
            "sample_id": sample.get("id"),
            "attempts": draft.attempts or 1,
            "provider": draft.provider,
            "model": draft.model,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
            "duration_ms": round(total_elapsed * 1000, 1),
        })

    except ProviderError as pe:
        total_elapsed = time.time() - started_at
        if pe.kind == "cancelled" or cancel_event.is_set():
            with _lock:
                draft.status = "cancelled"
                draft.updated_at = _now()
            emit(draft, "cancelled", {})
            log.info("draft.cancelled", extra={
                "draft_id": draft.draft_id,
                "request_id": http_request_id,
                "client": draft.client,
            })
        else:
            with _lock:
                draft.status = "failed"
                draft.error = {
                    "kind": pe.kind,
                    "message": pe.message,
                    "retryable": pe.retryable,
                }
                draft.updated_at = _now()
            emit(draft, "error", draft.error)
            log.error("draft.failed", extra={
                "draft_id": draft.draft_id,
                "request_id": http_request_id,
                "client": draft.client,
                "sample_id": sample.get("id"),
                "kind": pe.kind,
                "message": pe.message,
                "duration_ms": round(total_elapsed * 1000, 1),
            })
    except Exception as exc:
        total_elapsed = time.time() - started_at
        with _lock:
            draft.status = "failed"
            draft.error = {
                "kind": "unknown",
                "message": "Внутрішня помилка виконання фонової задачі",
                "retryable": False,
            }
            draft.updated_at = _now()
        emit(draft, "error", draft.error)
        log.exception("draft.exception", extra={
            "draft_id": draft.draft_id,
            "request_id": http_request_id,
            "client": draft.client,
            "sample_id": sample.get("id"),
            "duration_ms": round(total_elapsed * 1000, 1),
        })


def start(sample: dict, client: str, fresh: bool = False) -> Draft:
    """Створити чернетку для звернення `sample` і запустити генерацію у
    фоні."""
    draft = new_draft(sample["id"], client)
    cancel_event = threading.Event()
    with _lock:
        _cancel_events[draft.draft_id] = cancel_event

    http_request_id = telemetry.request_id_var.get() or telemetry.new_request_id()

    log.info("draft.created", extra={
        "draft_id": draft.draft_id,
        "request_id": http_request_id,
        "client": client,
        "sample_id": sample.get("id"),
        "fresh": fresh,
    })

    ctx = contextvars.copy_context()
    t = threading.Thread(
        target=ctx.run,
        args=(_run_draft_job, draft, sample, cancel_event, fresh, http_request_id),
        daemon=True,
    )
    t.start()
    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову роботу зупинитися."""
    with _lock:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise KeyError(f"Чернетку '{draft_id}' не знайдено")
        if draft.status in TERMINAL:
            raise ValueError(f"Чернетка вже завершена зі статусом '{draft.status}'")

        event = _cancel_events.get(draft_id)
        if event is not None:
            event.set()

        if draft.status == "queued":
            draft.status = "cancelled"
            draft.updated_at = _now()
            emit(draft, "cancelled", {})

    return draft
