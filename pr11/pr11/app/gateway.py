"""Шлюз до моделі: єдиний шлях, яким застосунок звертається до
провайдера.
"""

import logging
import os
import queue
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from . import budget, cache, costs
from .providers.base import Chunk, Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()

log = logging.getLogger("app.gateway")

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("LLM_API_KEY", "")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)


def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """Зібрати провайдера за назвою."""
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAICompatProvider(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            timeout=TIMEOUT,
        )
    return OpenAICompatProvider(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=name,
        timeout=TIMEOUT,
    )


def _stream_attempt(
    provider: Provider,
    request: Request,
    attempt_deadline: float,
    total_deadline: float,
    is_cancelled: Callable[[], bool],
    on_delta: Callable[[str], None],
) -> tuple[str, dict | None, str | None, float | None]:
    """Виконати одну спробу streaming з контролем таймауту бездіяльності та скасування."""
    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def worker() -> None:
        try:
            for chunk in provider.stream(request):
                if stop_event.is_set():
                    break
                q.put((chunk, None))
            q.put((None, None))
        except Exception as exc:
            q.put((None, exc))

    producer = threading.Thread(target=worker, daemon=True)
    producer.start()

    accumulated: list[str] = []
    final_usage: dict | None = None
    final_finish_reason: str | None = None
    start_time = time.time()
    last_chunk_time = start_time
    first_token_time: float | None = None

    while True:
        if is_cancelled():
            stop_event.set()
            raise ProviderError("cancelled", "Генерацію скасовано користувачем", retryable=False)

        now = time.time()
        if now >= total_deadline:
            stop_event.set()
            raise ProviderError(
                "timeout",
                f"Вичерпано загальний ліміт часу на генерацію ({TOTAL_TIMEOUT:.0f} с)",
                retryable=True,
            )

        if now >= attempt_deadline:
            stop_event.set()
            raise ProviderError(
                "timeout",
                f"Час очікування відповіді від моделі вичерпано ({TIMEOUT:.0f} с)",
                retryable=True,
            )

        wait_time = min(IDLE_TIMEOUT, attempt_deadline - now, total_deadline - now, 0.2)
        if wait_time <= 0:
            wait_time = 0.05

        try:
            item, err = q.get(timeout=wait_time)
        except queue.Empty:
            if time.time() - last_chunk_time >= IDLE_TIMEOUT:
                stop_event.set()
                raise ProviderError(
                    "timeout",
                    f"Перервано через відсутність нових фрагментів довше ніж {IDLE_TIMEOUT:.0f} с",
                    retryable=True,
                )
            continue

        if err is not None:
            stop_event.set()
            if isinstance(err, ProviderError):
                raise err
            raise ProviderError("unavailable", str(err), retryable=True)

        if item is None:
            break

        last_chunk_time = time.time()
        if item.text:
            if first_token_time is None:
                first_token_time = last_chunk_time
            accumulated.append(item.text)
            on_delta(item.text)
        if item.usage:
            final_usage = item.usage
        if item.finish_reason:
            final_finish_reason = item.finish_reason

    ttft = (first_token_time - start_time) if first_token_time else None
    return "".join(accumulated), final_usage, final_finish_reason, ttft


def _sleep_with_cancel(seconds: float, is_cancelled: Callable[[], bool]) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if is_cancelled():
            return True
        time.sleep(min(0.1, max(0.01, end - time.time())))
    return is_cancelled()


def generate(
    request: Request,
    on_delta: Callable[[str], None],
    on_status: Callable[[dict], None],
    is_cancelled: Callable[[], bool],
) -> Result:
    """Згенерувати відповідь з підтримкою кешу, бюджету, структурованого журналу та спроб."""
    meta = request.metadata or {}
    draft_id = meta.get("draft_id", "")
    request_id = meta.get("request_id_header") or meta.get("request_id", "")
    client_id = meta.get("client", "anonymous")
    prompt_version = meta.get("prompt_version", "drafts-v1")
    is_fresh = bool(meta.get("fresh", False))

    cache_key = cache.make_key(request, prompt_version)

    # 1. Перевірка кешу
    if not is_fresh:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            log.info("cache.hit", extra={
                "draft_id": draft_id,
                "request_id": request_id,
                "cache_key": cache_key,
                "provider": cached_result.provider,
                "model": cached_result.model,
            })
            on_status({
                "status": "running",
                "attempt": 1,
                "provider": cached_result.provider,
                "model": cached_result.model,
                "cached": True,
                "fallback": False,
            })
            # Передаємо збережений текст порціями або цілком
            on_delta(cached_result.text)
            return cached_result
    else:
        cache.record_bypass()
        log.info("cache.bypass", extra={
            "draft_id": draft_id,
            "request_id": request_id,
            "cache_key": cache_key,
        })

    # 2. Очікування слота одночасності (семафор)
    t_enter = time.time()
    total_deadline = t_enter + TOTAL_TIMEOUT

    acquired = False
    while time.time() < total_deadline:
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано", retryable=False)
        if _semaphore.acquire(timeout=0.2):
            acquired = True
            break

    queue_wait_ms = round((time.time() - t_enter) * 1000, 1)

    if not acquired:
        log.error("llm.queue_timeout", extra={
            "draft_id": draft_id,
            "request_id": request_id,
            "queue_wait_ms": queue_wait_ms,
        })
        raise ProviderError(
            "timeout",
            f"Не вдалося отримати слот генерації: черга заповнена ({MAX_CONCURRENT})",
            retryable=True,
        )

    try:
        primary_provider = make_provider(PROVIDER)
        current_provider = primary_provider
        last_error: ProviderError | None = None
        is_fallback = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано", retryable=False)

            # Перевірка бюджету перед платним викликом
            try:
                budget.check()
            except budget.BudgetExceeded as be:
                log.warning("budget.exceeded", extra={
                    "draft_id": draft_id,
                    "request_id": request_id,
                    "spent": be.spent,
                    "limit": be.limit,
                })
                raise ProviderError(
                    "budget",
                    f"Денний ліміт витрат {be.limit:.2f} $ вичерпано. Відновиться {be.resets_at}.",
                    retryable=False,
                )

            now = time.time()
            if now >= total_deadline:
                raise ProviderError(
                    "timeout",
                    f"Вичерпано загальний ліміт часу ({TOTAL_TIMEOUT:.0f} с)",
                    retryable=True,
                )

            attempt_deadline = now + min(TIMEOUT, total_deadline - now)
            attempt_start = time.time()

            on_status({
                "status": "running",
                "attempt": attempt,
                "provider": current_provider.name,
                "model": getattr(current_provider, "model", MODEL),
                "fallback": is_fallback,
                "cached": False,
            })

            try:
                text, usage, finish_reason, ttft = _stream_attempt(
                    current_provider,
                    request,
                    attempt_deadline,
                    total_deadline,
                    is_cancelled,
                    on_delta,
                )
                attempt_duration_ms = round((time.time() - attempt_start) * 1000, 1)
                cost, cost_est = costs.estimate(
                    usage,
                    provider=current_provider.name,
                    model=getattr(current_provider, "model", MODEL),
                    text=text,
                )
                budget.record(cost)

                log.info("llm.attempt", extra={
                    "draft_id": draft_id,
                    "request_id": request_id,
                    "client": client_id,
                    "attempt": attempt,
                    "provider": current_provider.name,
                    "model": getattr(current_provider, "model", MODEL),
                    "prompt_version": prompt_version,
                    "queue_wait_ms": queue_wait_ms,
                    "first_token_ms": round(ttft * 1000, 1) if ttft else None,
                    "duration_ms": attempt_duration_ms,
                    "status": "success",
                    "fallback": is_fallback,
                    "tokens": usage,
                    "cost": cost,
                    "cost_estimated": cost_est,
                    "cached": False,
                })

                result = Result(
                    text=text,
                    provider=current_provider.name,
                    model=getattr(current_provider, "model", MODEL),
                    usage=usage,
                    finish_reason=finish_reason,
                    elapsed=time.time() - t_enter,
                    cost=cost,
                    cost_estimated=cost_est,
                    cached=False,
                )

                # Зберігаємо в кеш
                cache.put(cache_key, result)
                return result

            except ProviderError as pe:
                attempt_duration_ms = round((time.time() - attempt_start) * 1000, 1)
                last_error = pe

                # Рахуємо вартість навіть невдалої спроби, якщо були токени або текст
                cost, cost_est = costs.estimate(
                    None,
                    provider=current_provider.name,
                    model=getattr(current_provider, "model", MODEL),
                    text=None,
                )
                budget.record(cost)

                lvl = logging.ERROR if (not pe.retryable or attempt == MAX_ATTEMPTS) else logging.WARNING
                log.log(lvl, "llm.attempt", extra={
                    "draft_id": draft_id,
                    "request_id": request_id,
                    "client": client_id,
                    "attempt": attempt,
                    "provider": current_provider.name,
                    "model": getattr(current_provider, "model", MODEL),
                    "prompt_version": prompt_version,
                    "queue_wait_ms": queue_wait_ms,
                    "duration_ms": attempt_duration_ms,
                    "status": "failed",
                    "kind": pe.kind,
                    "fallback": is_fallback,
                    "cost": cost,
                    "cost_estimated": cost_est,
                    "detail": pe.detail,
                })

                if pe.kind in ("cancelled", "budget") or is_cancelled():
                    raise pe
                if not pe.retryable:
                    break

                if attempt < MAX_ATTEMPTS:
                    backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) * random.uniform(0.5, 1.5)
                    if pe.retry_after is not None:
                        backoff = max(backoff, pe.retry_after)

                    if time.time() + backoff >= total_deadline:
                        break

                    on_status({
                        "status": "running",
                        "attempt": attempt + 1,
                        "provider": current_provider.name,
                        "fallback": is_fallback,
                        "reset": True,
                        "detail": f"Спроба {attempt} зазнала невдачі ({pe.kind}). Очікування {backoff:.1f} с перед повтором...",
                    })

                    cancelled = _sleep_with_cancel(backoff, is_cancelled)
                    if cancelled:
                        raise ProviderError("cancelled", "Генерацію скасовано", retryable=False)

        # Перехід на fallback
        if FALLBACK != "none" and not is_cancelled() and time.time() < total_deadline:
            is_fallback = True
            fallback_provider = make_provider(FALLBACK)
            log.warning("llm.fallback", extra={
                "draft_id": draft_id,
                "request_id": request_id,
                "fallback_provider": fallback_provider.name,
                "fallback_model": getattr(fallback_provider, "model", FALLBACK),
            })
            on_status({
                "status": "running",
                "attempt": 1,
                "provider": fallback_provider.name,
                "model": getattr(fallback_provider, "model", FALLBACK),
                "fallback": True,
                "reset": True,
                "detail": f"Перехід на запасний варіант ({FALLBACK})",
            })

            now = time.time()
            attempt_deadline = now + min(TIMEOUT, total_deadline - now)
            attempt_start = time.time()
            try:
                text, usage, finish_reason, ttft = _stream_attempt(
                    fallback_provider,
                    request,
                    attempt_deadline,
                    total_deadline,
                    is_cancelled,
                    on_delta,
                )
                attempt_duration_ms = round((time.time() - attempt_start) * 1000, 1)
                cost, cost_est = costs.estimate(
                    usage,
                    provider=fallback_provider.name,
                    model=getattr(fallback_provider, "model", FALLBACK),
                    text=text,
                )
                budget.record(cost)

                log.info("llm.attempt", extra={
                    "draft_id": draft_id,
                    "request_id": request_id,
                    "client": client_id,
                    "attempt": 1,
                    "provider": fallback_provider.name,
                    "model": getattr(fallback_provider, "model", FALLBACK),
                    "prompt_version": prompt_version,
                    "queue_wait_ms": queue_wait_ms,
                    "first_token_ms": round(ttft * 1000, 1) if ttft else None,
                    "duration_ms": attempt_duration_ms,
                    "status": "success",
                    "fallback": True,
                    "tokens": usage,
                    "cost": cost,
                    "cost_estimated": cost_est,
                })

                return Result(
                    text=text,
                    provider=fallback_provider.name,
                    model=getattr(fallback_provider, "model", FALLBACK),
                    usage=usage,
                    finish_reason=finish_reason,
                    elapsed=time.time() - t_enter,
                    cost=cost,
                    cost_estimated=cost_est,
                    cached=False,
                )
            except ProviderError as fbe:
                last_error = fbe

        if last_error is not None:
            raise last_error

        raise ProviderError(
            "unavailable",
            "Не вдалося згенерувати відповідь після всіх спроб",
            retryable=True,
        )
    finally:
        _semaphore.release()
