"""Шлюз до моделі: єдиний шлях, яким застосунок звертається до
провайдера.
"""

import os
import queue
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()

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

# Параметри генерації — спільні для всіх провайдерів.
MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)


def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу. Ключа тут
    немає і бути не може."""
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
    """Зібрати провайдера за назвою: `openai` — з налаштувань доступу в
    `.env`, `mock` — підставний. Інша назва — та сама адреса й ключ, але
    інша модель (для запасного варіанту)."""
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAICompatProvider(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            timeout=TIMEOUT,
        )
    # Якщо передано назву іншої моделі як запасний варіант
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
) -> tuple[str, dict | None, str | None]:
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
    last_chunk_time = time.time()

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
            # Стрім завершено успішно
            break

        last_chunk_time = time.time()
        if item.text:
            accumulated.append(item.text)
            on_delta(item.text)
        if item.usage:
            final_usage = item.usage
        if item.finish_reason:
            final_finish_reason = item.finish_reason

    return "".join(accumulated), final_usage, final_finish_reason


def _sleep_with_cancel(seconds: float, is_cancelled: Callable[[], bool]) -> bool:
    """Заснути на вказаний час з періодичною перевіркою скасування. Повертає True, якщо скасовано."""
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
    """Згенерувати відповідь, дотримуючись таймаутів, спроб, обмеження
    одночасності й запасного варіанту."""
    start_time = time.time()
    total_deadline = start_time + TOTAL_TIMEOUT

    # Контроль черги / обмеження одночасних звертань
    acquired = False
    while time.time() < total_deadline:
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано", retryable=False)
        if _semaphore.acquire(timeout=0.2):
            acquired = True
            break

    if not acquired:
        raise ProviderError(
            "timeout",
            f"Не вдалося отримати слот генерації: перевищено ліміт одночасних запитів ({MAX_CONCURRENT})",
            retryable=True,
        )

    try:
        primary_provider = make_provider(PROVIDER)
        current_provider = primary_provider
        last_error: ProviderError | None = None
        is_fallback = False

        # Основні спроби
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано", retryable=False)

            now = time.time()
            if now >= total_deadline:
                raise ProviderError(
                    "timeout",
                    f"Вичерпано загальний ліміт часу ({TOTAL_TIMEOUT:.0f} с)",
                    retryable=True,
                )

            attempt_deadline = now + min(TIMEOUT, total_deadline - now)
            on_status({
                "status": "running",
                "attempt": attempt,
                "provider": current_provider.name,
                "model": getattr(current_provider, "model", MODEL),
                "fallback": False,
            })

            try:
                text, usage, finish_reason = _stream_attempt(
                    current_provider,
                    request,
                    attempt_deadline,
                    total_deadline,
                    is_cancelled,
                    on_delta,
                )
                return Result(
                    text=text,
                    provider=current_provider.name,
                    model=getattr(current_provider, "model", MODEL),
                    usage=usage,
                    finish_reason=finish_reason,
                    elapsed=time.time() - start_time,
                )
            except ProviderError as pe:
                last_error = pe
                if pe.kind == "cancelled" or is_cancelled():
                    raise pe
                if not pe.retryable:
                    # Помилки автентифікації, некоректні запити тощо — повторювати заборонено
                    break

                if attempt < MAX_ATTEMPTS:
                    # Розрахунок паузи з випадковою часткою (exponential backoff + jitter)
                    backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) * random.uniform(0.5, 1.5)
                    if pe.retry_after is not None:
                        backoff = max(backoff, pe.retry_after)

                    if time.time() + backoff >= total_deadline:
                        # Пауза перевищує загальний бюджет часу — припиняємо
                        break

                    on_status({
                        "status": "running",
                        "attempt": attempt + 1,
                        "provider": current_provider.name,
                        "fallback": False,
                        "reset": True,
                        "detail": f"Спроба {attempt} зазнала невдачі ({pe.kind}). Очікування {backoff:.1f} с перед повтором...",
                    })

                    cancelled = _sleep_with_cancel(backoff, is_cancelled)
                    if cancelled:
                        raise ProviderError("cancelled", "Генерацію скасовано", retryable=False)

        # Якщо основний провайдер не впорався, перевіряємо запасний варіант (fallback)
        if FALLBACK != "none" and not is_cancelled() and time.time() < total_deadline:
            is_fallback = True
            fallback_provider = make_provider(FALLBACK)
            on_status({
                "status": "running",
                "attempt": 1,
                "provider": fallback_provider.name,
                "model": getattr(fallback_provider, "model", FALLBACK),
                "fallback": True,
                "reset": True,
                "detail": f"Перехід на запасний варіант ({FALLBACK}) після невдачі основного",
            })

            now = time.time()
            attempt_deadline = now + min(TIMEOUT, total_deadline - now)
            try:
                text, usage, finish_reason = _stream_attempt(
                    fallback_provider,
                    request,
                    attempt_deadline,
                    total_deadline,
                    is_cancelled,
                    on_delta,
                )
                return Result(
                    text=text,
                    provider=fallback_provider.name,
                    model=getattr(fallback_provider, "model", FALLBACK),
                    usage=usage,
                    finish_reason=finish_reason,
                    elapsed=time.time() - start_time,
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
