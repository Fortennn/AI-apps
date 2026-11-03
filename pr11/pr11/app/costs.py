"""Вартість виклику: токени → гроші.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Ціни за мільйон токенів, у доларах.
PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"


def settings() -> dict:
    """Чинні ціни — для сторінки й журналу."""
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def estimate(
    usage: dict | None,
    *,
    provider: str | None = None,
    model: str | None = None,
    text: str | None = None,
) -> tuple[float | None, bool]:
    """Оцінити вартість одного звертання до провайдера.

    `usage` — те, що повернув провайдер (може бути `None`); `provider` і
    `model` — хто відповідав; `text` — текст відповіді, якщо доведеться
    оцінювати без `usage`.

    Повертає пару: вартість у доларах і ознаку «це оцінка, а не розрахунок за фактичним `usage`».
    """
    if provider == "mock" and not MOCK_BILLABLE:
        return 0.0, False

    if usage and ("prompt_tokens" in usage or "completion_tokens" in usage):
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0

        cached_tokens = 0
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens") or 0

        uncached_input = max(0, prompt_tokens - cached_tokens)
        cost = (
            uncached_input * PRICE_INPUT
            + cached_tokens * PRICE_CACHED_INPUT
            + completion_tokens * PRICE_OUTPUT
        ) / 1_000_000.0
        return round(cost, 6), False

    # Якщо usage не прийшов (обрив потоку або провайдер без usage):
    # Оцінюємо за довжиною тексту відповіді + середня довжина вхідного промпту (~350 токенів)
    est_prompt = 350
    est_completion = max(1, int(len(text) / 2.8)) if text else 0
    cost = (est_prompt * PRICE_INPUT + est_completion * PRICE_OUTPUT) / 1_000_000.0
    return round(cost, 6), True
