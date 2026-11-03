"""Підставний провайдер: поводиться як справжній, але без мережі й
ключа — і збоїть тоді, коли ви просите.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")

SAMPLES_FILE = Path(__file__).parent.parent.parent / "samples" / "requests.json"


def _load_reply_examples() -> dict[str, str]:
    if not SAMPLES_FILE.exists():
        return {}
    try:
        data = json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))
        return {s["id"]: s.get("reply_example", "") for s in data if "id" in s}
    except Exception:
        return {}


_REPLY_EXAMPLES = _load_reply_examples()

DEFAULT_REPLY = (
    "Доброго дня! Дякуємо за звернення до служби підтримки «Сузірʼя». "
    "Ваш запит успішно зареєстровано в системі. Наші фахівці перевіряють інформацію "
    "і нададуть вичерпну відповідь найближчим часом. Бажаємо приємного дня!"
)


class MockProvider:
    """Провайдер без моделі: канонічні тексти, затримки, збої на вимогу."""

    name = "mock"
    model = "mock-1"

    def _get_text(self, request: Request) -> str:
        req_id = request.metadata.get("request_id")
        if req_id and req_id in _REPLY_EXAMPLES:
            text = _REPLY_EXAMPLES[req_id]
            if text:
                return text
        return DEFAULT_REPLY

    def _resolve_params(self, request: Request) -> tuple[int, int, float, str]:
        first_delay = request.metadata.get("mock_first_delay_ms", FIRST_DELAY_MS)
        delay = request.metadata.get("mock_delay_ms", DELAY_MS)
        failure_rate = request.metadata.get("mock_failure_rate", FAILURE_RATE)
        failure_kind = request.metadata.get("mock_failure_kind", FAILURE_KIND)
        return int(first_delay), int(delay), float(failure_rate), str(failure_kind)

    def _check_failure(self, kind: str) -> None:
        if kind == "rate_limit":
            raise ProviderError(
                kind="rate_limit",
                message="Перевищено ліміт запитів до підставного провайдера (429)",
                retryable=True,
                retry_after=2.0,
                detail="Mock simulated rate limit error with Retry-After: 2",
            )
        if kind == "auth":
            raise ProviderError(
                kind="auth",
                message="Помилка автентифікації ключа доступу (401)",
                retryable=False,
                detail="Mock simulated authentication failure",
            )
        if kind == "bad_request":
            raise ProviderError(
                kind="bad_request",
                message="Некоректний запит до моделі (400)",
                retryable=False,
                detail="Mock simulated bad request error",
            )
        if kind == "timeout":
            raise ProviderError(
                kind="timeout",
                message="Час очікування відповіді від моделі вичерпано",
                retryable=True,
                detail="Mock simulated timeout error",
            )
        # default: unavailable
        raise ProviderError(
            kind="unavailable",
            message="Підставний провайдер тимчасово недоступний (503)",
            retryable=True,
            detail="Mock simulated 503 Service Unavailable",
        )

    def complete(self, request: Request) -> Result:
        """Повернути відповідь цілком після затримки."""
        first_delay, delay, failure_rate, failure_kind = self._resolve_params(request)
        text = self._get_text(request)

        if failure_kind == "timeout" and failure_rate >= 1.0 and first_delay > 0:
            time.sleep(first_delay / 1000.0)
            self._check_failure("timeout")

        time.sleep(first_delay / 1000.0)

        if failure_rate > 0 and random.random() < failure_rate:
            self._check_failure(failure_kind)

        words = text.split(" ")
        time.sleep((len(words) * delay) / 1000.0)

        usage = {
            "prompt_tokens": 40,
            "completion_tokens": len(words),
            "total_tokens": 40 + len(words),
        }
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason="stop",
            elapsed=(first_delay + len(words) * delay) / 1000.0,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати текст фрагментами із затримкою."""
        first_delay, delay, failure_rate, failure_kind = self._resolve_params(request)
        text = self._get_text(request)
        will_fail = failure_rate > 0 and (random.random() < failure_rate)

        if first_delay > 0:
            time.sleep(first_delay / 1000.0)

        if will_fail and failure_kind != "mid_stream":
            self._check_failure(failure_kind)

        raw_words = text.split(" ")
        tokens = [w + " " for w in raw_words[:-1]] + [raw_words[-1]]

        for idx, token in enumerate(tokens):
            if will_fail and failure_kind == "mid_stream" and idx >= 3:
                raise ProviderError(
                    kind="unavailable",
                    message="Обрив з'єднання посеред передавання даних у режимі streaming",
                    retryable=True,
                    detail="Mock simulated mid-stream connection drop",
                )

            if idx > 0 and delay > 0:
                time.sleep(delay / 1000.0)

            yield Chunk(text=token)

        usage = {
            "prompt_tokens": 40,
            "completion_tokens": len(tokens),
            "total_tokens": 40 + len(tokens),
        }
        yield Chunk(text="", usage=usage, finish_reason="stop")
