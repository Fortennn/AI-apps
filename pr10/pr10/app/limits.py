"""Обмеження частоти звертань до власного API.
"""

import math
import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
WINDOW_SECONDS = 60.0

_lock = threading.Lock()
_client_history: dict[str, list[float]] = {}


class RateLimited(Exception):
    """Ліміт вичерпано. `retry_after` — через скільки секунд пробувати."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


def settings() -> dict:
    """Чинні обмеження — для сторінки."""
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    """Зарахувати одне звертання клієнта або підняти `RateLimited`.

    Викликається веб-рівнем перед створенням кожної чернетки.
    Використовує алгоритм ковзного вікна (sliding window) тривалістю 60 секунд.
    """
    now = time.time()
    cutoff = now - WINDOW_SECONDS

    with _lock:
        # Очищення застарілих записів поточного клієнта
        history = _client_history.setdefault(client, [])
        valid_history = [t for t in history if t > cutoff]

        if len(valid_history) >= PER_MINUTE:
            oldest = valid_history[0]
            retry_after = max(1.0, math.ceil((oldest + WINDOW_SECONDS) - now))
            _client_history[client] = valid_history
            raise RateLimited(float(retry_after))

        valid_history.append(now)
        _client_history[client] = valid_history

        # Періодичне очищення старих клієнтів від витоку пам'яті
        if len(_client_history) > 500:
            stale_keys = [
                k for k, v in _client_history.items()
                if not v or v[-1] <= cutoff
            ]
            for k in stale_keys:
                _client_history.pop(k, None)
