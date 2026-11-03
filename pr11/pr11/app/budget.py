"""Денний ліміт витрат: скільки застосунок може витратити за добу.
"""

import datetime
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")

_lock = threading.Lock()
_state = {
    "day": datetime.date.today().isoformat(),
    "spent": 0.0,
}


def _get_resets_at() -> str:
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    return f"{tomorrow.isoformat()}T00:00:00"


def _load_disk() -> None:
    if not BUDGET_FILE:
        return
    path = Path(BUDGET_FILE)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        today = datetime.date.today().isoformat()
        if data.get("day") == today:
            _state["spent"] = float(data.get("spent", 0.0))
            _state["day"] = today
        else:
            _state["day"] = today
            _state["spent"] = 0.0
            _save_disk()
    except Exception:
        pass


def _save_disk() -> None:
    if not BUDGET_FILE:
        return
    path = Path(BUDGET_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = path.with_suffix(".tmp")
    try:
        temp_file.write_text(json.dumps(_state, indent=2), encoding="utf-8")
        temp_file.replace(path)
    except Exception:
        pass


_load_disk()


class BudgetExceeded(Exception):
    """Денний ліміт вичерпано. `spent`, `limit` — у доларах; `resets_at` —
    коли ліміт відновиться, у форматі ISO."""

    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(f"денний ліміт витрат {limit:.2f} $ вичерпано (витрачено {spent:.4f} $); "
                         f"відновиться {resets_at}")
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at


def settings() -> dict:
    """Чинні налаштування — для сторінки."""
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE}


def _ensure_today() -> None:
    today = datetime.date.today().isoformat()
    if _state["day"] != today:
        _state["day"] = today
        _state["spent"] = 0.0
        _save_disk()


def check(expected_cost: float | None = None) -> None:
    """Переконатися, що платний виклик ще дозволений; інакше підняти `BudgetExceeded`."""
    with _lock:
        _ensure_today()
        spent = _state["spent"]
        add = expected_cost or 0.0
        if (spent + add) >= DAILY_USD:
            raise BudgetExceeded(spent, DAILY_USD, _get_resets_at())


def record(cost: float | None) -> None:
    """Зарахувати вартість виконаного звертання до витрат дня."""
    if not cost or cost <= 0:
        return
    with _lock:
        _ensure_today()
        _state["spent"] += cost
        _save_disk()


def status() -> dict:
    """Стан для `GET /api/usage` і сторінки."""
    with _lock:
        _ensure_today()
        spent = _state["spent"]
        remaining = max(0.0, DAILY_USD - spent)
        return {
            "day": _state["day"],
            "spent": round(spent, 6),
            "limit": DAILY_USD,
            "remaining": round(remaining, 6),
            "warning": spent >= (DAILY_USD * WARN_AT),
            "exceeded": spent >= DAILY_USD,
            "resets_at": _get_resets_at(),
        }
