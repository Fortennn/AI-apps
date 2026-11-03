"""Кеш відповідей: однакове питання — одна оплата.
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")

_lock = threading.Lock()
_entries: dict[str, dict] = {}
_stats = {"hits": 0, "misses": 0, "bypassed": 0}


def _load_disk() -> None:
    if not CACHE_FILE:
        return
    path = Path(CACHE_FILE)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        now = time.time()
        for k, v in data.get("entries", {}).items():
            if v.get("expires_at", 0) > now:
                _entries[k] = v
        stats_data = data.get("stats", {})
        _stats["hits"] = stats_data.get("hits", 0)
        _stats["misses"] = stats_data.get("misses", 0)
        _stats["bypassed"] = stats_data.get("bypassed", 0)
    except Exception:
        pass


def _save_disk() -> None:
    if not CACHE_FILE:
        return
    path = Path(CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = path.with_suffix(".tmp")
    data = {
        "entries": _entries,
        "stats": _stats,
    }
    try:
        temp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_file.replace(path)
    except Exception:
        pass


_load_disk()


def settings() -> dict:
    """Чинні налаштування — для сторінки й журналу."""
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def make_key(request: Request, prompt_version: str) -> str:
    """Ключ кешу для запиту: однакові запити — однаковий ключ, будь-яка
    відмінність у тому, що впливає на відповідь, — інший."""
    canonical_data = {
        "prompt_version": prompt_version or request.metadata.get("prompt_version", ""),
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    }
    payload = json.dumps(canonical_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str) -> Result | None:
    """Результат за ключем, якщо він є й не прострочений; інакше `None`."""
    if not ENABLED:
        return None

    with _lock:
        entry = _entries.get(key)
        if entry is None:
            _stats["misses"] += 1
            return None

        if time.time() > entry["expires_at"]:
            _entries.pop(key, None)
            _stats["misses"] += 1
            _save_disk()
            return None

        _stats["hits"] += 1
        res_data = entry["result"]
        return Result(
            text=res_data.get("text", ""),
            provider=res_data.get("provider", "cache"),
            model=res_data.get("model", ""),
            usage=res_data.get("usage"),
            finish_reason=res_data.get("finish_reason", "stop"),
            elapsed=0.0,
            cost=0.0,
            cost_estimated=False,
            cached=True,
        )


def put(key: str, result: Result) -> None:
    """Зберегти результат. Коли кеш вимкнено — нічого не робити."""
    if not ENABLED or result.cached:
        return

    # Не кешуємо порожній текст або збійні результати
    if not result.text:
        return

    with _lock:
        now = time.time()
        # Очищення старих записів при перевищенні ліміту
        if len(_entries) >= MAX_ENTRIES:
            oldest_key = min(_entries, key=lambda k: _entries[k]["created_at"])
            _entries.pop(oldest_key, None)

        res_dict = asdict(result)
        res_dict["cached"] = True
        res_dict["cost"] = 0.0

        _entries[key] = {
            "key": key,
            "result": res_dict,
            "created_at": now,
            "expires_at": now + TTL_SECONDS,
        }
        _save_disk()


def record_bypass() -> None:
    """Зафіксувати обхід кешу (fresh=True)."""
    with _lock:
        _stats["bypassed"] += 1


def stats() -> dict:
    """Лічильники для `GET /api/usage` і сторінки."""
    with _lock:
        now = time.time()
        active_entries = sum(1 for v in _entries.values() if v.get("expires_at", 0) > now)
        return {
            "hits": _stats["hits"],
            "misses": _stats["misses"],
            "bypassed": _stats["bypassed"],
            "entries": active_entries,
        }
