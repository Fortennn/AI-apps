"""Аналіз журналу: повільні, невдалі й повторні виклики, витрати, кеш.

Запуск із папки pr11:
    python eval/analyze.py                       # logs/app.jsonl, усі розділи
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    """Прочитати JSON lines."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    """p-й процентиль (0 ≤ p ≤ 1) за найближчим рангом; `None` для порожнього списку."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def summary(records: list[dict]) -> None:
    """Що взагалі є в журналі."""
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records if "ts" in r]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    """Розподіл тривалості та найповільніші виклики."""
    attempts = [r for r in records if r.get("event") == "llm.attempt" and "duration_ms" in r]
    if not attempts:
        print("Немає записів llm.attempt")
        return

    durations = [float(r["duration_ms"]) for r in attempts]
    med = percentile(durations, 0.5)
    p95 = percentile(durations, 0.95)
    max_d = max(durations)

    print(f"Звертань до LLM: {len(attempts)}")
    print(f"Тривалість: медіана {med:.1f} мс, p95 {p95:.1f} мс, максимум {max_d:.1f} мс")

    # Топ найповільніших
    sorted_attempts = sorted(attempts, key=lambda x: float(x.get("duration_ms", 0)), reverse=True)[:10]
    print("\nТоп найдовших звертань:")
    for a in sorted_attempts:
        d_ms = a.get("duration_ms", 0)
        q_ms = a.get("queue_wait_ms", 0)
        ttft = a.get("first_token_ms")
        ttft_str = f"{ttft:.1f} мс" if ttft is not None else "—"
        prov = a.get("provider", "?")
        mod = a.get("model", "?")
        st = a.get("status", "?")
        print(f"  {d_ms:>7.1f} мс | черга: {q_ms:>5.1f} мс | 1-й токен: {ttft_str:>8} | {prov}/{mod} (спроба {a.get('attempt')}, {st})")


def failed(records: list[dict]) -> None:
    """Невдалі спроби та збої."""
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    failed_attempts = [r for r in attempts if r.get("status") == "failed"]

    print(f"Всього звертань до LLM: {len(attempts)}, невдалих спроб: {len(failed_attempts)}")

    kinds = Counter(r.get("kind", "unknown") for r in failed_attempts)
    print("Збої за видами:")
    for k, count in kinds.most_common():
        print(f"  {k:<15}: {count}")

    # Скільки чернеток урятовано повторами
    draft_attempts = defaultdict(list)
    for a in attempts:
        d_id = a.get("draft_id")
        if d_id:
            draft_attempts[d_id].append(a)

    rescued = 0
    multi_attempt_drafts = 0
    for d_id, atts in draft_attempts.items():
        if len(atts) > 1:
            multi_attempt_drafts += 1
            if any(a.get("status") == "success" for a in atts):
                rescued += 1

    print(f"\nЧернеток із кількома спробами: {multi_attempt_drafts}")
    print(f"З них врятовано повторами (успіх): {rescued} ({rescued/multi_attempt_drafts*100:.1f}%)" if multi_attempt_drafts else "\nПовторів не було.")

    failed_cost = sum(float(r.get("cost", 0) or 0) for r in failed_attempts)
    print(f"Загальна вартість невдалих спроб: {failed_cost:.6f} $")


def repeated(records: list[dict]) -> None:
    """Повторні запити та ключі кешу."""
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    cache_hits = [r for r in records if r.get("event") == "cache.hit"]
    bypasses = [r for r in records if r.get("event") == "cache.bypass"]

    print(f"Влучань у кеш (повторних відповідей): {len(cache_hits)}")
    print(f"Обходів кешу (fresh=True): {len(bypasses)}")

    # Повторні звернення до того самого sample_id
    created = [r for r in records if r.get("event") == "draft.created"]
    sample_counts = Counter(r.get("sample_id") for r in created if r.get("sample_id"))
    print("\nНайчастіші звернення клієнтів:")
    for sid, count in sample_counts.most_common(5):
        print(f"  {sid}: {count} раз(ів)")


def spend(records: list[dict]) -> None:
    """Витрати та токени."""
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    total_cost = sum(float(r.get("cost", 0) or 0) for r in attempts)

    by_model = defaultdict(float)
    tokens_by_model = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
    for a in attempts:
        m = a.get("model", "unknown")
        c = float(a.get("cost", 0) or 0)
        by_model[m] += c
        toks = a.get("tokens") or {}
        if isinstance(toks, dict):
            tokens_by_model[m]["prompt"] += toks.get("prompt_tokens", 0) or 0
            tokens_by_model[m]["completion"] += toks.get("completion_tokens", 0) or 0
            tokens_by_model[m]["total"] += toks.get("total_tokens", 0) or 0

    print(f"Загальні витрати за журналом: {total_cost:.6f} $")
    print("\nВитрати за моделями:")
    for m, c in by_model.items():
        t = tokens_by_model[m]
        print(f"  {m:<20}: {c:.6f} $ | токени: prompt={t['prompt']}, completion={t['completion']}, разом={t['total']}")

    done_drafts = [r for r in records if r.get("event") == "draft.done"]
    if done_drafts:
        costs_done = [float(r.get("cost", 0) or 0) for r in done_drafts]
        avg_cost = sum(costs_done) / len(costs_done)
        print(f"\nЗавершених чернеток: {len(done_drafts)}")
        print(f"Середня вартість чернетки: {avg_cost:.6f} $")

    estimated_count = sum(1 for r in attempts if r.get("cost_estimated"))
    print(f"Записів з розрахунковою оцінкою (без фактичного usage): {estimated_count}")


def cache_report(records: list[dict]) -> None:
    """Ефективність кешу."""
    cache_hits = [r for r in records if r.get("event") == "cache.hit"]
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    bypasses = [r for r in records if r.get("event") == "cache.bypass"]

    total_requests = len(cache_hits) + len(attempts)
    hit_rate = (len(cache_hits) / total_requests * 100) if total_requests else 0.0

    print(f"Всього звернень: {total_requests}")
    print(f"Влучань у кеш (Hits): {len(cache_hits)}")
    print(f"Викликів провайдера (Misses): {len(attempts)}")
    print(f"Обходів кешу (Bypasses): {len(bypasses)}")
    print(f"Частка влучань (Hit rate): {hit_rate:.1f}%")

    # Скільки зекономлено:
    # Кожне влучання зекономило приблизно середню вартість одного виклику
    avg_call_cost = (sum(float(r.get("cost", 0) or 0) for r in attempts) / len(attempts)) if attempts else 0.0003
    saved_money = len(cache_hits) * avg_call_cost
    print(f"Зекономлено звертань до LLM: {len(cache_hits)}")
    print(f"Орієнтовно заощаджено коштів: {saved_money:.6f} $")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        try:
            SECTIONS[name](records)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"(заготовка: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
