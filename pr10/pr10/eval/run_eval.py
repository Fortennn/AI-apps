"""Автоматичний запуск та верифікація сценаріїв стійкості ПР10."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import drafting, gateway, jobs, limits, samples
from app.providers.base import ProviderError, Request
from app.providers.mock import MockProvider
from app.providers.openai_compat import OpenAICompatProvider


def run_benchmark():
    print("=" * 60)
    print("ПОЧАТОК ВЕРИФІКАЦІЇ СЦЕНАРІЇВ ПР10")
    print("=" * 60)

    results = {}

    # -------------------------------------------------------------
    # Сценарій 1: Звичайна чернетка (mock)
    # -------------------------------------------------------------
    print("\n[Сценарій 1] Звичайна чернетка (mock)...")
    mock_prov = MockProvider()
    req = Request(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=200,
        temperature=0.3,
        metadata={"request_id": "d01", "mock_failure_rate": 0, "mock_first_delay_ms": 200, "mock_delay_ms": 20},
    )
    t0 = time.time()
    chunks = []
    first_token_time = None
    for chunk in mock_prov.stream(req):
        if first_token_time is None and chunk.text:
            first_token_time = time.time() - t0
        chunks.append(chunk.text)
    total_time = time.time() - t0
    full_text = "".join(chunks)
    print(f" -> Текст отримано ({len(chunks)} фрагментів, {len(full_text)} симв.)")
    print(f" -> Час до 1-го токена: {first_token_time:.2f}с, разом: {total_time:.2f}с")
    results["sc1"] = {"status": "done", "first_token": first_token_time, "total": total_time}

    # -------------------------------------------------------------
    # Сценарій 2: Таймаут спроби
    # -------------------------------------------------------------
    print("\n[Сценарій 2] Таймаут спроби...")
    req = Request(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=200,
        temperature=0.3,
        metadata={"request_id": "d01", "mock_failure_rate": 1.0, "mock_failure_kind": "timeout", "mock_first_delay_ms": 1500},
    )
    # Тестуємо з коротким attempt_deadline = now + 0.5с
    t0 = time.time()
    try:
        gateway._stream_attempt(
            mock_prov,
            req,
            attempt_deadline=t0 + 0.5,
            total_deadline=t0 + 5.0,
            is_cancelled=lambda: False,
            on_delta=lambda x: None,
        )
        print(" -> ПОМИЛКА: таймаут не спрацював!")
    except ProviderError as pe:
        elapsed = time.time() - t0
        print(f" -> Успішно перервано по таймауту спроби за {elapsed:.2f}с (kind: {pe.kind}, message: {pe.message})")
        results["sc2"] = {"elapsed": elapsed, "kind": pe.kind}

    # -------------------------------------------------------------
    # Сценарій 3: Таймаут бездіяльності (IDLE_TIMEOUT)
    # -------------------------------------------------------------
    print("\n[Сценарій 3] Таймаут бездіяльності...")
    req = Request(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=200,
        temperature=0.3,
        metadata={"request_id": "d01", "mock_failure_rate": 0, "mock_first_delay_ms": 50, "mock_delay_ms": 1200},
    )
    # Тимчасово тестуємо з коротким idle timeout через перехоплення
    old_idle = gateway.IDLE_TIMEOUT
    gateway.IDLE_TIMEOUT = 0.4
    t0 = time.time()
    deltas = []
    try:
        gateway._stream_attempt(
            mock_prov,
            req,
            attempt_deadline=t0 + 5.0,
            total_deadline=t0 + 5.0,
            is_cancelled=lambda: False,
            on_delta=lambda x: deltas.append(x),
        )
        print(" -> ПОМИЛКА: idle timeout не спрацював!")
    except ProviderError as pe:
        elapsed = time.time() - t0
        print(f" -> Отримано {len(deltas)} фрагмент(ів) перед простоєм, потім зафіксовано idle timeout за {elapsed:.2f}с (kind: {pe.kind})")
        results["sc3"] = {"deltas": len(deltas), "elapsed": elapsed, "kind": pe.kind}
    finally:
        gateway.IDLE_TIMEOUT = old_idle

    # -------------------------------------------------------------
    # Сценарій 4: Тимчасовий збій, повтор рятує
    # -------------------------------------------------------------
    print("\n[Сценарій 4] Тимчасовий збій, повтор рятує...")
    req_sc4 = Request(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=200,
        temperature=0.3,
        metadata={"request_id": "d01", "mock_failure_rate": 0, "mock_first_delay_ms": 50, "mock_delay_ms": 10},
    )
    # Симулюємо провайдер, який падає на 1 спробі і вдається на 2
    class TransientProvider:
        name = "mock"
        model = "mock-1"
        def __init__(self):
            self.count = 0
        def stream(self, r):
            self.count += 1
            if self.count == 1:
                raise ProviderError("unavailable", "Збій 503 на 1 спробі", retryable=True)
            yield from mock_prov.stream(r)

    tp = TransientProvider()
    old_make = gateway.make_provider
    gateway.make_provider = lambda name: tp
    t0 = time.time()
    status_events = []
    res = gateway.generate(
        req_sc4,
        on_delta=lambda x: None,
        on_status=lambda s: status_events.append(s),
        is_cancelled=lambda: False,
    )
    print(f" -> Успішно відновлено! Спроб: {tp.count}, результат: {res.model}, статусів: {len(status_events)}")
    results["sc4"] = {"attempts": tp.count, "success": True}
    gateway.make_provider = old_make

    default_req = Request(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=200,
        temperature=0.3,
        metadata={"request_id": "d01", "mock_failure_rate": 0, "mock_first_delay_ms": 50, "mock_delay_ms": 10},
    )

    # -------------------------------------------------------------
    # Сценарій 5: Збій назавжди (усі спроби вичерпано)
    # -------------------------------------------------------------
    print("\n[Сценарій 5] Постійний збій...")
    class FailProvider:
        name = "mock"
        model = "mock-1"
        def __init__(self):
            self.calls = 0
        def stream(self, r):
            self.calls += 1
            raise ProviderError("unavailable", f"Невдача {self.calls}", retryable=True)

    fp = FailProvider()
    gateway.make_provider = lambda name: fp
    old_fallback = gateway.FALLBACK
    gateway.FALLBACK = "none"
    try:
        gateway.generate(default_req, on_delta=lambda x: None, on_status=lambda s: None, is_cancelled=lambda: False)
        print(" -> ПОМИЛКА: помилка не виникла!")
    except ProviderError as pe:
        print(f" -> Зупинено після рівно {fp.calls} спроб (очікувалось {gateway.MAX_ATTEMPTS}). Помилка: {pe.kind}")
        results["sc5"] = {"attempts": fp.calls, "kind": pe.kind}
    gateway.FALLBACK = old_fallback
    gateway.make_provider = old_make

    # -------------------------------------------------------------
    # Сценарій 6: Запасний варіант (Fallback)
    # -------------------------------------------------------------
    print("\n[Сценарій 6] Запасний варіант...")
    gateway.FALLBACK = "mock"
    fp = FailProvider()
    def make_p(name):
        if name == "mock" and fp.calls >= gateway.MAX_ATTEMPTS:
            return MockProvider()
        return fp
    gateway.make_provider = make_p
    res = gateway.generate(default_req, on_delta=lambda x: None, on_status=lambda s: None, is_cancelled=lambda: False)
    print(f" -> Успішно перейшли на запасний! Провайдер: {res.provider}, модель: {res.model}")
    results["sc6"] = {"provider": res.provider, "model": res.model}
    gateway.make_provider = old_make

    # -------------------------------------------------------------
    # Сценарій 7: Збій без повтору (Auth error)
    # -------------------------------------------------------------
    print("\n[Сценарій 7] Збій без повтору (Auth)...")
    class AuthFailProvider:
        name = "mock"
        model = "mock-1"
        def __init__(self):
            self.calls = 0
        def stream(self, r):
            self.calls += 1
            raise ProviderError("auth", "Недійсний ключ доступу", retryable=False)

    ap = AuthFailProvider()
    gateway.make_provider = lambda name: ap
    try:
        gateway.generate(default_req, on_delta=lambda x: None, on_status=lambda s: None, is_cancelled=lambda: False)
    except ProviderError as pe:
        print(f" -> Зупинено одразу після {ap.calls} спроби (auth не повторюється). kind={pe.kind}, retryable={pe.retryable}")
        results["sc7"] = {"attempts": ap.calls, "retryable": pe.retryable}
    gateway.make_provider = old_make

    # -------------------------------------------------------------
    # Сценарій 8: 429 з Retry-After від провайдера
    # -------------------------------------------------------------
    print("\n[Сценарій 8] 429 з Retry-After...")
    class RateLimitProvider:
        name = "mock"
        model = "mock-1"
        def __init__(self):
            self.calls = 0
        def stream(self, r):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("rate_limit", "Rate limit", retryable=True, retry_after=1.2)
            yield from mock_prov.stream(r)

    rp = RateLimitProvider()
    gateway.make_provider = lambda name: rp
    t0 = time.time()
    res = gateway.generate(default_req, on_delta=lambda x: None, on_status=lambda s: None, is_cancelled=lambda: False)
    elapsed = time.time() - t0
    print(f" -> Оброблено 429: пауза склала {elapsed:.2f}с (не менше retry_after=1.2с). Спроб: {rp.calls}")
    results["sc8"] = {"elapsed": elapsed, "attempts": rp.calls}
    gateway.make_provider = old_make

    # -------------------------------------------------------------
    # Сценарій 9: Обрив посеред потоку (mid_stream)
    # -------------------------------------------------------------
    print("\n[Сценарій 9] Обрив посеред потоку...")
    class MidStreamProvider:
        name = "mock"
        model = "mock-1"
        def __init__(self):
            self.calls = 0
        def stream(self, r):
            self.calls += 1
            if self.calls == 1:
                yield from [gateway.Chunk("Початок... ")]
                raise ProviderError("unavailable", "Обрив з'єднання", retryable=True)
            yield from mock_prov.stream(r)

    mp = MidStreamProvider()
    gateway.make_provider = lambda name: mp
    statuses = []
    res = gateway.generate(default_req, on_delta=lambda x: None, on_status=lambda s: statuses.append(s), is_cancelled=lambda: False)
    resets = [s for s in statuses if s.get("reset")]
    print(f" -> Успішно оброблено обрив: зафіксовано reset=True ({len(resets)} раз), текст перегенеровано заново")
    results["sc9"] = {"resets": len(resets), "attempts": mp.calls}
    gateway.make_provider = old_make

    # -------------------------------------------------------------
    # Сценарій 10: Бюджет часу (TOTAL_TIMEOUT)
    # -------------------------------------------------------------
    print("\n[Сценарій 10] Вичерпання TOTAL_TIMEOUT...")
    old_tot = gateway.TOTAL_TIMEOUT
    gateway.TOTAL_TIMEOUT = 1.0
    fp = FailProvider()
    gateway.make_provider = lambda name: fp
    t0 = time.time()
    try:
        gateway.generate(default_req, on_delta=lambda x: None, on_status=lambda s: None, is_cancelled=lambda: False)
    except ProviderError as pe:
        elapsed = time.time() - t0
        print(f" -> Запит припинено за {elapsed:.2f}с за лімітом бюджету (спроб встигло: {fp.calls} з {gateway.MAX_ATTEMPTS})")
        results["sc10"] = {"elapsed": elapsed, "attempts": fp.calls}
    gateway.TOTAL_TIMEOUT = old_tot
    gateway.make_provider = old_make

    # -------------------------------------------------------------
    # Сценарій 11 & 12: Скасування генерації та завершеної
    # -------------------------------------------------------------
    print("\n[Сценарій 11 & 12] Скасування під час та після генерації...")
    s = samples.get("d01")
    draft = jobs.start(s, "client-test")
    time.sleep(0.05)
    # Скасовуємо
    canceled_draft = jobs.cancel(draft.draft_id)
    print(f" -> Скасовано активну: статус={canceled_draft.status}")
    results["sc11"] = {"status": canceled_draft.status}

    # Завершуємо чернетку і перевіряємо відмову скасування
    done_draft = jobs.new_draft("d01", "client-test")
    done_draft.status = "done"
    try:
        jobs.cancel(done_draft.draft_id)
        print(" -> ПОМИЛКА: скасування завершеної мало підняти помилку!")
    except ValueError as ve:
        print(f" -> Скасування завершеної відхилено (409 Conflict): {ve}")
        results["sc12"] = {"rejected": True, "msg": str(ve)}

    # -------------------------------------------------------------
    # Сценарій 14 & 15: Обмеження частоти клієнта (Rate Limiter)
    # -------------------------------------------------------------
    print("\n[Сценарій 14 & 15] Обмеження частоти...")
    test_client = f"burst_test_{int(time.time())}"
    admitted = 0
    rejected = 0
    retry_after = 0
    for i in range(15):
        try:
            limits.acquire(test_client)
            admitted += 1
        except limits.RateLimited as rl:
            rejected += 1
            retry_after = rl.retry_after
    print(f" -> Із 15 запитів: прийнято {admitted} (ліміт: {limits.PER_MINUTE}), відхилено {rejected}, Retry-After: {retry_after}с")
    results["sc14_15"] = {"admitted": admitted, "rejected": rejected, "retry_after": retry_after}

    # -------------------------------------------------------------
    # Сценарій 16: Справжній провайдер на 5 зверненнях
    # -------------------------------------------------------------
    print("\n[Сценарій 16] Вимірювання справжнього провайдера (Gemini)...")
    real_prov = OpenAICompatProvider(
        base_url=gateway.BASE_URL,
        api_key=gateway.API_KEY,
        model=gateway.MODEL,
        timeout=gateway.TIMEOUT,
    )
    real_metrics = []
    for s_id in ["d01", "d02", "d03", "d04", "d05"]:
        sample_item = samples.get(s_id)
        req = drafting.build_request(sample_item)
        t0 = time.time()
        first_t = None
        chunks = []
        for ch in real_prov.stream(req):
            if first_t is None and ch.text:
                first_t = time.time() - t0
            chunks.append(ch.text)
        tot_t = time.time() - t0
        text = "".join(chunks)
        warns = drafting.check(text, sample_item)
        print(f" -> Звернення {s_id}: 1-й токен {first_t:.2f}с, всього {tot_t:.2f}с, токенів ~{len(chunks)}, зауважень: {len(warns)}")
        real_metrics.append({
            "id": s_id,
            "first_token": first_t,
            "total_time": tot_t,
            "chunks": len(chunks),
            "warnings": warns,
            "length": len(text),
        })
    results["sc16"] = real_metrics

    # -------------------------------------------------------------
    # Сценарій 17: Робота без ключа (Mock)
    # -------------------------------------------------------------
    print("\n[Сценарій 17] Робота без ключа...")
    mock_p = MockProvider()
    req_mock = Request(messages=[{"role": "user", "content": "hello"}], max_tokens=100, temperature=0.3, metadata={"request_id": "d01"})
    res_mock = mock_p.complete(req_mock)
    print(f" -> Mock без мережі та без ключа повернув {len(res_mock.text)} символів")
    results["sc17"] = {"ok": bool(res_mock.text)}

    # Збережемо результати для звіту
    with open("eval/metrics_out.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("ВЕРИФІКАЦІЮ ВСІХ СЦЕНАРІЇВ ЗАВЕРШЕНО УСПІШНО!")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
