"""Скрипт автоматизованого порівняння трьох конфігурацій моделі для ПР4.
Підтримує збереження проміжних результатів та автоматичну обробку Rate Limit (429).
"""

import json
import os
import re
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import openai

from app.schema import output_schema, validate, SchemaValidationError
from app.llm import SYSTEM_INSTRUCTION, estimate_tokens, fit_budget

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

COMPARE_DIR = Path(__file__).parent
REQUESTS_FILE = COMPARE_DIR / "requests.json"
CONTEXT_FILE = COMPARE_DIR.parent / "context.md"
OUTPUT_FILE = COMPARE_DIR / "results.json"

client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=45)
context = CONTEXT_FILE.read_text(encoding="utf-8")
schema = output_schema()
response_format_schema = {
    "type": "json_schema",
    "json_schema": {"name": "support_response", "schema": schema},
}


def call_llm_with_retry(messages, response_format=None, max_retries=6):
    """Виклик моделі з обробкою ліміту 5 запитів/хвилину (Rate Limit 429)."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            }
            if response_format:
                kwargs["response_format"] = response_format

            t0 = time.perf_counter()
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            return resp, elapsed
        except openai.RateLimitError as exc:
            # Шукаємо час затримки з повідомлення про помилку або чекаємо 20 секунд
            err_text = str(exc)
            delay = 21.0
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_text, re.IGNORECASE)
            if match:
                delay = float(match.group(1)) + 1.0
            print(f" [RateLimit 429] Очікування {delay:.1f}с перед повторною спробою ({attempt+1}/{max_retries})...", flush=True)
            time.sleep(delay)
        except Exception as exc:
            print(f" [Помилка API ({type(exc).__name__})]: {exc}. Пауза 5с...", flush=True)
            time.sleep(5)
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("Перевищено кількість повторних спроб через Rate Limit")


def run_config_1_fuzzy(text: str) -> dict:
    """Нечіткий запит: однорядкова інструкція, без правил, без історії, без схеми."""
    messages = [
        {"role": "system", "content": "Ти — помічник служби підтримки. Відповідай на запитання клієнта."},
        {"role": "user", "content": text},
    ]
    resp, elapsed = call_llm_with_retry(messages, response_format=None)
    raw = resp.choices[0].message.content or ""
    usage = resp.usage
    return {
        "raw": raw,
        "elapsed": round(elapsed, 2),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }


def run_config_2_structured(text: str) -> dict:
    """Структурований prompt: повна інструкція з прикладами і схемою, без правил і без історії."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": text},
    ]
    resp, elapsed = call_llm_with_retry(messages, response_format=response_format_schema)
    raw = resp.choices[0].message.content or ""
    usage = resp.usage

    passed_schema = True
    parsed = None
    try:
        parsed = validate(raw, conversation_text=text)
    except SchemaValidationError:
        passed_schema = False

    return {
        "raw": raw,
        "parsed": parsed,
        "passed_schema": passed_schema,
        "elapsed": round(elapsed, 2),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }


def run_config_3_context(text: str, history: list[dict]) -> dict:
    """Context engineering: інструкція + context.md + історія в межах бюджету + схема."""
    base_text = SYSTEM_INSTRUCTION + context + text
    base_tokens = estimate_tokens(base_text)
    fitted_history = fit_budget(history, budget=3000, base_tokens=base_tokens)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "system", "content": f"ПРАВИЛА ОБСЛУГОВУВАННЯ МАГАЗИНУ:\n{context}"},
    ]
    for turn in fitted_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": text})

    user_full_text = " ".join(t["content"] for t in history if t["role"] == "user") + " " + text

    resp, elapsed = call_llm_with_retry(messages, response_format=response_format_schema)
    raw = resp.choices[0].message.content or ""
    usage = resp.usage

    passed_schema = True
    parsed = None
    try:
        parsed = validate(raw, conversation_text=user_full_text)
    except SchemaValidationError:
        passed_schema = False

    return {
        "raw": raw,
        "parsed": parsed,
        "passed_schema": passed_schema,
        "elapsed": round(elapsed, 2),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }


def save_results(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    print(f"Запуск порівняння трьох конфігурацій (модель: {MODEL})...\n", flush=True)
    with open(REQUESTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Завантажуємо існуючі результати, якщо є (для відновлення)
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {"single_requests": [], "dialogues": []}
    else:
        results = {"single_requests": [], "dialogues": []}

    # 1. Одиночні звернення
    print("--- 1. Одиночні звернення ---", flush=True)
    existing_single_kinds = {item["kind"] for item in results.get("single_requests", [])}
    for item in data.get("звернення", []):
        kind = item.get("вид", "")
        text = item.get("текст", "")
        if kind in existing_single_kinds:
            print(f" [Пропуск] {kind} вже виконано", flush=True)
            continue

        print(f"\n[Запит: {kind}] '{text}'", flush=True)
        print("  -> Конфігурація 1 (нечітка)...", end="", flush=True)
        r1 = run_config_1_fuzzy(text)
        print(f" OK ({r1['elapsed']}s)", flush=True)
        time.sleep(14)

        print("  -> Конфігурація 2 (структурована)...", end="", flush=True)
        r2 = run_config_2_structured(text)
        print(f" OK ({r2['elapsed']}s)", flush=True)
        time.sleep(14)

        print("  -> Конфігурація 3 (context engineering)...", end="", flush=True)
        r3 = run_config_3_context(text, history=[])
        print(f" OK ({r3['elapsed']}s)", flush=True)
        time.sleep(14)

        results["single_requests"].append({
            "kind": kind,
            "text": text,
            "config_1_fuzzy": r1,
            "config_2_structured": r2,
            "config_3_context": r3,
        })
        save_results(results)

    # 2. Діалоги
    print("\n--- 2. Діалоги ---", flush=True)
    existing_dialogue_ids = {d["dialogue_id"] for d in results.get("dialogues", [])}
    for d_idx, d_item in enumerate(data.get("діалоги", []), 1):
        if d_idx in existing_dialogue_ids:
            print(f" [Пропуск] Діалог #{d_idx} вже виконано", flush=True)
            continue

        kind = d_item.get("вид", "")
        check = d_item.get("перевіряє", "")
        replicas = d_item.get("репліки", [])
        print(f"\n[Діалог #{d_idx}] {kind} ({len(replicas)} реплік)", flush=True)

        diag_res = {
            "dialogue_id": d_idx,
            "kind": kind,
            "checks": check,
            "turns": [],
        }

        history_c3 = []

        for turn_idx, turn_text in enumerate(replicas, 1):
            print(f"  Репліка {turn_idx}: '{turn_text}'", flush=True)
            print("    c1...", end="", flush=True)
            r1 = run_config_1_fuzzy(turn_text)
            print(f" OK ({r1['elapsed']}s); ", end="", flush=True)
            time.sleep(14)

            print("c2...", end="", flush=True)
            r2 = run_config_2_structured(turn_text)
            print(f" OK ({r2['elapsed']}s); ", end="", flush=True)
            time.sleep(14)

            print("c3...", end="", flush=True)
            r3 = run_config_3_context(turn_text, history=list(history_c3))
            print(f" OK ({r3['elapsed']}s)", flush=True)
            time.sleep(14)

            history_c3.append({"role": "user", "content": turn_text})
            assistant_reply = r3.get("parsed", {}).get("reply", "") if r3.get("parsed") else r3["raw"]
            history_c3.append({"role": "assistant", "content": assistant_reply})

            diag_res["turns"].append({
                "turn": turn_idx,
                "text": turn_text,
                "config_1_fuzzy": r1,
                "config_2_structured": r2,
                "config_3_context": r3,
            })
            save_results(results)

        results["dialogues"].append(diag_res)
        save_results(results)

    print(f"\nВсі тести успішно завершено! Результати збережено у {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
