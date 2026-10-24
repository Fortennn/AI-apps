"""Скрипт повного оцінювання конвеєра розбору рахунків на зразках."""

import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PR7_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PR7_ROOT))

from app import extraction, rules, schema

SAMPLES_DIR = PR7_ROOT / "samples"
EXPECTED_FILE = SAMPLES_DIR / "expected.json"


def normalize_val(v):
    if v is None:
        return None
    s = str(v).strip().replace("\u00a0", " ").replace("’", "ʼ").replace("'", "ʼ")
    # Якщо сума/число
    s_clean = s.replace(" ", "").replace(",", ".")
    try:
        from decimal import Decimal, InvalidOperation
        dec = Decimal(s_clean)
        # якщо було десяткове число
        if "." in s_clean or isinstance(v, float):
            return f"{dec:.2f}"
    except Exception:
        pass
    # Нормалізація схожих літер (гомогліфів Latin -> Cyrillic) для текстових назв
    return s.translate(str.maketrans("ABCEHIKMOPTX", "АВСЕНІКМОРТХ"))


def compare_fields(expected: dict, actual: dict, prefix=""):
    """Рекурсивне порівняння полів: правильно, помилка, пропуск, вигадка."""
    results = {}

    all_keys = set(expected.keys()) | set(actual.keys() if actual else [])

    for k in all_keys:
        path = f"{prefix}.{k}" if prefix else k
        exp_v = expected.get(k)
        act_v = actual.get(k) if actual else None

        if isinstance(exp_v, dict):
            sub_res = compare_fields(exp_v, act_v if isinstance(act_v, dict) else {}, path)
            results.update(sub_res)
        elif k == "items" and isinstance(exp_v, list):
            # Порівняння списку позицій
            act_items = act_v if isinstance(act_v, list) else []
            max_len = max(len(exp_v), len(act_items))
            for idx in range(max_len):
                exp_it = exp_v[idx] if idx < len(exp_v) else {}
                act_it = act_items[idx] if idx < len(act_items) else {}
                for field_name in ["name", "unit", "quantity", "price", "amount"]:
                    it_path = f"items.{idx}.{field_name}"
                    e_f = exp_it.get(field_name)
                    a_f = act_it.get(field_name)
                    ne_f = normalize_val(e_f)
                    na_f = normalize_val(a_f)

                    if ne_f is None and na_f is None:
                        label = "правильно"
                    elif ne_f is not None and na_f is None:
                        label = "пропуск"
                    elif ne_f is None and na_f is not None:
                        label = "вигадка"
                    elif ne_f == na_f:
                        label = "правильно"
                    else:
                        label = "помилка"
                    results[it_path] = {"expected": ne_f, "actual": na_f, "label": label}
        else:
            ne_v = normalize_val(exp_v)
            na_v = normalize_val(act_v)

            if ne_v is None and na_v is None:
                label = "правильно"
            elif ne_v is not None and na_v is None:
                label = "пропуск"
            elif ne_v is None and na_v is not None:
                label = "вигадка"
            elif ne_v == na_v:
                label = "правильно"
            else:
                label = "помилка"
            results[path] = {"expected": ne_v, "actual": na_v, "label": label}

    return results


def run_benchmark(config_name: str, output_file: str) -> dict:
    with open(EXPECTED_FILE, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Додаємо власні зразки
    own_samples = {
        "own/rahunok-02-scale0.5-jpeg40.jpg": {
            "source": "rahunok-02",
            "variant": "own",
            "note": "Власний зразок 1 (scale 0.5, jpeg 40)",
            "fields": all_samples["clean/rahunok-02.png"]["fields"],
            "expected_checks": [],
            "expected_decision": "auto"
        },
        "own/rahunok-04-photo.jpg": {
            "source": "rahunok-04",
            "variant": "own",
            "note": "Власний зразок 2 (contrast 0.7, blur 1.2, noise 15)",
            "fields": all_samples["clean/rahunok-04.png"]["fields"],
            "expected_checks": [],
            "expected_decision": "auto"
        }
    }
    all_samples.update(own_samples)

    print(f"\n========================================================")
    print(f"Запуск оцінювання: {config_name}")
    print(f"Кількість файлів для тестування: {len(all_samples)}")
    print(f"========================================================\n")

    eval_results = []
    variant_stats = {}

    for idx, (rel_path, exp_data) in enumerate(all_samples.items(), start=1):
        file_path = SAMPLES_DIR / rel_path
        if not file_path.exists():
            print(f"[!] Файл не знайдено: {file_path}")
            continue

        variant = exp_data.get("variant", "unknown")
        if variant not in variant_stats:
            variant_stats[variant] = {
                "count": 0, "correct": 0, "errors": 0, "misses": 0, "hallucinations": 0,
                "silent_errors": 0, "decision_match": 0, "prompt_tokens": 0, "caught_by_rules": 0
            }

        print(f"[{idx}/{len(all_samples)}] Обробка {rel_path} ({variant})...")
        with open(file_path, "rb") as f:
            content = f.read()

        res = extraction.process(content)

        # Порівняння полів
        field_eval = compare_fields(exp_data.get("fields", {}), res.document or {})

        total_correct = sum(1 for v in field_eval.values() if v["label"] == "правильно")
        total_errors = sum(1 for v in field_eval.values() if v["label"] == "помилка")
        total_misses = sum(1 for v in field_eval.values() if v["label"] == "пропуск")
        total_hallucinations = sum(1 for v in field_eval.values() if v["label"] == "вигадка")

        # Перевірка критичних полів (supplier.iban, supplier.code, total)
        crit_keys = ["supplier.iban", "supplier.code", "total"]
        crit_eval = {k: field_eval.get(k) for k in crit_keys if k in field_eval}
        crit_has_error = any(v["label"] in ("помилка", "вигадка") for v in crit_eval.values() if v)

        # Перевірка спрацювання правил
        actual_rule_names = list({i.rule for i in res.issues})
        exp_checks = exp_data.get("expected_checks", [])
        has_field_issue = (total_errors > 0 or total_hallucinations > 0)
        caught = len(res.issues) > 0 if has_field_issue else True

        # Тиха помилка: документ отримав рішення "auto", але містить помилку або вигадку
        is_silent_error = (res.decision == "auto" and has_field_issue)

        decision_matched = (res.decision == exp_data.get("expected_decision"))

        p_tokens = res.usage.get("prompt_tokens", 0) if res.usage else 0

        # Оновлення статистики
        st = variant_stats[variant]
        st["count"] += 1
        st["correct"] += total_correct
        st["errors"] += total_errors
        st["misses"] += total_misses
        st["hallucinations"] += total_hallucinations
        st["prompt_tokens"] += p_tokens
        if is_silent_error:
            st["silent_errors"] += 1
        if decision_matched:
            st["decision_match"] += 1
        if caught and has_field_issue:
            st["caught_by_rules"] += 1

        print(f"  -> Рішення: {res.decision} (очікувалось: {exp_data.get('expected_decision')})")
        print(f"  -> Поля: {total_correct} правильно, {total_errors} помилок, {total_misses} пропусків, {total_hallucinations} вигадок")
        print(f"  -> Проблеми: {actual_rule_names} | Тиха помилка: {is_silent_error}")

        eval_results.append({
            "file": rel_path,
            "variant": variant,
            "expected_decision": exp_data.get("expected_decision"),
            "actual_decision": res.decision,
            "decision_matched": decision_matched,
            "expected_checks": exp_checks,
            "actual_rules": actual_rule_names,
            "issues": [{"field": i.field, "rule": i.rule, "message": i.message, "severity": i.severity} for i in res.issues],
            "field_summary": {
                "correct": total_correct,
                "errors": total_errors,
                "misses": total_misses,
                "hallucinations": total_hallucinations,
                "silent_error": is_silent_error
            },
            "critical_fields": crit_eval,
            "field_eval": field_eval,
            "raw_document": res.document,
            "image": res.image,
            "elapsed": res.elapsed,
            "usage": res.usage
        })

    summary = {
        "config_name": config_name,
        "total_files": len(eval_results),
        "variant_stats": variant_stats,
        "total_silent_errors": sum(s["silent_errors"] for s in variant_stats.values()),
        "total_decision_matches": sum(s["decision_match"] for s in variant_stats.values()),
        "results": eval_results
    }

    out_path = PR7_ROOT / "eval" / output_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"ПІДСУМКОВА СТАТИСТИКА ЗА ВАРІАНТАМИ ({config_name}):")
    print("Варіант | Файлів | Правильно | Помилок | Пропусків | Вигадок | Тихих помилок | Збіг рішення")
    print("-" * 75)
    for var, s in variant_stats.items():
        print(f"{var:8s} | {s['count']:6d} | {s['correct']:9d} | {s['errors']:7d} | {s['misses']:9d} | {s['hallucinations']:7d} | {s['silent_errors']:13d} | {s['decision_match']}/{s['count']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_benchmark("Config A (Default: 1600px, PNG)", "results_config_A.json")
