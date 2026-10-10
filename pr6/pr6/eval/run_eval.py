"""Скрипт оцінювання RAG-помічника на наборі тестових запитань."""

import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Додаємо корінь pr6 до python path
PR6_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PR6_ROOT))

from app import index, keyword, rag, llm


def run_benchmark(config_label: str, output_file: str) -> dict:
    questions_file = Path(__file__).parent / "questions.json"
    if not questions_file.exists():
        print(f"Помилка: файл {questions_file} не знайдено.")
        sys.exit(1)

    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("питання", [])

    print(f"Завантаження індексу для {config_label}...")
    search_index = index.load()
    kw_index = keyword.build(search_index.chunks)

    print(f"Початок оцінювання на {len(questions)} питаннях...")

    results = []
    retrieval_success_count = 0
    found_match_count = 0
    citation_valid_count = 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time_retrieval = 0.0
    total_time_generation = 0.0

    search_errors = 0
    model_errors = 0

    for idx, item in enumerate(questions, start=1):
        q_id = item.get("id", idx)
        q_text = item["питання"]
        expected_docs = item.get("очікувані_документи", [])
        expected_answer_facts = item.get("очікувана_відповідь", "")
        expected_found = item.get("очікується_відповідь", True)
        q_type = item.get("вид", "невідомий")
        filters = item.get("filters")

        print(f"\n[{idx}/{len(questions)}] [ID:{q_id}] ({q_type}) {q_text}")

        res = rag.answer(q_text, search_index, kw_index, filters=filters)

        # Аналіз пошуку: чи потрапили очікувані документи у контекст (retrieved)
        retrieved_sources = [h.chunk.source for h in res.retrieved]
        cited_sources = [s.chunk.source for s in res.sources]

        # Перевірка наявності хоча б одного очікуваного документа (якщо вони очікувались)
        if expected_docs:
            retrieval_hit = any(doc in retrieved_sources for doc in expected_docs)
        else:
            # Для питань поза базою — успіх пошуку це відсутність небажаних/внутрішніх документів
            retrieval_hit = len(retrieved_sources) == 0 or not any("внутрішня" in s for s in retrieved_sources)

        if retrieval_hit:
            retrieval_success_count += 1

        # Перевірка збігу статусу found
        found_matches = (res.found == expected_found)
        if found_matches:
            found_match_count += 1

        # Перевірка валідності посилань
        citation_valid = True
        for s in res.sources:
            if s.ref < 1 or s.ref > len(res.retrieved):
                citation_valid = False
                break
        if citation_valid:
            citation_valid_count += 1

        # Класифікація помилок
        error_type = "Ні"
        if not found_matches or (expected_docs and not retrieval_hit):
            if expected_docs and not retrieval_hit:
                error_type = "Помилка пошуку (не знайдено необхідний документ)"
                search_errors += 1
            elif expected_found and not res.found:
                error_type = "Помилка моделі (безпідставна відмова)"
                model_errors += 1
            elif not expected_found and res.found:
                error_type = "Помилка моделі (відповідь з голови / невизнана відсутність)"
                model_errors += 1
            else:
                error_type = "Помилка моделі (невідповідність фактам)"
                model_errors += 1

        prompt_tok = res.usage.get("prompt_tokens", 0) if res.usage else 0
        comp_tok = res.usage.get("completion_tokens", 0) if res.usage else 0
        total_prompt_tokens += prompt_tok
        total_completion_tokens += comp_tok

        t_ret = res.elapsed.get("retrieval", 0.0)
        t_gen = res.elapsed.get("generation", 0.0)
        total_time_retrieval += t_ret
        total_time_generation += t_gen

        print(f"  -> Знайдено: {res.found} (очікувалось: {expected_found}) | Посилання: {[s.ref for s in res.sources]}")
        print(f"  -> Документи контексту: {retrieved_sources[:3]}")
        print(f"  -> Час: пошук {t_ret:.3f}s, генерація {t_gen:.3f}s | Токени: {prompt_tok}+{comp_tok}")
        if error_type != "Ні":
            print(f"  [!] {error_type}")

        results.append({
            "id": q_id,
            "question": q_text,
            "type": q_type,
            "expected_docs": expected_docs,
            "expected_found": expected_found,
            "actual_found": res.found,
            "retrieved_sources": retrieved_sources,
            "cited_sources": cited_sources,
            "answer_text": res.text,
            "retrieval_hit": retrieval_hit,
            "error_type": error_type,
            "elapsed": res.elapsed,
            "usage": res.usage,
        })

    summary = {
        "config_label": config_label,
        "model": llm.MODEL,
        "total_questions": len(questions),
        "retrieval_accuracy": round(retrieval_success_count / len(questions), 3),
        "found_accuracy": round(found_match_count / len(questions), 3),
        "citation_validity": round(citation_valid_count / len(questions), 3),
        "search_errors": search_errors,
        "model_errors": model_errors,
        "avg_prompt_tokens": round(total_prompt_tokens / len(questions), 1),
        "avg_completion_tokens": round(total_completion_tokens / len(questions), 1),
        "avg_retrieval_time": round(total_time_retrieval / len(questions), 3),
        "avg_generation_time": round(total_time_generation / len(questions), 3),
        "results": results,
    }

    out_path = Path(__file__).parent / output_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"ПІДСУМОК {config_label}:")
    print(f"Точність пошуку (Retrieval Recall): {summary['retrieval_accuracy'] * 100:.1f}%")
    print(f"Точність статусу Found: {summary['found_accuracy'] * 100:.1f}%")
    print(f"Валідність посилань: {summary['citation_validity'] * 100:.1f}%")
    print(f"Помилки пошуку: {search_errors} | Помилки моделі: {model_errors}")
    print(f"Середні токени: prompt={summary['avg_prompt_tokens']}, completion={summary['avg_completion_tokens']}")
    print(f"Середній час: пошук={summary['avg_retrieval_time']}s, генерація={summary['avg_generation_time']}s")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_benchmark("Configuration A (Default: Top-4, Thresh 0.35)", "results_config_A.json")
