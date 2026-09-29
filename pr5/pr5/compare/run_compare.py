"""Скрипт автоматизованого порівняння семантичного та ключового пошуку для ПР5.
Обчислює hit@1, hit@3, час пошуку та оцінки релевантності.
Також перевіряє «одну зміну» — семантичний пошук без префіксів E5 (no prefix).
"""

import json
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Ukrainian characters and symbols
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

# Додаємо корінь проєкту до sys.path
PR5_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PR5_DIR))

from app import embeddings, index, keyword

QUERIES_FILE = PR5_DIR / "compare" / "queries.json"
RESULTS_FILE = PR5_DIR / "compare" / "results.json"


def evaluate_query(search_fn, query: str, expected_doc: str, filters: dict | None, top_k: int = 5) -> dict:
    """Оцінити якість видачі для одного запиту."""
    t0 = time.perf_counter()
    hits = search_fn(query, top_k=top_k, filters=filters)
    elapsed = time.perf_counter() - t0

    if not hits:
        return {
            "position": None,
            "top1_source": None,
            "top1_score": None,
            "expected_score": None,
            "elapsed": round(elapsed, 4),
            "hits": [],
        }

    top1_source = hits[0].chunk.source
    top1_score = round(float(hits[0].score), 4)

    expected_pos = None
    expected_score = None

    if expected_doc:
        for pos, h in enumerate(hits, start=1):
            if h.chunk.source == expected_doc:
                expected_pos = pos
                expected_score = round(float(h.score), 4)
                break

    return {
        "position": expected_pos,
        "top1_source": top1_source,
        "top1_score": top1_score,
        "expected_score": expected_score,
        "elapsed": round(elapsed, 4),
        "hits": [
            {
                "pos": idx + 1,
                "source": h.chunk.source,
                "score": round(float(h.score), 4),
                "title": h.chunk.metadata.get("title", ""),
                "section": h.chunk.metadata.get("section", ""),
            }
            for idx, h in enumerate(hits[:3])
        ],
    }


def main():
    print("Завантаження індексів...")
    idx = index.load()
    kw_idx = keyword.build(idx.chunks)
    model = embeddings.get_model()

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    queries = queries_data.get("запити", [])
    print(f"Кількість тестових запитів: {len(queries)}")

    # Функції пошуку:
    # 1. Семантичний (з префіксами E5)
    def search_semantic(q, top_k=5, filters=None):
        q_vec = embeddings.embed_query(q)
        return index.search(idx, q_vec, top_k=top_k, filters=filters)

    # 2. Пошук за ключовими словами (BM25)
    def search_keyword(q, top_k=5, filters=None):
        return keyword.search(kw_idx, q, top_k=top_k, filters=filters)

    # 3. «Одна зміна»: Семантичний пошук БЕЗ префікса 'query: '
    def search_semantic_no_prefix(q, top_k=5, filters=None):
        raw_vec = model.encode(q, normalize_embeddings=True, show_progress_bar=False)
        vec = np.asarray(raw_vec, dtype=np.float32)
        return index.search(idx, vec, top_k=top_k, filters=filters)

    results = {
        "queries_count": len(queries),
        "runs": [],
        "metrics": {}
    }

    sem_hit1, sem_hit3 = 0, 0
    kw_hit1, kw_hit3 = 0, 0
    noprefix_hit1, noprefix_hit3 = 0, 0
    eval_queries_count = 0

    print("\n--- Запуск порівняльного тестування ---")

    for q_item in queries:
        q_id = q_item.get("id")
        q_kind = q_item.get("вид")
        q_text = q_item.get("запит")
        q_expected = q_item.get("очікуваний_документ", "")
        q_filter = q_item.get("фільтр")

        is_evaluable = bool(q_expected)
        if is_evaluable:
            eval_queries_count += 1

        print(f"[{q_id}/{len(queries)}] {q_kind}: '{q_text}' (очікується: {q_expected or '—'})")

        # 1. Semantic search
        res_sem = evaluate_query(search_semantic, q_text, q_expected, q_filter)
        if is_evaluable:
            if res_sem["position"] == 1:
                sem_hit1 += 1
            if res_sem["position"] and res_sem["position"] <= 3:
                sem_hit3 += 1

        # 2. Keyword search
        res_kw = evaluate_query(search_keyword, q_text, q_expected, q_filter)
        if is_evaluable:
            if res_kw["position"] == 1:
                kw_hit1 += 1
            if res_kw["position"] and res_kw["position"] <= 3:
                kw_hit3 += 1

        # 3. One change: Semantic without prefix
        res_noprefix = evaluate_query(search_semantic_no_prefix, q_text, q_expected, q_filter)
        if is_evaluable:
            if res_noprefix["position"] == 1:
                noprefix_hit1 += 1
            if res_noprefix["position"] and res_noprefix["position"] <= 3:
                noprefix_hit3 += 1

        results["runs"].append({
            "id": q_id,
            "kind": q_kind,
            "query": q_text,
            "expected_doc": q_expected,
            "filters": q_filter,
            "semantic": res_sem,
            "keyword": res_kw,
            "semantic_no_prefix": res_noprefix,
        })

    # Обчислюємо підсумкові метрики
    results["metrics"] = {
        "evaluable_queries": eval_queries_count,
        "semantic": {
            "hit_at_1": sem_hit1,
            "hit_at_3": sem_hit3,
            "hit_at_1_pct": round(sem_hit1 / eval_queries_count * 100, 1),
            "hit_at_3_pct": round(sem_hit3 / eval_queries_count * 100, 1),
        },
        "keyword": {
            "hit_at_1": kw_hit1,
            "hit_at_3": kw_hit3,
            "hit_at_1_pct": round(kw_hit1 / eval_queries_count * 100, 1),
            "hit_at_3_pct": round(kw_hit3 / eval_queries_count * 100, 1),
        },
        "semantic_no_prefix": {
            "hit_at_1": noprefix_hit1,
            "hit_at_3": noprefix_hit3,
            "hit_at_1_pct": round(noprefix_hit1 / eval_queries_count * 100, 1),
            "hit_at_3_pct": round(noprefix_hit3 / eval_queries_count * 100, 1),
        }
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Підсумкові результати ===")
    print(f"Запитів із відомою відповіддю: {eval_queries_count}")
    print(f"Семантичний пошук (E5 + prefixes):  Hit@1 = {sem_hit1}/{eval_queries_count} ({results['metrics']['semantic']['hit_at_1_pct']}%), Hit@3 = {sem_hit3}/{eval_queries_count} ({results['metrics']['semantic']['hit_at_3_pct']}%)")
    print(f"Пошук за словами (BM25):            Hit@1 = {kw_hit1}/{eval_queries_count} ({results['metrics']['keyword']['hit_at_1_pct']}%), Hit@3 = {kw_hit3}/{eval_queries_count} ({results['metrics']['keyword']['hit_at_3_pct']}%)")
    print(f"Зміна (Семантичний без префіксів):  Hit@1 = {noprefix_hit1}/{eval_queries_count} ({results['metrics']['semantic_no_prefix']['hit_at_1_pct']}%), Hit@3 = {noprefix_hit3}/{eval_queries_count} ({results['metrics']['semantic_no_prefix']['hit_at_3_pct']}%)")
    print(f"\nДетальні результати збережено у: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
