"""Відбір фрагментів для моделі та збирання контексту."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from .documents import Chunk
from .index import Hit, SearchIndex, search as index_search
from .keyword import KeywordIndex
from . import embeddings

load_dotenv()

CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві."""
    ref: int
    chunk: Chunk
    score: float


def estimate_tokens(text: str) -> int:
    """Оцінити кількість токенів для тексту (приблизно 1 токен = ~3.5 символи для української)."""
    return max(1, len(text) // 3)


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None = None,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти, які варто показати моделі."""
    if not query.strip() or index is None:
        return []

    # 1. Рахуємо вектор запиту
    query_vec = embeddings.embed_query(query)

    # 2. Пошук у семантичному індексі (без порога, щоб поріг застосувати з перевіркою безпеки)
    raw_hits = index_search(index, query_vec, top_k=15, filters=filters, threshold=None)

    filtered_hits: list[Hit] = []
    seen_texts = set()

    for hit in raw_hits:
        chunk = hit.chunk
        meta = chunk.metadata or {}
        source_file = chunk.source.lower()

        # Безпековий запобіжник в коді (персонал / внутрішні документи)
        if meta.get("audience") == "персонал" or meta.get("status") == "внутрішній":
            continue
        if "оператора" in source_file or "внутрішня" in source_file:
            continue

        # Архівні документи відсікаються для клієнта
        if meta.get("status") == "архівний" or "2025" in source_file:
            continue

        # Поріг схожості
        if hit.score < SIMILARITY_THRESHOLD:
            continue

        # Дедуплікація повторюваних фрагментів
        cleaned_text = chunk.text.strip()
        if cleaned_text in seen_texts:
            continue
        seen_texts.add(cleaned_text)

        filtered_hits.append(hit)

    filtered_hits.sort(key=lambda h: h.score, reverse=True)
    return filtered_hits[:CONTEXT_CHUNKS]


def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    """Зібрати з влучень текст контексту для моделі та перелік джерел у межах токенного бюджету."""
    if not hits:
        return "", []

    sources: list[Source] = []
    context_parts: list[str] = []
    used_tokens = 0

    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        meta = chunk.metadata or {}

        title = meta.get("title", chunk.source)
        section = meta.get("section", "Основний розділ")
        updated = meta.get("updated", "")

        date_str = f" | Редакція від: {updated}" if updated else ""
        header = f"[Джерело {idx}] Документ: \"{title}\" | Розділ: \"{section}\"{date_str}"
        chunk_entry = f"{header}\nТекст:\n{chunk.text.strip()}\n"

        tokens = estimate_tokens(chunk_entry)
        if used_tokens + tokens > budget and sources:
            break

        sources.append(Source(ref=idx, chunk=chunk, score=hit.score))
        context_parts.append(chunk_entry)
        used_tokens += tokens

    context_str = "\n".join(context_parts)
    return context_str, sources
