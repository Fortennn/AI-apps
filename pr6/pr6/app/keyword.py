"""Пошук за ключовими словами (BM25) — точка порівняння для семантичного.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжування.
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit, _matches_filters


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами."""

    chunks: list[Chunk]
    bm25: BM25Okapi | None = None
    extra: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова та токени для індексування й для запиту.

    Враховує:
    - регістронезалежність (lowercase);
    - українські та латинські літери й цифри;
    - апострофи в українських словах (звʼязок, пʼять);
    - артикули та коди (OR-X2-BLK, E3, AX3).
    """
    if not text:
        return []

    # Токенізуємо слова з урахуванням дефісів та апострофів
    tokens = re.findall(
        r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+(?:['’ʼ-][a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+)*",
        text.lower()
    )
    return tokens


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    if not chunks:
        return KeywordIndex(chunks=[])

    corpus = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    return KeywordIndex(chunks=chunks, bm25=bm25)


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту за допомогою алгоритму BM25."""
    if not index.chunks or index.bm25 is None:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Отримуємо оцінки BM25 для всіх фрагментів колекції
    doc_scores = index.bm25.get_scores(query_tokens)

    hits: list[Hit] = []
    for i, chunk in enumerate(index.chunks):
        if _matches_filters(chunk.metadata, filters):
            score = float(doc_scores[i])
            if score > 0:  # Включаємо лише результати, де є бодай один збіг
                hits.append(Hit(chunk=chunk, score=round(score, 4)))

    # Сортуємо за спаданням оцінки BM25
    hits.sort(key=lambda h: h.score, reverse=True)

    return hits[:top_k]
