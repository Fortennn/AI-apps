"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Індекс — це вектори всіх фрагментів, самі фрагменти з метаданими й назва
моделі, якою вектори отримано.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті."""

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх нормалізованих векторів."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Невідповідність кількості: {len(chunks)} фрагментів і {len(vectors)} векторів"
        )

    # Переконуємося у валідності векторів та їх нормалізації
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Уникаємо ділення на нуль
    norms[norms == 0] = 1.0
    normalized_vectors = vectors / norms

    return SearchIndex(
        chunks=chunks,
        vectors=normalized_vectors,
        model_name=model_name,
        extra={"count": len(chunks), "dim": vectors.shape[1] if len(vectors) > 0 else 0}
    )


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск."""
    path.mkdir(parents=True, exist_ok=True)

    # 1. Зберігаємо матрицю векторів
    np.save(path / "vectors.npy", index.vectors)

    # 2. Зберігаємо фрагменти з метаданими
    chunks_data = [
        {
            "text": chunk.text,
            "source": chunk.source,
            "metadata": chunk.metadata,
        }
        for chunk in index.chunks
    ]
    with open(path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    # 3. Зберігаємо метаінформацію (назву моделі тощо)
    meta = {
        "model_name": index.model_name,
        "count": len(index.chunks),
        "dim": index.vectors.shape[1] if len(index.vectors) > 0 else 0,
        "extra": index.extra,
    }
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска."""
    vectors_file = path / "vectors.npy"
    chunks_file = path / "chunks.json"
    meta_file = path / "meta.json"

    if not (vectors_file.exists() and chunks_file.exists() and meta_file.exists()):
        raise RuntimeError(
            f"Індекс не знайдено за шляхом '{path}'. Побудуйте його командою: python ingest.py"
        )

    vectors = np.load(vectors_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = [
        Chunk(
            text=item["text"],
            source=item["source"],
            metadata=item.get("metadata", {}),
        )
        for item in chunks_data
    ]

    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=meta.get("model_name", "unknown"),
        extra=meta.get("extra", {}),
    )


def _matches_filters(chunk_meta: dict, filters: dict | None) -> bool:
    """Перевірити, чи відповідають метадані фрагмента заданим фільтрам."""
    if not filters:
        return True

    for key, expected_value in filters.items():
        if expected_value is None or expected_value == "":
            continue
        actual_value = chunk_meta.get(key)
        if actual_value is None:
            return False
        # Регістронезалежне порівняння рядків
        if str(actual_value).strip().lower() != str(expected_value).strip().lower():
            return False
    return True


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі за косинусною схожістю до вектора запиту."""
    if len(index) == 0:
        return []

    # 1. Відбираємо індекси фрагментів, які проходять фільтри за метаданими
    valid_indices = []
    for i, chunk in enumerate(index.chunks):
        if _matches_filters(chunk.metadata, filters):
            valid_indices.append(i)

    if not valid_indices:
        return []

    # 2. Обчислюємо схожість для дозволених векторів
    # Оскільки вектори нормалізовані, косинусна схожість = скалярний добуток
    q_norm = np.linalg.norm(query_vector)
    if q_norm > 0:
        q_vec = query_vector / q_norm
    else:
        q_vec = query_vector

    candidate_vectors = index.vectors[valid_indices]
    scores = np.dot(candidate_vectors, q_vec)

    # 3. Формуємо список результатів із фільтрацією за порогом
    scored_hits = []
    for local_idx, orig_idx in enumerate(valid_indices):
        score = float(scores[local_idx])
        if threshold is not None and score < threshold:
            continue
        scored_hits.append(Hit(chunk=index.chunks[orig_idx], score=score))

    # 4. Сортуємо за спаданням оцінки схожості
    scored_hits.sort(key=lambda h: h.score, reverse=True)

    return scored_hits[:top_k]
