"""Модуль ембедінгів: перетворення тексту на вектори локальною моделлю.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають внутрішньої реалізації моделі.
"""

import os
import threading
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

_model = None
_lock = threading.Lock()


def get_model():
    """Повернути готову до роботи модель (сінглтон)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def _is_e5_model(name: str = MODEL_NAME) -> bool:
    """Чи є модель сімейства E5 (вимагає префікси passage: та query:)."""
    return "e5" in name.lower()


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Перетворити тексти фрагментів на нормалізовані вектори (N × D)."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_model()
    # E5 вимагає префікс 'passage: ' для документів
    if _is_e5_model():
        formatted_texts = [f"passage: {t}" for t in texts]
    else:
        formatted_texts = texts

    vectors = model.encode(
        formatted_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на нормалізований вектор (розмірність D)."""
    model = get_model()
    # E5 вимагає префікс 'query: ' для запитів
    if _is_e5_model():
        formatted_text = f"query: {text}"
    else:
        formatted_text = text

    vector = model.encode(
        formatted_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vec = np.asarray(vector, dtype=np.float32)
    # Гарантуємо 1D форму (D,)
    if vec.ndim > 1:
        vec = vec.squeeze()
    return vec
