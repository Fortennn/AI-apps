"""Колекція документів: читання, метадані, поділ на фрагменти.

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс розширені поля (section, chunk_index).
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції."""
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def _split_section_text(body: str, max_size: int, overlap: int) -> list[str]:
    """Розбити довгий текст розділу на підфрагменти з перекриттям."""
    body = body.strip()
    if not body:
        return []
    if len(body) <= max_size:
        return [body]

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks = []
    current_para = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) + 2 <= max_size:
            current_para.append(p)
            current_len += len(p) + 2
        else:
            if current_para:
                chunks.append("\n\n".join(current_para))
            if len(p) > max_size:
                # Довгий параграф ділимо вікном
                start = 0
                while start < len(p):
                    end = start + max_size
                    chunks.append(p[start:end])
                    if end >= len(p):
                        break
                    start += max_size - overlap
                current_para = []
                current_len = 0
            else:
                current_para = [p]
                current_len = len(p)

    if current_para:
        chunks.append("\n\n".join(current_para))

    return chunks


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити текст документа на фрагменти зі збереженням контексту.

    Стратегія поділу:
    1. Розбиття за заголовками рівня ## (розділи).
    2. Якщо розділ перевищує CHUNK_SIZE — додатковий поділ за абзацами з перекриттям.
    3. Контекстне збагачення: до кожного фрагмента додається заголовок документа та розділу,
       щоб фрагмент був самодостатнім для моделі ембедінгів і користувача.
    """
    doc_title = metadata.get("title", source)
    chunks: list[Chunk] = []

    # Видаляємо головний заголовок # (якщо він є на початку)
    lines = text.strip().splitlines()
    body_lines = []
    for line in lines:
        if line.startswith("# ") and not body_lines:
            continue
        body_lines.append(line)
    clean_text = "\n".join(body_lines).strip()

    # Розбиваємо за секціями "## "
    section_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    sections = []
    last_pos = 0
    last_title = "Загальна інформація"

    for match in section_pattern.finditer(clean_text):
        section_body = clean_text[last_pos:match.start()].strip()
        if section_body:
            sections.append((last_title, section_body))
        last_title = match.group(1).strip()
        last_pos = match.end()

    trailing_body = clean_text[last_pos:].strip()
    if trailing_body:
        sections.append((last_title, trailing_body))

    if not sections and clean_text:
        sections.append(("Загальна інформація", clean_text))

    chunk_idx = 1
    for sec_title, sec_body in sections:
        sub_chunks = _split_section_text(sec_body, CHUNK_SIZE, CHUNK_OVERLAP)
        for part in sub_chunks:
            enriched_text = f"[{doc_title} | {sec_title}]\n{part}"
            chunk_meta = dict(metadata)
            chunk_meta["section"] = sec_title
            chunk_meta["chunk_index"] = chunk_idx
            chunk_meta["source"] = source

            chunks.append(Chunk(
                text=enriched_text,
                source=source,
                metadata=chunk_meta,
            ))
            chunk_idx += 1

    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
