"""Конвеєр відповіді за базою знань: від питання до відповіді з джерелами."""

import time
from dataclasses import dataclass, field

from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source, retrieve, build_context
from . import llm


@dataclass
class Answer:
    """Результат конвеєра — те, з чим працює веб-рівень."""
    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None = None,
    filters: dict | None = None,
) -> Answer:
    """Відповісти на питання за базою знань."""

    # 1. Пошук фрагментів у базі знань
    t0_ret = time.perf_counter()
    hits = retrieve(question, index, keyword_index, filters)
    t_ret = time.perf_counter() - t0_ret

    # Якщо після відбору не лишилося жодного фрагмента — модель викликати не потрібно
    if not hits:
        return Answer(
            text="У базі знань онлайн-магазину «Сузірʼя» немає інформації за вашим запитом.",
            found=False,
            sources=[],
            retrieved=[],
            model=llm.MODEL,
            elapsed={"retrieval": round(t_ret, 3), "generation": 0.0},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # 2. Збирання контексту в межах бюджету
    context_str, context_sources = build_context(hits)

    # 3. Запит до мовної моделі
    llm_result = llm.ask(question, context_str)

    parsed = llm_result["parsed"]
    model_name = llm_result["model"]
    t_gen = llm_result["generation_time"]
    usage = llm_result["usage"]

    raw_answer_text = parsed.get("answer", "")
    raw_found = bool(parsed.get("found", False))
    cited_refs = parsed.get("sources", [])

    # 4. Перевірка посилань на джерела кодом
    # Створюємо словник доступних джерел за їхнім ref
    source_map = {s.ref: s for s in context_sources}

    # Залишаємо лише ті джерела, які дійсно були в контексті й названі моделлю
    valid_sources: list[Source] = []
    for ref in cited_refs:
        if isinstance(ref, int) and ref in source_map:
            valid_sources.append(source_map[ref])

    # Коригуємо ознаку 'found':
    # Якщо модель заявила found=True, але не надала жодного валідного джерела — це підозріло (хибне посилання / вигадка)
    actual_found = raw_found and (len(valid_sources) > 0)

    # Якщо found == False або відсутні валідні джерела, а модель видала стандартну відповідь відмови
    if not actual_found and not valid_sources:
        if "немає інформації" not in raw_answer_text.lower():
            raw_answer_text = "У базі знань немає інформації з цього приводу."

    return Answer(
        text=raw_answer_text,
        found=actual_found,
        sources=valid_sources,
        retrieved=hits,
        model=model_name,
        elapsed={
            "retrieval": round(t_ret, 3),
            "generation": round(t_gen, 3),
        },
        usage=usage,
    )
