"""Конвеєр обробки документа: від байтів файлу до рішення, що з ним робити."""

import time
from dataclasses import dataclass, field

from . import images, llm, rules
from .rules import Issue


@dataclass
class Result:
    """Результат конвеєра — те, з чим працює веб-рівень."""
    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def decide(document: dict, issues: list[Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими проблемами.
    
    Повертає пару: (decision, reasons).
    """
    if document is None:
        return "reject", ["Не вдалося прочитати дані з документа."]

    # 1. Якщо це не рахунок на оплату (наприклад, видаткова накладна) — reject
    type_issues = [i for i in issues if i.rule == "document_type"]
    if type_issues:
        return "reject", [i.message for i in type_issues]

    # 2. Якщо є помилки (severity == 'error') — review
    error_issues = [i for i in issues if i.severity == "error"]
    if error_issues:
        return "review", [f"[{i.rule}] {i.message}" for i in error_issues]

    # 3. Якщо є попередження (severity == 'warning') — review
    warn_issues = [i for i in issues if i.severity == "warning"]
    if warn_issues:
        return "review", [f"[{i.rule}] {i.message}" for i in warn_issues]

    # 4. Якщо все чисто — auto (у реєстр платежів)
    return "auto", ["Усі реквізити, розрахунки та суми перевірено, проблем не виявлено."]


def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    t0 = time.perf_counter()
    prep = images.prepare(content)
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    llm_res = llm.extract(prep)
    t_ext = time.perf_counter() - t1

    t2 = time.perf_counter()
    doc = llm_res["document"]
    issues = rules.check(doc)
    decision, reasons = decide(doc, issues)
    t_chk = time.perf_counter() - t2

    return Result(
        decision=decision,
        reasons=reasons,
        document=doc,
        issues=issues,
        image={
            "original": prep.original,
            "sent": prep.sent,
        },
        model=llm_res["model"],
        elapsed={
            "prepare": round(t_prep, 3),
            "extraction": round(t_ext, 3),
            "checks": round(t_chk, 4),
        },
        usage=llm_res["usage"],
    )
