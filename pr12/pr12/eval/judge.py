"""Модель-суддя: оцінка вільного тексту за рубрикою з бінарних критеріїв.

Код перевіряє те, що формалізується: наявність номера, відсутність
промокоду, довжину, викликані інструменти. Тон, зрозумілість наступного
кроку, обґрунтованість фактами — не формалізуються, і для них у набір
беруть суддю: модель, якій дають відповідь системи, вхід, факти й
рубрику, а вердикт отримують **за схемою** — так/ні за кожним критерієм
із поясненням. Суддя — така сама AI-функція, як і та, що оцінюється:
зі своїм prompt, схемою, моделлю й помилками, тому його калібрують —
порівнюють із власними оцінками на десятку прикладів
(`eval/agreement.py`).

Готове тут: рубрика-заготовка `RUBRIC` (адаптуйте під свою інструкцію
чернетки), контракт вердикту `Verdict`, його JSON Schema для
`response_format` і розбір відповіді `parse_verdict`, що перевіряє її
за схемою. Заготовки — `build_messages` і `evaluate`. Рішення з
розділу 2 практичної роботи:

* якою моделлю судити: інша, ніж та, що генерує (самоуподобання), —
  `JUDGE_MODEL` у `pr12/.env`; температура 0;
* що суддя бачить: відповідь, звернення клієнта, факти, критерії — і
  чи бачить еталон (`reply_example`), якщо він є; критерії — лише ті,
  що в прикладі (`criteria`), або всю рубрику;
* як сформулювати інструкцію судді, щоб він оцінював зміст, а не
  довжину, і щоб позначав причину за кожним критерієм — міркування
  перед вердиктом підвищує згоду з людьми;
* що робити, коли суддя повернув не JSON або не всі критерії: повторити
  раз, провалити приклад із поясненням «суддя не відповів», але не
  зарахувати як пройдений;
* витрати: суддя — ще одне звертання на кожен приклад; скільки це додає
  до вартості прогону і де суддю можна замінити кодом.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "30"))

# Рубрика: ідентифікатор критерію → формулювання для судді. Кожен
# критерій бінарний. Відправна точка — адаптуйте під свою інструкцію
# чернетки й під те, що саме у вашому застосунку не перевіряє код.
RUBRIC = {
    "tone": "Текст ввічливий і діловий: звертання до клієнта, без повчань, без надмірних "
            "вибачень, без канцеляриту; його можна надіслати клієнтові без правок тону.",
    "next_step": "Клієнтові зрозуміло, що буде далі або що зробити йому: куди принести, "
                 "скільки чекати, що надійде окремо.",
    "grounded": "Кожне твердження про замовлення, строки, суми, адреси й умови є серед наданих "
                "фактів; нічого не додано від себе й нічого з фактів не перекручено.",
    "addresses_question": "Відповідь стосується саме того, про що запитав клієнт, а не "
                          "сусіднього питання.",
    "no_extra_promises": "Немає обіцянок, винятків, компенсацій, знижок і строків, яких немає у "
                         "фактах.",
    "refuses_properly": "Якщо це відмова — названо причину з фактів, без вигаданих альтернатив, "
                        "без звинувачень і без «як виняток».",
    "stays_on_topic": "Текст лишається в межах підтримки магазину і не виконує стороннє "
                      "прохання клієнта (вірш, переклад, порада поза темою).",
    "asks_clarification": "Коли фактів бракує або вони не стосуються питання, текст чесно каже, "
                          "що оператор уточнить, і не вигадує подробиць.",
}


@dataclass
class CriterionResult:
    id: str
    passed: bool
    reason: str = ""


@dataclass
class Verdict:
    """Вердикт судді: за кожним критерієм — так/ні з причиною; `overall` —
    підсумок судді (не обовʼязково збігається з «усі критерії пройдено»);
    `raw` — сира відповідь для журналу й розбору суперечок."""

    criteria: list[CriterionResult] = field(default_factory=list)
    overall: bool = False
    model: str | None = None
    usage: dict | None = None
    raw: str | None = None


def settings() -> dict:
    """Чинні налаштування судді — для звіту й перевірки середовища. Ключа
    тут немає."""
    return {
        "base_url": JUDGE_BASE_URL,
        "model": JUDGE_MODEL,
        "configured": bool(JUDGE_BASE_URL and JUDGE_API_KEY and JUDGE_MODEL),
    }


def verdict_schema() -> dict:
    """JSON Schema відповіді судді — для `response_format` і для перевірки
    того, що повернулося."""
    return {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "passed", "reason"],
                },
            },
            "overall": {"type": "boolean"},
        },
        "required": ["criteria", "overall"],
    }


def parse_verdict(raw: str, expected_ids: list[str]) -> Verdict:
    """Розібрати відповідь судді й перевірити її за схемою: JSON, потрібні
    поля, правильні типи, рівно ті критерії, які просили. Невідповідність
    — `ValueError` з поясненням."""
    clean = raw.strip()
    if "```" in clean:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if m:
            clean = m.group(1).strip()
    elif not clean.startswith("{") and "{" in clean:
        m = re.search(r"(\{[\s\S]*\})", clean)
        if m:
            clean = m.group(1).strip()

    try:
        data = json.loads(clean)
    except ValueError as exc:
        raise ValueError(f"відповідь судді — не JSON: {exc}") from exc
    if not isinstance(data, dict) or "criteria" not in data or "overall" not in data:
        raise ValueError("у відповіді судді немає полів criteria і overall")
    if not isinstance(data["overall"], bool):
        raise ValueError("overall має бути true або false")
    criteria: list[CriterionResult] = []
    for item in data["criteria"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) \
                or not isinstance(item.get("passed"), bool):
            raise ValueError(f"критерій не за схемою: {item!r}")
        criteria.append(CriterionResult(item["id"], item["passed"], str(item.get("reason", ""))))
    got = {c.id for c in criteria}
    missing = [cid for cid in expected_ids if cid not in got]
    if missing:
        raise ValueError("суддя не оцінив критерії: " + ", ".join(missing))
    unknown = sorted(got - set(expected_ids))
    if unknown:
        raise ValueError("суддя оцінив критерії, яких не просили: " + ", ".join(unknown))
    return Verdict(criteria=criteria, overall=data["overall"], raw=raw)


def build_messages(case: dict, output: dict, criteria: list[str]) -> list[dict]:
    """Скласти повідомлення для судді: інструкція з рубрикою й форматом
    відповіді, далі вхід прикладу (звернення, факти), відповідь системи.

    `case["input"]` — те, що отримала функція; `output["text"]` — що вона
    відповіла; `criteria` — ідентифікатори з `RUBRIC`, які оцінюються.
    Інструкція має сказати судді, що оцінювати лише за наданими фактами,
    кожен критерій окремо, і повернути JSON за схемою.
    """
    case_input = case.get("input", {})
    client_text = case_input.get("text") or case_input.get("question") or ""
    customer = case_input.get("customer") or case_input.get("customer_id") or ""
    facts = case_input.get("facts")
    if isinstance(facts, list):
        facts_text = "\n".join(f"- {f}" for f in facts) if facts else "Факти відсутні."
    elif facts:
        facts_text = str(facts)
    else:
        facts_text = "Факти відсутні."

    rubric_lines = [f'- "{cid}": {RUBRIC.get(cid, "Немає опису")}' for cid in criteria]
    criteria_block = "\n".join(rubric_lines)

    system_instruction = (
        "Ти — безсторонній та суворий суддя (LLM-as-a-judge) якості відповідей служби підтримки інтернет-магазину «Сузірʼя».\n"
        "Твоє завдання — оцінити відповідь системи ВИКЛЮЧНО за наведеними критеріями, враховуючи надані факти та звернення клієнта.\n\n"
        "Правила:\n"
        "1. Оцінюй зміст, фактичну точність і відповідність інструкції, а не особисті стилістичні вподобання.\n"
        "2. Якщо критерій вимагає, щоб твердження були у фактах — будь-яке вигадане або спотворене твердження означає passed: false.\n"
        "3. Оцінюй РІВНО ті критерії, які вказано нижче, не додавай і не пропускай жодного ID.\n"
        "4. Відповідай ВИКЛЮЧНО валідним JSON-об'єктом за такою схемою:\n"
        "{\n"
        '  "criteria": [\n'
        '    {"id": "<точно той самий id критерію>", "passed": true/false, "reason": "<коротке обґрунтування українською>"}\n'
        "  ],\n"
        '  "overall": true/false\n'
        "}\n"
        "Поле overall є true, тільки якщо ВСІ перелічені критерії отримали passed: true. Не пиши жодного тексту до чи після JSON."
    )

    user_content = (
        f"Звернення клієнта ({customer}):\n«{client_text}»\n\n"
        f"Надані факти (єдине джерело істини):\n{facts_text}\n\n"
        f"Відповідь системи для оцінки:\n«{output.get('text', '')}»\n\n"
        f"Оціни за наступними критеріями:\n{criteria_block}\n\n"
        "Поверни JSON з оцінкою."
    )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def evaluate(case: dict, output: dict, criteria: list[str] | None = None) -> Verdict:
    """Оцінити відповідь системи суддею за критеріями `criteria` (типово
    — уся рубрика). Повертає `Verdict`; якщо суддя не відповів за схемою
    після повтору — підняти виняток із поясненням (перевірка `judge`
    зарахує приклад як непройдений із цією причиною)."""
    if criteria is None:
        criteria = list(RUBRIC.keys())

    import openai

    client = openai.OpenAI(
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        timeout=JUDGE_TIMEOUT,
        max_retries=1,
    )
    messages = build_messages(case, output, criteria)
    schema_format = {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": verdict_schema()},
    }

    last_exc = None
    for attempt in range(2):
        try:
            try:
                completion = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=600,
                    response_format=schema_format,
                )
            except openai.BadRequestError:
                completion = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
            raw = completion.choices[0].message.content or ""
            usage = completion.usage.model_dump() if completion.usage else None
            verdict = parse_verdict(raw, criteria)
            verdict.model = JUDGE_MODEL
            verdict.usage = usage
            return verdict
        except Exception as exc:
            last_exc = exc

    raise ValueError(f"Суддя не повернув вердикт за схемою: {last_exc}")
