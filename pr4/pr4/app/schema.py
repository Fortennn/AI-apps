"""Контракт відповіді помічника: що саме модель має повернути і як це
перевіряється.

Окремий модуль навмисно. Схема — це домовленість між моделлю і рештою
застосунку, і вона не залежить від провайдера: змінивши модель або спосіб
виклику, схему ви не змінюєте. Веб-рівень і сторінка працюють лише з тим,
що пройшло перевірку тут.
"""

import json
import logging
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Допустимі теми звернення для маршрутизації до відповідних відділів
ALLOWED_TOPICS = ("доставка", "оплата", "повернення", "гарантія", "замовлення", "інше")
TopicType = Literal["доставка", "оплата", "повернення", "гарантія", "замовлення", "інше"]


class AssistantResponse(BaseModel):
    """Схема структурованої відповіді помічника служби підтримки."""

    reply: str = Field(
        ...,
        description="Текст відповіді для клієнта зрозумілою та ввічливою українською мовою."
    )
    topic: TopicType = Field(
        ...,
        description="Тема звернення з фіксованого переліку категорій магазину."
    )
    based_on_rules: bool = Field(
        ...,
        description="Чи ґрунтується відповідь безпосередньо на правилах магазину (true), "
                    "чи правила про це мовчать або інформація відсутня (false)."
    )
    needs_clarification: bool = Field(
        ...,
        description="Чи потрібне уточнення від клієнта через неточність чи неповноту запиту."
    )
    escalate_to_operator: bool = Field(
        ...,
        description="Чи потрібно передати розмову оператору підтримки (нестандартні випадки, "
                    "відсутність інформації в правилах, скарги)."
    )
    order_number: Optional[str] = Field(
        default=None,
        description="Шестизначний номер замовлення (лише 6 цифр), якщо клієнт назвав його в розмові. "
                    "Якщо клієнт не називав номера замовлення — строго null."
    )

    @field_validator("order_number")
    @classmethod
    def validate_order_number(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v_clean = str(v).strip()
        if not v_clean or v_clean.lower() in ("null", "none", "—", "-"):
            return None
        # Номер замовлення за правилами — строго 6 цифр
        if not re.fullmatch(r"\d{6}", v_clean):
            # Якщо модель повернула щось на кшталт "№482913" чи "#482913", вилучимо цифри
            match = re.search(r"\b\d{6}\b", v_clean)
            if match:
                return match.group(0)
            raise ValueError(f"Номер замовлення має містити рівно 6 цифр, отримано: {v!r}")
        return v_clean


class SchemaValidationError(Exception):
    """Помилка валідації відповіді моделі за схемою."""


def output_schema() -> dict:
    """Повернути JSON Schema відповіді помічника.

    Ця сама схема передається моделі як опис очікуваного результату
    (через `response_format` або текстом в інструкції) і використовується
    для перевірки того, що повернулося.
    """
    return AssistantResponse.model_json_schema()


def _strip_json_codeblock(raw: str) -> str:
    """Очистити сирий текст від можливих маркерів ```json ... ```."""
    text = raw.strip()
    if text.startswith("```"):
        # Прибираємо початковий ``` або ```json
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def validate(raw: str, conversation_text: Optional[str] = None) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно.

    На вході — текст, який повернула модель. На виході — обʼєкт, що
    відповідає схемі. Якщо текст не є JSON або не проходить схем —
    підняти SchemaValidationError з поясненням.

    Крім перевірки форми (JSON Schema), виконується семантична перевірка:
    якщо модель заповнила order_number, перевіряємо, чи такий номер
    дійсно згадувався користувачем у розмові (захист від галюцинацій).
    """
    cleaned = _strip_json_codeblock(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(f"Відповідь моделі не є валідним JSON: {raw[:150]!r}")
        raise SchemaValidationError(f"Відповідь моделі не є валідним JSON: {exc}") from exc

    try:
        obj = AssistantResponse.model_validate(data)
    except Exception as exc:
        logger.warning(f"Відповідь моделі не пройшла перевірку схеми: {exc}; дані: {data}")
        raise SchemaValidationError(f"Помилка структури даних: {exc}") from exc

    result = obj.model_dump()

    # Семантична перевірка номера замовлення на вигадування (галлюцинацію)
    if result["order_number"] and conversation_text is not None:
        if result["order_number"] not in conversation_text:
            logger.warning(
                f"Модель вигадала номер замовлення {result['order_number']}, якого не було в розмові!"
            )
            # Коригуємо вигаданий номер, щоб не передавати хибні дані далі в систему
            result["order_number"] = None

    return result
