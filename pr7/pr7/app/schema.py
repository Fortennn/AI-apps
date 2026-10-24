"""Контракт відповіді моделі: опис очікуваних полів і перевірка форми."""

import json
import re
from decimal import Decimal, InvalidOperation
from jsonschema import validate as json_schema_validate, ValidationError


def output_schema() -> dict:
    """Повернути JSON Schema очікуваної відповіді моделі."""
    return {
        "type": "object",
        "properties": {
            "document_type": {
                "type": "string",
                "description": "Тип документа: 'рахунок' (або 'рахунок на оплату', 'видаткова накладна' тощо)"
            },
            "number": {
                "type": ["string", "null"],
                "description": "Номер документа (без назви 'Рахунок на оплату №')"
            },
            "date": {
                "type": ["string", "null"],
                "description": "Дата документа у форматі YYYY-MM-DD (наприклад, '2026-09-03')"
            },
            "valid_until": {
                "type": ["string", "null"],
                "description": "Строк дії або оплати до (формат YYYY-MM-DD або null, якщо не зазначено)"
            },
            "supplier": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"], "description": "ЄДРПОУ або РНОКПП (тільки цифри)"},
                    "iban": {"type": ["string", "null"], "description": "IBAN рахунок (без пробілів)"}
                },
                "required": ["name", "code", "iban"],
                "additionalProperties": False
            },
            "buyer": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"], "description": "ЄДРПОУ покупця"}
                },
                "required": ["name", "code"],
                "additionalProperties": False
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "unit": {"type": "string"},
                        "quantity": {"type": ["number", "integer"]},
                        "price": {"type": ["string", "number", "null"]},
                        "amount": {"type": ["string", "number", "null"]}
                    },
                    "required": ["name", "unit", "quantity", "price", "amount"],
                    "additionalProperties": False
                }
            },
            "total_without_vat": {
                "type": ["string", "number", "null"],
                "description": "Разом без ПДВ (рядок з двома знаками після крапки або null)"
            },
            "vat": {
                "type": ["string", "number", "null"],
                "description": "ПДВ (рядок з двома знаками після крапки або null)"
            },
            "total": {
                "type": ["string", "number", "null"],
                "description": "Усього до сплати (рядок з двома знаками після крапки або null)"
            }
        },
        "required": [
            "document_type",
            "number",
            "date",
            "valid_until",
            "supplier",
            "buyer",
            "items",
            "total_without_vat",
            "vat",
            "total"
        ],
        "additionalProperties": False
    }


def normalize_money(val: str | int | float | None) -> str | None:
    """Нормалізувати грошове значення до рядка виду '12570.00' без втрати копійок."""
    if val is None:
        return None
    s = str(val).strip().replace(" ", "").replace("\u00a0", "").replace(",", ".")
    if not s or s.lower() == "null":
        return None
    try:
        dec = Decimal(s)
        return f"{dec:.2f}"
    except InvalidOperation:
        return s


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й нормалізувати формати полів."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Невалідний JSON у відповіді моделі: {exc}") from exc

    try:
        json_schema_validate(instance=data, schema=output_schema())
    except ValidationError as exc:
        raise ValueError(f"Відповідь моделі не відповідає схемі: {exc.message}") from exc

    # Нормалізація полів
    # 1. Тип документа (нижній регістр)
    if data.get("document_type"):
        doc_t = data["document_type"].lower().strip()
        if "видаткова накладна" in doc_t or "накладна" in doc_t:
            data["document_type"] = "видаткова накладна"
        elif "рахунок" in doc_t:
            data["document_type"] = "рахунок"

    # 2. Номер і дати
    if data.get("number") is not None:
        data["number"] = str(data["number"]).strip()
    if data.get("date") is not None:
        data["date"] = str(data["date"]).strip()
    if data.get("valid_until") is not None:
        data["valid_until"] = str(data["valid_until"]).strip()

    # 3. Постачальник та покупець
    if data.get("supplier"):
        s = data["supplier"]
        if s.get("code") is not None:
            s["code"] = re.sub(r"\D", "", str(s["code"]))
        if s.get("iban") is not None:
            s["iban"] = str(s["iban"]).replace(" ", "").strip()
        if s.get("name") is not None:
            s["name"] = str(s["name"]).strip()

    if data.get("buyer"):
        b = data["buyer"]
        if b.get("code") is not None:
            b["code"] = re.sub(r"\D", "", str(b["code"]))
        if b.get("name") is not None:
            b["name"] = str(b["name"]).strip()

    # 4. Позиції
    if data.get("items"):
        for it in data["items"]:
            it["price"] = normalize_money(it.get("price"))
            it["amount"] = normalize_money(it.get("amount"))

    # 5. Підсумки
    data["total_without_vat"] = normalize_money(data.get("total_without_vat"))
    data["vat"] = normalize_money(data.get("vat"))
    data["total"] = normalize_money(data.get("total"))

    # Якщо документ без ПДВ (total_without_vat == total), а vat повернуто як None
    if data["total_without_vat"] is not None and data["total"] is not None:
        if data["total_without_vat"] == data["total"] and data["vat"] is None:
            data["vat"] = "0.00"

    return data
