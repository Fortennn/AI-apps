"""Контракт відповіді моделі: що саме вона має повернути і як це перевіряється."""

import json
from jsonschema import validate as json_schema_validate, ValidationError


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі."""
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Текст відповіді клієнту українською мовою."
            },
            "found": {
                "type": "boolean",
                "description": "Чи знайдено відповідь у наданих фрагментах бази знань (True якщо є факт, False якщо немає)."
            },
            "sources": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Список номерів джерел [ref], використаних для формулювання відповіді."
            }
        },
        "required": ["answer", "found", "sources"],
        "additionalProperties": False
    }


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути розпарсений словник."""
    cleaned = raw.strip()
    # Якщо модель обгорнула в ```json ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Невалідна JSON відповідь від моделі: {exc}") from exc

    try:
        json_schema_validate(instance=data, schema=output_schema())
    except ValidationError as exc:
        raise ValueError(f"Відповідь не відповідає JSON-схемі: {exc.message}") from exc

    return data
