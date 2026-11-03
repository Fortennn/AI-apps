import json
from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["return", "cancel", "status", "other"]
        },
        "order_id": {
            "type": ["string", "null"]
        },
        "item": {
            "type": ["string", "null"]
        },
        "reason": {
            "type": ["string", "null"],
            "enum": ["not_suitable", "defect", "wrong_item", "damaged", "other", None]
        },
        "reason_text": {
            "type": ["string", "null"]
        },
        "quantity": {
            "type": ["integer", "null"]
        },
        "has_multiple_actions": {
            "type": "boolean"
        }
    },
    "required": [
        "kind",
        "order_id",
        "item",
        "reason",
        "reason_text",
        "quantity",
        "has_multiple_actions"
    ],
    "additionalProperties": False
}


def request_schema() -> dict:
    return SCHEMA


def validate(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Порожня відповідь моделі")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc}") from exc
    try:
        jsonschema_validate(instance=data, schema=SCHEMA)
    except ValidationError as exc:
        raise ValueError(f"Помилка валідації схеми: {exc.message}") from exc
    return data
