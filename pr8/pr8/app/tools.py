"""Інструменти помічника: що модель може попросити виконати і як це перевіряється.

Єдине місце застосунку, яке знає і про модель, і про сервіс магазину (`shop/service.py`).
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import jsonschema

from shop import service


@dataclass
class Context:
    """Те, що відомо про сеанс незалежно від моделі."""

    customer_id: str


@dataclass
class ToolResult:
    """Підсумок одного виклику інструмента."""

    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


# Описи інструментів у форматі OpenAI / Gemini Tools
TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "Отримати перелік усіх замовлень поточного клієнта (номери, дати, статуси, суми та кількість позицій). Параметрів немає, клієнт визначається автоматично.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Отримати детальну інформацію про конкретне замовлення клієнта: склад товарів, статус виконання, службу доставки, номер ТТН (відправлення) та інформацію про оплату.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення (наприклад, '10458')",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Пошук товарів у каталозі магазину за назвою, категорією або описом.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Слова для пошуку (наприклад, 'роутер Альтаїр', 'телевізор Андромеда')",
                    },
                    "category": {
                        "type": "string",
                        "description": "Категорія товару (наприклад, 'навушники', 'мережеве обладнання')",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Максимальна ціна товару в гривнях",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальна кількість товарів у видачі (від 1 до 50)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Отримати повну картку товару за його артикулом (SKU): назва, характеристики, ціна, вага, гарантія, чи підлягає поверненню належної якості.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару (наприклад, 'ALT-AX3', 'ORN-X2', 'SIR-R10')",
                    },
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Перевірити наявність товару на складі за його артикулом (SKU) та дату очікуваного надходження, якщо його немає в наявності.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару (SKU)",
                    },
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delivery_quote",
            "description": "Розрахувати вартість і строки доставки набору товарів у місто України.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Місто доставки в Україні (наприклад, 'Львів', 'Київ', 'Полтава')",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["branch", "courier", "pickup"],
                        "description": "Спосіб доставки: 'branch' (відділення), 'courier' (курʼєр), 'pickup' (самовивіз зі складу в Кременчуці)",
                    },
                    "items": {
                        "type": "array",
                        "description": "Список товарів для доставки",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {
                                    "type": "string",
                                    "description": "Артикул товару",
                                },
                                "quantity": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "description": "Кількість одиниць",
                                },
                            },
                            "required": ["sku"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["city", "method", "items"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Створити заявку на повернення товару з отриманого замовлення клієнта.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення",
                    },
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару, який повертається",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["not_suitable", "defect", "wrong_item", "damaged"],
                        "description": "Причина: 'not_suitable' (товар належної якості не підійшов), 'defect' (виробничий дефект), 'wrong_item' (не той товар), 'damaged' (пошкоджено під час доставки)",
                    },
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Кількість одиниць (за замовчуванням 1)",
                    },
                    "comment": {
                        "type": "string",
                        "description": "Коментар або пояснення клієнта",
                    },
                },
                "required": ["order_id", "sku", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Скасувати замовлення клієнта. Можливо лише якщо замовлення ще в статусі 'Готується'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення для скасування",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
]

# Словник схем параметрів для швидкої валідації
TOOL_SCHEMAS: dict[str, dict] = {
    spec["function"]["name"]: spec["function"]["parameters"]
    for spec in TOOL_SPECS
}


def specs() -> list[dict]:
    """Повернути описи дозволених клієнтських інструментів."""
    return TOOL_SPECS


def _normalize_order_id(val: Any) -> str:
    """Нормалізувати номер замовлення (прибрати пробіли, символи № та #)."""
    s = str(val).strip()
    s = re.sub(r"[№#\s]", "", s)
    return s


def _sanitize_order(order: dict) -> dict:
    """Очистити запис замовлення від службових полів перед надсиланням моделі."""
    return {
        "order_id": order.get("order_id"),
        "created_at": order.get("created_at"),
        "status": order.get("status"),
        "status_label": order.get("status_label", order.get("status")),
        "items": order.get("items", []),
        "total": order.get("total"),
        "delivery": {
            "method": order.get("delivery", {}).get("method"),
            "carrier": order.get("delivery", {}).get("carrier"),
            "city": order.get("delivery", {}).get("city"),
            "point": order.get("delivery", {}).get("point"),
            "tracking": order.get("delivery", {}).get("tracking"),
            "delivered_at": order.get("delivery", {}).get("delivered_at"),
            "cost": order.get("delivery", {}).get("cost"),
        },
        "payment": {
            "method": order.get("payment", {}).get("method"),
            "paid": order.get("payment", {}).get("paid"),
            "card_last4": order.get("payment", {}).get("card_last4"),
        },
    }


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    """Перевірити й виконати виклик інструмента, запропонований моделлю."""
    # 1. Перевірка наявності дозволеного інструмента
    if name not in TOOL_SCHEMAS:
        return ToolResult(
            status="rejected",
            content={
                "error": True,
                "code": "forbidden_tool",
                "message": f"Інструмент '{name}' не існує або недоступний для клієнтського помічника.",
            },
            reason=f"невідомий або недозволений інструмент '{name}'",
            arguments=None,
        )

    # 2. Парсинг JSON аргументів
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
        if not isinstance(args, dict):
            return ToolResult(
                status="rejected",
                content={
                    "error": True,
                    "code": "invalid_arguments",
                    "message": "Аргументи виклику мають бути JSON-об'єктом.",
                },
                reason="аргументи не є об'єктом",
                arguments=None,
            )
    except Exception as exc:
        return ToolResult(
            status="rejected",
            content={
                "error": True,
                "code": "invalid_json",
                "message": f"Некоректний формат JSON в аргументах: {exc}",
            },
            reason="невалідний JSON аргументів",
            arguments=None,
        )

    # 3. Нормалізація специфічних полів перед валідацією схеми
    if "order_id" in args:
        args["order_id"] = _normalize_order_id(args["order_id"])

    # 4. Валідація за JSON Schema
    schema = TOOL_SCHEMAS[name]
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as err:
        return ToolResult(
            status="rejected",
            content={
                "error": True,
                "code": "schema_violation",
                "message": f"Аргументи не відповідають схемі: {err.message}",
            },
            reason=f"аргументи поза схемою: {err.message}",
            arguments=args,
        )

    # 5. Виконання виклику сервісу з контролем доступу (Security & Ownership)
    try:
        if name == "list_orders":
            # Клієнт береться ВИКЛЮЧНО з контексту сеансу
            orders = service.list_orders(ctx.customer_id)
            return ToolResult(
                status="ok",
                content=orders,
                arguments=args,
            )

        elif name == "get_order":
            order_id = args["order_id"]
            # Отримуємо замовлення
            order = service.get_order(order_id)
            # Перевіряємо право власності
            if order.get("customer_id") != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    content={
                        "error": True,
                        "code": "forbidden",
                        "message": f"Замовлення {order_id} не належить вашому обліковому запису.",
                    },
                    reason="чуже замовлення (відхилено кодом)",
                    arguments=args,
                )
            # Очищуємо від внутрішніх приміток
            clean_order = _sanitize_order(order)
            return ToolResult(
                status="ok",
                content=clean_order,
                arguments=args,
            )

        elif name == "search_products":
            results = service.search_products(
                query=args["query"],
                category=args.get("category"),
                max_price=args.get("max_price"),
                limit=args.get("limit", 10),
            )
            return ToolResult(
                status="ok",
                content=results,
                arguments=args,
            )

        elif name == "get_product":
            product = service.get_product(args["sku"])
            return ToolResult(
                status="ok",
                content=product,
                arguments=args,
            )

        elif name == "get_stock":
            stock = service.get_stock(args["sku"])
            return ToolResult(
                status="ok",
                content=stock,
                arguments=args,
            )

        elif name == "delivery_quote":
            quote = service.delivery_quote(
                city=args["city"],
                method=args["method"],
                items=args["items"],
            )
            return ToolResult(
                status="ok",
                content=quote,
                arguments=args,
            )

        elif name == "create_return":
            order_id = args["order_id"]
            # Перевіряємо приналежність замовлення
            order = service.get_order(order_id)
            if order.get("customer_id") != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    content={
                        "error": True,
                        "code": "forbidden",
                        "message": f"Замовлення {order_id} не належить вашому обліковому запису.",
                    },
                    reason="чуже замовлення (відхилено кодом)",
                    arguments=args,
                )
            res = service.create_return(
                order_id=order_id,
                sku=args["sku"],
                reason=args["reason"],
                quantity=args.get("quantity", 1),
                comment=args.get("comment", ""),
            )
            return ToolResult(
                status="ok",
                content=res,
                arguments=args,
            )

        elif name == "cancel_order":
            order_id = args["order_id"]
            # Перевіряємо приналежність замовлення
            order = service.get_order(order_id)
            if order.get("customer_id") != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    content={
                        "error": True,
                        "code": "forbidden",
                        "message": f"Замовлення {order_id} не належить вашому обліковому запису.",
                    },
                    reason="чуже замовлення (відхилено кодом)",
                    arguments=args,
                )
            res = service.cancel_order(order_id)
            return ToolResult(
                status="ok",
                content=_sanitize_order(res),
                arguments=args,
            )

        else:
            return ToolResult(
                status="rejected",
                content={"error": True, "message": f"Невідомий інструмент: {name}"},
                reason=f"невідомий інструмент {name}",
                arguments=args,
            )

    except service.ShopError as exc:
        return ToolResult(
            status="error",
            content={
                "error": True,
                "code": exc.code,
                "message": str(exc),
            },
            reason=f"{exc.code}: {exc}",
            arguments=args,
        )
    except Exception as exc:
        return ToolResult(
            status="error",
            content={
                "error": True,
                "code": "internal_error",
                "message": f"Помилка виконання інструмента: {exc}",
            },
            reason=f"виняток сервісу: {exc}",
            arguments=args,
        )
