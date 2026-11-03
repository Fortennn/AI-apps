import copy
import json
from dataclasses import dataclass

from shop import service


@dataclass
class Context:
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


ALLOWED_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "Отримати список замовлень поточного клієнта з основними статусами та сумами.",
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
            "description": "Отримати деталі конкретного замовлення клієнта (товари, статус, доставка).",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Номер замовлення"}
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
            "description": "Пошук товарів у каталозі магазину за назвою або описом.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Пошуковий запит"},
                    "category": {"type": "string", "description": "Категорія товару"},
                    "max_price": {"type": "number", "description": "Максимальна ціна"},
                    "limit": {"type": "integer", "description": "Кількість результатів (1-50)"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Отримати інформацію про товар за артикулом (SKU).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Артикул товару"}
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
            "description": "Перевірити наявність товару на складі за артикулом.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Артикул товару"}
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
            "description": "Розрахувати вартість та строк доставки товарів у вказане місто.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Місто доставки"},
                    "method": {
                        "type": "string",
                        "enum": ["branch", "courier", "pickup"],
                        "description": "Спосіб доставки",
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer"},
                            },
                            "required": ["sku", "quantity"],
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
            "name": "list_returns",
            "description": "Отримати список створених заявок на повернення для замовлення.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Номер замовлення"}
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Створити заявку на повернення товару із замовлення.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Номер замовлення"},
                    "sku": {"type": "string", "description": "Артикул товару"},
                    "reason": {
                        "type": "string",
                        "enum": ["not_suitable", "defect", "wrong_item", "damaged"],
                        "description": "Причина повернення",
                    },
                    "quantity": {"type": "integer", "description": "Кількість"},
                    "comment": {"type": "string", "description": "Коментар клієнта"},
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
            "description": "Скасувати замовлення клієнта, якщо воно в статусі Готується.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Номер замовлення"}
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
]


def specs() -> list[dict]:
    return copy.deepcopy(ALLOWED_SPECS)


def _sanitize_order(order: dict) -> dict:
    return {
        "order_id": order.get("order_id"),
        "customer_id": order.get("customer_id"),
        "created_at": order.get("created_at"),
        "status": order.get("status"),
        "status_label": order.get("status_label"),
        "total": order.get("total"),
        "items": [
            {
                "sku": item.get("sku"),
                "name": item.get("name"),
                "quantity": item.get("quantity"),
                "price": item.get("price"),
            }
            for item in order.get("items", [])
        ],
        "delivery": {
            "method": order.get("delivery", {}).get("method"),
            "city": order.get("delivery", {}).get("city"),
            "tracking_number": order.get("delivery", {}).get("tracking_number"),
            "delivered_at": order.get("delivery", {}).get("delivered_at"),
        },
    }


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    admin_tools = {"issue_refund", "update_price", "add_bonus", "set_order_status"}
    if name in admin_tools:
        return ToolResult(
            status="rejected",
            reason=f"Операція {name} є адміністративною і недоступна.",
            content=f"Операція {name} недоступна.",
        )

    allowed_names = {s["function"]["name"] for s in ALLOWED_SPECS}
    if name not in allowed_names:
        return ToolResult(
            status="rejected",
            reason=f"Невідомий інструмент {name}.",
            content=f"Інструмент {name} не підтримується.",
        )

    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except Exception as exc:
        return ToolResult(
            status="error",
            reason=f"Невалідний JSON аргументів: {exc}",
            content="Помилка передачі аргументів інструменту.",
        )

    try:
        if name == "list_orders":
            orders = service.list_orders(ctx.customer_id)
            return ToolResult(status="ok", content=orders, arguments=args)

        elif name == "get_order":
            order_id = str(args.get("order_id", "")).strip()
            if not order_id:
                return ToolResult(
                    status="rejected",
                    reason="Номер замовлення не вказано.",
                    content="Номер замовлення не вказано.",
                    arguments=args,
                )
            order = service.get_order(order_id)
            if str(order.get("customer_id")) != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    reason="Замовлення не належить поточному клієнту.",
                    content="Замовлення не знайдено серед ваших замовлень.",
                    arguments=args,
                )
            return ToolResult(
                status="ok",
                content=_sanitize_order(order),
                arguments=args,
            )

        elif name == "search_products":
            results = service.search_products(
                query=args.get("query", ""),
                category=args.get("category"),
                max_price=args.get("max_price"),
                limit=int(args.get("limit", 10)),
            )
            return ToolResult(status="ok", content=results, arguments=args)

        elif name == "get_product":
            sku = str(args.get("sku", "")).strip()
            prod = service.get_product(sku)
            safe_prod = {
                "sku": prod.get("sku"),
                "name": prod.get("name"),
                "category": prod.get("category"),
                "price": prod.get("price"),
                "warranty_months": prod.get("warranty_months"),
                "returnable": prod.get("returnable"),
                "description": prod.get("description"),
            }
            return ToolResult(status="ok", content=safe_prod, arguments=args)

        elif name == "get_stock":
            sku = str(args.get("sku", "")).strip()
            stock = service.get_stock(sku)
            return ToolResult(status="ok", content=stock, arguments=args)

        elif name == "delivery_quote":
            quote = service.delivery_quote(
                city=args.get("city", ""),
                method=args.get("method", ""),
                items=args.get("items", []),
            )
            return ToolResult(status="ok", content=quote, arguments=args)

        elif name == "list_returns":
            order_id = args.get("order_id")
            if order_id:
                order_id = str(order_id).strip()
                try:
                    order = service.get_order(order_id)
                    if str(order.get("customer_id")) != ctx.customer_id:
                        return ToolResult(
                            status="rejected",
                            reason="Замовлення не належить поточному клієнту.",
                            content=[],
                            arguments=args,
                        )
                except service.NotFound:
                    return ToolResult(status="ok", content=[], arguments=args)
            returns = service.list_returns(order_id=order_id)
            return ToolResult(status="ok", content=returns, arguments=args)

        elif name == "create_return":
            order_id = str(args.get("order_id", "")).strip()
            order = service.get_order(order_id)
            if str(order.get("customer_id")) != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    reason="Замовлення не належить поточному клієнту.",
                    content="Замовлення не належить клієнту.",
                    arguments=args,
                )
            res = service.create_return(
                order_id=order_id,
                sku=str(args.get("sku", "")).strip(),
                reason=str(args.get("reason", "")).strip(),
                quantity=int(args.get("quantity", 1)),
                comment=str(args.get("comment", "")),
            )
            return ToolResult(status="ok", content=res, arguments=args)

        elif name == "cancel_order":
            order_id = str(args.get("order_id", "")).strip()
            order = service.get_order(order_id)
            if str(order.get("customer_id")) != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    reason="Замовлення не належить поточному клієнту.",
                    content="Замовлення не належить клієнту.",
                    arguments=args,
                )
            res = service.cancel_order(order_id)
            return ToolResult(status="ok", content=_sanitize_order(res), arguments=args)

        return ToolResult(
            status="error",
            reason=f"Не реалізовано: {name}",
            arguments=args,
        )

    except service.PolicyViolation as exc:
        return ToolResult(status="rejected", reason=str(exc), content=str(exc), arguments=args)
    except service.NotFound as exc:
        return ToolResult(status="rejected", reason=str(exc), content=str(exc), arguments=args)
    except service.InvalidRequest as exc:
        return ToolResult(status="rejected", reason=str(exc), content=str(exc), arguments=args)
    except service.Unavailable as exc:
        return ToolResult(status="error", reason=str(exc), content=str(exc), arguments=args)
    except Exception as exc:
        return ToolResult(status="error", reason=str(exc), content=str(exc), arguments=args)
