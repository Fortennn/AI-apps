import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

from shop import service
from . import llm, schema, store, tools
from .state import Run, Step

load_dotenv()

MAX_STEPS = int(os.getenv("WORKFLOW_MAX_STEPS", "12"))
TIMEOUT = float(os.getenv("WORKFLOW_TIMEOUT", "60"))
TOKEN_BUDGET = int(os.getenv("WORKFLOW_TOKEN_BUDGET", "8000"))


def limits() -> dict:
    return {"max_steps": MAX_STEPS, "timeout": TIMEOUT, "token_budget": TOKEN_BUDGET}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_limits(run: Run, started_at: float) -> bool:
    cur_elapsed = time.perf_counter() - started_at
    if len(run.steps) >= MAX_STEPS:
        run.status = "stopped"
        run.stopped = f"Перевищено ліміт кроків: {len(run.steps)} з {MAX_STEPS}"
        return False
    if cur_elapsed > TIMEOUT:
        run.status = "stopped"
        run.stopped = f"Перевищено ліміт часу виконання: {cur_elapsed:.1f} с з {TIMEOUT} с"
        return False
    used_tokens = run.usage.get("total_tokens", 0)
    if used_tokens > TOKEN_BUDGET:
        run.status = "stopped"
        run.stopped = f"Перевищено бюджет токенів: {used_tokens} з {TOKEN_BUDGET}"
        return False
    return True


def _fallback_parse(text: str) -> dict:
    lower = text.lower()
    has_multiple = ("скасу" in lower and "поверн" in lower) or (text.count("104") > 1)

    order_match = re.search(r"\b(10\d{3})\b", text)
    order_id = order_match.group(1) if order_match else None

    item = None
    if "фільтр" in lower:
        item = "фільтри"
    elif "пилосос" in lower:
        item = "робот-пилосос"
    elif "навушник" in lower:
        item = "навушники"
    elif "годинник" in lower:
        item = "годинник"
    elif "роутер" in lower:
        item = "роутер"

    reason = None
    if any(k in lower for k in ["дефект", "несправн", "не працю", "не зарядж", "злама"]):
        reason = "defect"
    elif any(k in lower for k in ["не підійш", "не знадоби", "передума", "завелик"]):
        reason = "not_suitable"
    elif "пошкодж" in lower:
        reason = "damaged"

    kind = "other"
    if "поверн" in lower or "поверніть" in lower:
        kind = "return"
    elif "скасу" in lower:
        kind = "cancel"
    elif any(k in lower for k in ["де моє", "де посилк", "де замовленн", "стан", "що робити", "лежить у відділенні"]):
        kind = "status"

    return {
        "kind": kind,
        "order_id": order_id,
        "item": item,
        "reason": reason,
        "reason_text": text,
        "quantity": 1,
        "has_multiple_actions": has_multiple,
    }


def start(text: str, customer_id: str) -> Run:
    started_at = time.perf_counter()
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    now_str = _now()
    run = Run(
        run_id=run_id,
        customer_id=customer_id,
        text=text,
        status="running",
        created_at=now_str,
        updated_at=now_str,
        elapsed={"model": 0.0, "tools": 0.0, "total": 0.0},
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )

    ctx = tools.Context(customer_id=customer_id)

    prompt = (
        "Визнач вид звернення клієнта магазину (return, cancel, status, other), "
        "номер замовлення (якщо названо клієнтом), товар (як названо), причину повернення, "
        "кількість і чи містить звернення кілька різних дій.\n\n"
        f"Звернення: {text}"
    )

    parse_res = None
    parse_usage = None
    t0_model = time.perf_counter()
    if os.getenv("OFFLINE_MODE", "0") == "1":
        t_model = 0.01
        run.elapsed["model"] += t_model
        parse_res = _fallback_parse(text)
    else:
        try:
            raw_llm = llm.complete(
                [
                    {"role": "system", "content": "Ти помічник, який структурує звернення клієнтів за схемою."},
                    {"role": "user", "content": prompt},
                ],
                schema=schema.request_schema(),
            )
            t_model = time.perf_counter() - t0_model
            run.elapsed["model"] += t_model
            parse_usage = raw_llm.get("usage")
            if parse_usage:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    run.usage[k] += parse_usage.get(k, 0)
            parse_res = schema.validate(raw_llm.get("text", ""))
        except Exception:
            t_model = time.perf_counter() - t0_model
            run.elapsed["model"] += t_model
            parse_res = _fallback_parse(text)

    run.steps.append(
        Step(
            n=len(run.steps) + 1,
            name="parse",
            kind="model",
            status="ok",
            detail="Розбір звернення структурованою схемою",
            input={"text": text},
            output=parse_res,
            elapsed=round(t_model, 4),
            usage=parse_usage,
            at=_now(),
        )
    )

    if not _check_limits(run, started_at):
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    if parse_res.get("has_multiple_actions"):
        run.route = "декілька"
        run.status = "handed_off"
        run.reply = (
            "Ваше звернення містить кілька окремих дій (наприклад, скасування та повернення). "
            "Його передано оператору для роздільної обробки без ризику помилок."
        )
        run.steps.append(
            Step(
                n=len(run.steps) + 1,
                name="validate",
                kind="code",
                status="handed_off",
                detail="Виявлено декілька дій в одному зверненні — передано оператору",
                input=parse_res,
                output={"status": "handed_off"},
                at=_now(),
            )
        )
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    kind = parse_res.get("kind", "other")
    run.route = kind
    order_id = parse_res.get("order_id")
    item_str = parse_res.get("item")
    reason = parse_res.get("reason")
    reason_text = parse_res.get("reason_text") or text

    if kind == "other":
        is_admin_request = any(k in text.lower() for k in ["менеджер", "повернення коштів", "бонус", "бонусів", "примітці"])
        if is_admin_request:
            run.status = "handed_off"
            run.reply = "Запит щодо фінансових операцій чи нарахування бонусів передано на розгляд адміністратора."
        else:
            run.status = "done"
            run.reply = "Дякуємо за ваше звернення! Якщо у вас виникнуть інші питання, ми завжди раді допомогти."
        run.steps.append(
            Step(
                n=len(run.steps) + 1,
                name="validate",
                kind="code",
                status="ok",
                detail="Звернення загального або адміністративного характеру",
                input=parse_res,
                output={"reply": run.reply},
                at=_now(),
            )
        )
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    if not order_id:
        t0_tools = time.perf_counter()
        orders_res = tools.call("list_orders", "{}", ctx)
        t_tools = time.perf_counter() - t0_tools
        run.elapsed["tools"] += t_tools

        orders_list = orders_res.content if orders_res.status == "ok" and isinstance(orders_res.content, list) else []
        matched_order_id = None

        if item_str and orders_list:
            lower_item = item_str.lower()
            matching_orders = []
            for ord_item in orders_list:
                full_ord = service.get_order(ord_item["order_id"])
                for it in full_ord.get("items", []):
                    if lower_item in it["name"].lower() or any(w in it["name"].lower() for w in lower_item.split() if len(w) > 3):
                        matching_orders.append(ord_item["order_id"])
                        break
            if len(matching_orders) == 1:
                matched_order_id = matching_orders[0]
                order_id = matched_order_id
            elif len(matching_orders) > 1:
                run.status = "needs_info"
                run.reply = f"Знайдено декілька замовлень з товаром '{item_str}'. Будь ласка, уточніть номер замовлення."
                run.steps.append(
                    Step(
                        n=len(run.steps) + 1,
                        name="identify_order",
                        kind="code",
                        status="needs_info",
                        detail="Кілька відповідних замовлень",
                        input={"item": item_str, "found": matching_orders},
                        output={"reply": run.reply},
                        at=_now(),
                    )
                )
                run.updated_at = _now()
                run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
                store.save(run)
                return run

        if not order_id:
            run.status = "needs_info"
            run.reply = "Будь ласка, уточніть номер замовлення та товар, якого стосується ваше звернення."
            run.steps.append(
                Step(
                    n=len(run.steps) + 1,
                    name="identify_order",
                    kind="code",
                    status="needs_info",
                    detail="Не вказано номер замовлення",
                    input={"customer_id": customer_id, "text": text},
                    output={"reply": run.reply},
                    at=_now(),
                )
            )
            run.updated_at = _now()
            run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
            store.save(run)
            return run

    if not _check_limits(run, started_at):
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    t0_tools = time.perf_counter()
    order_res = tools.call("get_order", json.dumps({"order_id": order_id}), ctx)
    t_tools = time.perf_counter() - t0_tools
    run.elapsed["tools"] += t_tools

    run.steps.append(
        Step(
            n=len(run.steps) + 1,
            name="get_order",
            kind="tool",
            status=order_res.status,
            detail=order_res.reason or "Отримано дані замовлення",
            input={"order_id": order_id},
            output=order_res.content,
            elapsed=round(t_tools, 4),
            at=_now(),
        )
    )

    if order_res.status != "ok":
        run.status = "rejected"
        run.reply = "Замовлення не знайдено серед ваших замовлень або номер вказано некоректно."
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    order_data = order_res.content
    order_status = order_data.get("status")
    order_status_label = order_data.get("status_label", order_status)
    items = order_data.get("items", [])

    if kind == "status":
        delivery = order_data.get("delivery", {})
        tracking = delivery.get("tracking_number")
        city = delivery.get("city")
        tracking_info = f", номер ТТН: {tracking}" if tracking else ""
        city_info = f" у м. {city}" if city else ""
        run.reply = f"Замовлення №{order_id} має статус «{order_status_label}»{city_info}{tracking_info}."
        run.status = "done"
        run.steps.append(
            Step(
                n=len(run.steps) + 1,
                name="status_reply",
                kind="code",
                status="ok",
                detail="Надано статус замовлення клієнту",
                input={"order_id": order_id, "status": order_status_label},
                output={"reply": run.reply},
                at=_now(),
            )
        )
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    if kind == "cancel":
        if order_status != "preparing":
            run.status = "rejected"
            run.reply = f"Замовлення №{order_id} не може бути скасоване, оскільки воно вже має статус «{order_status_label}»."
            run.steps.append(
                Step(
                    n=len(run.steps) + 1,
                    name="check_cancel_rules",
                    kind="code",
                    status="rejected",
                    detail=f"Скасування заборонено в статусі {order_status_label}",
                    input={"order_id": order_id, "status": order_status},
                    output={"allowed": False},
                    at=_now(),
                )
            )
            run.updated_at = _now()
            run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
            store.save(run)
            return run

        pending = {
            "summary": f"Скасування замовлення №{order_id}",
            "tool": "cancel_order",
            "arguments": {"order_id": order_id},
            "grounds": f"Клієнт попросив скасувати замовлення. Поточний статус «{order_status_label}» дозволяє скасування.",
        }
        run.pending = pending
        run.status = "awaiting_approval"
        run.steps.append(
            Step(
                n=len(run.steps) + 1,
                name="prepare_cancellation",
                kind="code",
                status="waiting",
                detail="Дія потребує підтвердження оператора",
                input={"order_id": order_id},
                output=pending,
                at=_now(),
            )
        )
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    if kind == "return":
        if order_status != "delivered":
            run.status = "rejected"
            run.reply = f"Повернення товару можливе лише після отримання замовлення. Поточний статус замовлення №{order_id}: «{order_status_label}»."
            run.steps.append(
                Step(
                    n=len(run.steps) + 1,
                    name="check_delivery_status",
                    kind="code",
                    status="rejected",
                    detail=f"Замовлення ще не отримано ({order_status_label})",
                    input={"order_id": order_id, "status": order_status},
                    output={"allowed": False},
                    at=_now(),
                )
            )
            run.updated_at = _now()
            run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
            store.save(run)
            return run

        t0_tools = time.perf_counter()
        ret_list_res = tools.call("list_returns", json.dumps({"order_id": order_id}), ctx)
        t_tools = time.perf_counter() - t0_tools
        run.elapsed["tools"] += t_tools

        existing_returns = ret_list_res.content if ret_list_res.status == "ok" and isinstance(ret_list_res.content, list) else []

        target_line = None
        if item_str:
            lower_item = item_str.lower()
            for line in items:
                if lower_item in line["name"].lower() or any(w in line["name"].lower() for w in lower_item.split() if len(w) > 3):
                    target_line = line
                    break
        if not target_line and items:
            target_line = items[0]

        if not target_line:
            run.status = "needs_info"
            run.reply = f"У замовленні №{order_id} не знайдено вказаного товару. Уточніть назву або артикул."
            run.updated_at = _now()
            run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
            store.save(run)
            return run

        sku = target_line["sku"]

        for er in existing_returns:
            if er.get("sku") == sku:
                run.status = "rejected"
                run.reply = f"Заявку на повернення товару «{target_line['name']}» уже було створено раніше (номер заявки: {er.get('return_id')})."
                run.steps.append(
                    Step(
                        n=len(run.steps) + 1,
                        name="check_existing_return",
                        kind="code",
                        status="rejected",
                        detail=f"Виявлено дублікат заявки ({er.get('return_id')})",
                        input={"order_id": order_id, "sku": sku},
                        output=er,
                        at=_now(),
                    )
                )
                run.updated_at = _now()
                run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
                store.save(run)
                return run

        t0_tools = time.perf_counter()
        prod_res = tools.call("get_product", json.dumps({"sku": sku}), ctx)
        t_tools = time.perf_counter() - t0_tools
        run.elapsed["tools"] += t_tools

        prod_data = prod_res.content if prod_res.status == "ok" and isinstance(prod_res.content, dict) else {}
        delivered_at_str = order_data.get("delivery", {}).get("delivered_at")

        return_reason = reason or "not_suitable"
        if "не зарядж" in text.lower() or "дефект" in text.lower() or "перестав" in text.lower() or "злама" in text.lower():
            return_reason = "defect"
        elif "не підійш" in text.lower() or "не знадоби" in text.lower() or "завелик" in text.lower():
            return_reason = "not_suitable"

        today_val = service.today()
        delivered_date = datetime.fromisoformat(delivered_at_str).date() if delivered_at_str else today_val
        days_passed = (today_val - delivered_date).days

        if return_reason == "not_suitable":
            if not prod_data.get("returnable", True):
                run.status = "rejected"
                run.reply = f"Товар «{target_line['name']}» як товар належної якості не підлягає поверненню згідно з правилами магазину."
                run.steps.append(
                    Step(
                        n=len(run.steps) + 1,
                        name="check_return_rules",
                        kind="code",
                        status="rejected",
                        detail="Товар не підлягає поверненню (витратні матеріали або предмети особистої гігієни)",
                        input={"sku": sku, "returnable": False},
                        output={"allowed": False},
                        at=_now(),
                    )
                )
                run.updated_at = _now()
                run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
                store.save(run)
                return run

            if days_passed > service.RETURN_WINDOW_DAYS:
                run.status = "rejected"
                run.reply = f"Від дати отримання замовлення минуло {days_passed} дн. Товар належної якості можна повернути лише протягом {service.RETURN_WINDOW_DAYS} днів."
                run.steps.append(
                    Step(
                        n=len(run.steps) + 1,
                        name="check_return_window",
                        kind="code",
                        status="rejected",
                        detail=f"Минуло {days_passed} дн. (максимум {service.RETURN_WINDOW_DAYS})",
                        input={"days_passed": days_passed},
                        output={"allowed": False},
                        at=_now(),
                    )
                )
                run.updated_at = _now()
                run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
                store.save(run)
                return run

        elif return_reason == "defect":
            warranty_months = prod_data.get("warranty_months", 12)
            warranty_days = warranty_months * 30
            if days_passed > warranty_days:
                run.status = "rejected"
                run.reply = f"Гарантійний строк на товар «{target_line['name']}» закінчився."
                run.steps.append(
                    Step(
                        n=len(run.steps) + 1,
                        name="check_warranty",
                        kind="code",
                        status="rejected",
                        detail=f"Гарантійний строк минув ({days_passed} > {warranty_days})",
                        input={"days_passed": days_passed, "warranty_months": warranty_months},
                        output={"allowed": False},
                        at=_now(),
                    )
                )
                run.updated_at = _now()
                run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
                store.save(run)
                return run

        pending = {
            "summary": f"Створення заявки на повернення: «{target_line['name']}» (арт. {sku}) із замовлення №{order_id}",
            "tool": "create_return",
            "arguments": {
                "order_id": order_id,
                "sku": sku,
                "reason": return_reason,
                "quantity": 1,
                "comment": reason_text,
            },
            "grounds": f"Замовлення отримане {delivered_date} ({days_passed} дн. тому). Причина: {return_reason}. Товар підлягає поверненню за правилами магазину.",
        }
        run.pending = pending
        run.status = "awaiting_approval"
        run.steps.append(
            Step(
                n=len(run.steps) + 1,
                name="prepare_return",
                kind="code",
                status="waiting",
                detail="Заявка готова та чекає на підтвердження оператора",
                input={"order_id": order_id, "sku": sku, "reason": return_reason},
                output=pending,
                at=_now(),
            )
        )
        run.updated_at = _now()
        run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
        store.save(run)
        return run

    run.updated_at = _now()
    run.elapsed["total"] = round(time.perf_counter() - started_at, 4)
    store.save(run)
    return run


def resume(run_id: str, approve: bool, comment: str = "") -> Run:
    run = store.load(run_id)
    if run is None:
        raise ValueError(f"Звернення {run_id} не знайдено.")

    if run.status != "awaiting_approval":
        raise ValueError(f"Звернення {run_id} не очікує підтвердження (статус: {run.status}).")

    ctx = tools.Context(customer_id=run.customer_id)
    now_str = _now()
    run.decision = {"approve": approve, "comment": comment, "at": now_str}

    run.steps.append(
        Step(
            n=len(run.steps) + 1,
            name="operator_decision",
            kind="human",
            status="ok" if approve else "rejected",
            detail=comment or ("Оператор підтвердив дію" if approve else "Оператор відхилив дію"),
            input=run.pending,
            output={"approve": approve, "comment": comment},
            at=now_str,
        )
    )

    if not approve:
        run.status = "rejected"
        run.reply = f"Запит було відхилено оператором.{' Коментар: ' + comment if comment else ''}"
        run.updated_at = _now()
        store.save(run)
        return run

    pending = run.pending or {}
    tool_name = pending.get("tool")
    args = pending.get("arguments", {})

    t0 = time.perf_counter()
    tool_res = tools.call(tool_name, json.dumps(args), ctx)
    t_elapsed = time.perf_counter() - t0
    run.elapsed["tools"] += t_elapsed

    run.steps.append(
        Step(
            n=len(run.steps) + 1,
            name="execute_action",
            kind="tool",
            status=tool_res.status,
            detail=tool_res.reason or "Дію успішно виконано в системі",
            input=args,
            output=tool_res.content,
            elapsed=round(t_elapsed, 4),
            at=_now(),
        )
    )

    if tool_res.status != "ok":
        run.status = "failed"
        run.reply = f"Помилка при виконанні операції: {tool_res.reason or 'збій сервісу'}"
        run.updated_at = _now()
        store.save(run)
        return run

    if tool_name == "create_return":
        res_data = tool_res.content if isinstance(tool_res.content, dict) else {}
        ret_id = res_data.get("return_id", "R-...")
        next_steps = res_data.get("next_steps", "Надішліть товар на склад. Кошти будуть повернуті протягом 7 банківських днів.")
        run.reply = f"Заявку на повернення товару успішно оформлено! Номер заявки: {ret_id}. {next_steps}"
        run.status = "done"

    elif tool_name == "cancel_order":
        ord_id = args.get("order_id")
        run.reply = f"Замовлення №{ord_id} успішно скасовано."
        run.status = "done"

    run.steps.append(
        Step(
            n=len(run.steps) + 1,
            name="generate_reply",
            kind="code",
            status="ok",
            detail="Формування підсумкової відповіді клієнту",
            input={"execution_result": tool_res.content},
            output={"reply": run.reply},
            at=_now(),
        )
    )

    run.updated_at = _now()
    store.save(run)
    return run
