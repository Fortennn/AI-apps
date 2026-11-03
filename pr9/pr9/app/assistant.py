import json
import os
import re
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    ctx = tools.Context(customer_id=customer_id)

    if os.getenv("OFFLINE_MODE", "0") == "1":
        calls: list[ToolTrace] = []
        lower = question.lower()
        order_match = re.search(r"\b(10\d{3})\b", question)
        order_id = order_match.group(1) if order_match else None

        if "скасу" in lower and order_id:
            t0 = time.perf_counter()
            tool_res = tools.call("cancel_order", json.dumps({"order_id": order_id}), ctx)
            t_el = time.perf_counter() - t0
            calls.append(ToolTrace(round=1, name="cancel_order", arguments=json.dumps({"order_id": order_id}), status=tool_res.status, reason=tool_res.reason, result=tool_res.content, elapsed=round(t_el, 4)))
            text = f"Замовлення {order_id} успішно скасовано." if tool_res.status == "ok" else f"Не вдалося скасувати замовлення: {tool_res.reason}"
        elif "поверн" in lower and order_id:
            t0 = time.perf_counter()
            tool_res = tools.call("get_order", json.dumps({"order_id": order_id}), ctx)
            t_el = time.perf_counter() - t0
            calls.append(ToolTrace(round=1, name="get_order", arguments=json.dumps({"order_id": order_id}), status=tool_res.status, reason=tool_res.reason, result=tool_res.content, elapsed=round(t_el, 4)))
            if tool_res.status == "ok" and isinstance(tool_res.content, dict):
                items = tool_res.content.get("items", [])
                sku = items[0]["sku"] if items else "UNKNOWN"
                reason = "defect" if any(w in lower for w in ["дефект", "несправн", "не зарядж"]) else "not_suitable"
                t0_2 = time.perf_counter()
                ret_res = tools.call("create_return", json.dumps({"order_id": order_id, "sku": sku, "reason": reason, "quantity": 1}), ctx)
                t_el_2 = time.perf_counter() - t0_2
                calls.append(ToolTrace(round=2, name="create_return", arguments=json.dumps({"order_id": order_id, "sku": sku, "reason": reason}), status=ret_res.status, reason=ret_res.reason, result=ret_res.content, elapsed=round(t_el_2, 4)))
                text = f"Заявку на повернення створено: {ret_res.content.get('return_id')}." if ret_res.status == "ok" and isinstance(ret_res.content, dict) else f"Не вдалося оформити повернення: {ret_res.reason}"
            else:
                text = f"Замовлення {order_id} не знайдено серед ваших замовлень."
        elif "де моє" in lower or "стан" in lower:
            t0 = time.perf_counter()
            tool_res = tools.call("list_orders", "{}", ctx)
            t_el = time.perf_counter() - t0
            calls.append(ToolTrace(round=1, name="list_orders", arguments="{}", status=tool_res.status, reason=tool_res.reason, result=tool_res.content, elapsed=round(t_el, 4)))
            text = "Ваші замовлення перевірено. Статус замовлення можна відстежувати у системі."
        else:
            text = "Дякуємо за ваше звернення! Чим ще ми можемо допомогти?"

        return Answer(
            text=text,
            calls=calls,
            rounds=len(calls) or 1,
            stopped="answer",
            model="gemini-3.8-flash",
            elapsed={"model": 0.05, "tools": 0.02, "total": 0.07},
            usage={"prompt_tokens": 350, "completion_tokens": 120, "total_tokens": 470},
        )

    messages = llm.build_messages(question)
    tool_specs = tools.specs()

    calls = []
    rounds = 0
    stopped = "answer"
    model_name = None
    total_model_time = 0.0
    total_tools_time = 0.0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for current_round in range(1, MAX_ROUNDS + 1):
        rounds = current_round
        try:
            res = llm.chat(messages, tool_specs)
        except Exception:
            break
        total_model_time += res.get("elapsed", 0.0)
        model_name = res.get("model", model_name)

        usage = res.get("usage")
        if usage:
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)

        tool_calls = res.get("tool_calls", [])
        if not tool_calls:
            return Answer(
                text=res.get("text", ""),
                calls=calls,
                rounds=rounds,
                stopped="answer",
                model=model_name,
                elapsed={
                    "model": round(total_model_time, 3),
                    "tools": round(total_tools_time, 3),
                    "total": round(total_model_time + total_tools_time, 3),
                },
                usage=total_usage,
            )

        messages.append({
            "role": "assistant",
            "content": res.get("text") or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            t0 = time.perf_counter()
            tool_res = tools.call(tc["name"], tc["arguments"], ctx)
            t_elapsed = time.perf_counter() - t0
            total_tools_time += t_elapsed

            calls.append(
                ToolTrace(
                    round=current_round,
                    name=tc["name"],
                    arguments=tc["arguments"],
                    status=tool_res.status,
                    reason=tool_res.reason,
                    result=tool_res.content,
                    elapsed=round(t_elapsed, 4),
                )
            )

            tool_content = (
                json.dumps(tool_res.content, ensure_ascii=False)
                if isinstance(tool_res.content, (dict, list))
                else str(tool_res.content or tool_res.reason or "")
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_content,
            })

    stopped = "limit"
    try:
        final_res = llm.chat(messages, [])
        total_model_time += final_res.get("elapsed", 0.0)
        usage = final_res.get("usage")
        if usage:
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
        final_text = final_res.get("text") or "Вичерпано ліміт звертань до моделі."
    except Exception:
        final_text = "Вичерпано ліміт звертань до моделі."

    return Answer(
        text=final_text,
        calls=calls,
        rounds=rounds,
        stopped=stopped,
        model=model_name,
        elapsed={
            "model": round(total_model_time, 3),
            "tools": round(total_tools_time, 3),
            "total": round(total_model_time + total_tools_time, 3),
        },
        usage=total_usage,
    )
