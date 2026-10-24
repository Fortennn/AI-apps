"""Цикл помічника: від питання клієнта до відповіді, спертої на дані магазину."""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

# Скільки разів можна звернутися до моделі за одне питання клієнта.
MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    """Запис про один виклик інструмента — для сторінки й для перевірки."""

    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    """Результат циклу — те, з чим працює веб-рівень."""

    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    """Відповісти клієнтові, за потреби викликаючи інструменти."""
    ctx = tools.Context(customer_id=customer_id)
    messages = llm.build_messages(question)
    tool_specs = tools.specs()

    traces: list[ToolTrace] = []
    elapsed_model = 0.0
    elapsed_tools = 0.0
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    current_model = None

    for round_idx in range(1, MAX_ROUNDS + 1):
        chat_res = llm.chat(messages, tools=tool_specs, tool_choice="auto")
        elapsed_model += chat_res["elapsed"]
        current_model = chat_res["model"]

        # Додавання токенів
        if "usage" in chat_res and chat_res["usage"]:
            for k in usage_total:
                usage_total[k] += chat_res["usage"].get(k, 0)

        msg = chat_res["message"]

        # Якщо модель не викликає інструментів — маємо фінальну відповідь
        if not getattr(msg, "tool_calls", None):
            return Answer(
                text=msg.content or "",
                calls=traces,
                rounds=round_idx,
                stopped="answer",
                model=current_model,
                elapsed={
                    "model": round(elapsed_model, 3),
                    "tools": round(elapsed_tools, 3),
                },
                usage=usage_total,
            )

        # Додаємо повідомлення моделі в історію розмови зі збереженням підписів думок (thought_signature)
        if hasattr(msg, "model_dump"):
            assistant_msg = msg.model_dump(exclude_none=True)
        else:
            assistant_msg = dict(msg)
        messages.append(assistant_msg)

        # Виконуємо кожен запропонований виклик інструмента
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            raw_args = tc.function.arguments or "{}"

            tool_start = time.perf_counter()
            tool_res = tools.call(tool_name, raw_args, ctx)
            t_elapsed = round(time.perf_counter() - tool_start, 4)
            elapsed_tools += t_elapsed

            traces.append(
                ToolTrace(
                    round=round_idx,
                    name=tool_name,
                    arguments=raw_args,
                    status=tool_res.status,
                    reason=tool_res.reason,
                    result=tool_res.content,
                    elapsed=t_elapsed,
                )
            )

            # Додаємо результат виконання інструмента в історію
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_res.content, ensure_ascii=False),
            })

        # Якщо ми щойно виконали виклики на останньому дозволеному раунді,
        # робимо один додатковий виклик без інструментів для формулювання відповіді
        if round_idx == MAX_ROUNDS:
            final_chat = llm.chat(messages, tools=None)
            elapsed_model += final_chat["elapsed"]
            if "usage" in final_chat and final_chat["usage"]:
                for k in usage_total:
                    usage_total[k] += final_chat["usage"].get(k, 0)

            return Answer(
                text=final_chat["message"].content or "Досягнуто ліміту звертань до інструментів.",
                calls=traces,
                rounds=round_idx,
                stopped="limit",
                model=current_model,
                elapsed={
                    "model": round(elapsed_model, 3),
                    "tools": round(elapsed_tools, 3),
                },
                usage=usage_total,
            )

    # Якщо цикл завершився несподівано
    return Answer(
        text="Не вдалося отримати відповідь від помічника.",
        calls=traces,
        rounds=MAX_ROUNDS,
        stopped="limit",
        model=current_model,
        elapsed={
            "model": round(elapsed_model, 3),
            "tools": round(elapsed_tools, 3),
        },
        usage=usage_total,
    )
