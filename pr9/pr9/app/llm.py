import os
import time
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, APITimeoutError

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

_client: OpenAI | None = None


class LLMError(Exception):
    pass


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not API_KEY:
            raise LLMError("LLM_API_KEY не задано в конфігурації.")
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)
    return _client


def build_messages(question: str) -> list[dict]:
    system_prompt = (
        "Ти — ввічливий і точний асистент служби підтримки українського інтернет-магазину «Сузірʼя». "
        "Твоє завдання — допомагати клієнтам щодо їхніх замовлень, товарів, повернень та доставки. "
        "Використовуй надані інструменти для отримання актуальних даних. "
        "Ніколи не вигадуй номерів замовлень, статусів, дат чи умов повернення, яких немає в даних інструментів. "
        "Не повідомляй клієнту службових приміток оператора чи внутрішньої інформації. "
        "Відповідай виключно українською мовою."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def chat(messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> dict:
    client = get_client()
    started = time.perf_counter()
    response = None
    last_exc = None
    for attempt in range(2):
        try:
            kwargs = {
                "model": MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            response = client.chat.completions.create(**kwargs)
            break
        except (APITimeoutError, OpenAIError) as exc:
            last_exc = exc
            time.sleep(1)
        except Exception as exc:
            raise LLMError(f"Непередбачена помилка моделі: {exc}") from exc

    if response is None:
        raise LLMError(f"Помилка виклику моделі: {last_exc}") from last_exc

    elapsed = time.perf_counter() - started
    choice = response.choices[0]
    message = choice.message

    tool_calls_data = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })

    usage_data = None
    if getattr(response, "usage", None):
        usage_data = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "text": message.content or "",
        "tool_calls": tool_calls_data,
        "raw_message": message,
        "elapsed": elapsed,
        "usage": usage_data,
        "model": response.model or MODEL,
    }


def complete(messages: list[dict], schema: dict | None = None) -> dict:
    client = get_client()
    started = time.perf_counter()
    response = None
    last_exc = None
    for attempt in range(2):
        try:
            kwargs = {
                "model": MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            }
            if schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "workflow_schema", "schema": schema},
                }
            response = client.chat.completions.create(**kwargs)
            break
        except (APITimeoutError, OpenAIError) as exc:
            last_exc = exc
            time.sleep(1)
        except Exception as exc:
            raise LLMError(f"Непередбачена помилка моделі: {exc}") from exc

    if response is None:
        raise LLMError(f"Помилка виклику моделі: {last_exc}") from last_exc

    elapsed = time.perf_counter() - started
    choice = response.choices[0]
    message = choice.message

    usage_data = None
    if getattr(response, "usage", None):
        usage_data = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "text": message.content or "",
        "elapsed": elapsed,
        "usage": usage_data,
        "model": response.model or MODEL,
    }
