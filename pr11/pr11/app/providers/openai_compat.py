"""Справжній провайдер: OpenAI-сумісний API — той самий, що в ПР3–ПР10.
"""

import time
from typing import Iterator

import openai
from openai import OpenAI

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "no-key",
            timeout=self.timeout,
            max_retries=0,
        )

    def _translate_error(self, exc: Exception) -> ProviderError:
        """Перетворити виняток SDK на типізовану ProviderError без витоку секретів."""
        if isinstance(exc, openai.APITimeoutError):
            return ProviderError(
                kind="timeout",
                message="Час очікування відповіді від моделі вичерпано",
                retryable=True,
                detail=str(exc),
            )
        if isinstance(exc, openai.RateLimitError):
            retry_after: float | None = None
            if hasattr(exc, "response") and exc.response is not None:
                header = exc.response.headers.get("retry-after")
                if header:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        pass
            return ProviderError(
                kind="rate_limit",
                message="Перевищено ліміт запитів до моделі (429)",
                retryable=True,
                retry_after=retry_after,
                detail=str(exc),
            )
        if isinstance(exc, openai.AuthenticationError):
            return ProviderError(
                kind="auth",
                message="Помилка автентифікації API ключа (401)",
                retryable=False,
                detail="Authentication failed. Check LLM_API_KEY.",
            )
        if isinstance(exc, openai.BadRequestError):
            return ProviderError(
                kind="bad_request",
                message="Некоректний запит до моделі (400)",
                retryable=False,
                detail=str(exc),
            )
        if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
            return ProviderError(
                kind="unavailable",
                message="Сервіс моделі тимчасово недоступний або сталася помилка з'єднання",
                retryable=True,
                detail=str(exc),
            )
        if isinstance(exc, openai.APIStatusError):
            status = exc.status_code
            if status >= 500:
                return ProviderError(
                    kind="unavailable",
                    message=f"Сервіс моделі повернув помилку сервера ({status})",
                    retryable=True,
                    detail=str(exc),
                )
            if status == 429:
                return ProviderError(
                    kind="rate_limit",
                    message="Перевищено ліміт запитів до моделі (429)",
                    retryable=True,
                    detail=str(exc),
                )
            return ProviderError(
                kind="bad_request" if status < 500 else "unavailable",
                message=f"Помилка API моделі (код {status})",
                retryable=(status >= 500),
                detail=str(exc),
            )
        if isinstance(exc, ProviderError):
            return exc
        return ProviderError(
            kind="unknown",
            message="Сталася непередбачена помилка при зверненні до моделі",
            retryable=False,
            detail=str(exc),
        )

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        target_model = request.model or self.model
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=target_model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            elapsed = time.time() - t0
            choice = resp.choices[0] if resp.choices else None
            text = (choice.message.content or "") if choice else ""
            finish_reason = choice.finish_reason if choice else None
            usage = None
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
                if hasattr(resp.usage, "prompt_tokens_details") and resp.usage.prompt_tokens_details:
                    cached_toks = getattr(resp.usage.prompt_tokens_details, "cached_tokens", 0)
                    if cached_toks:
                        usage["prompt_tokens_details"] = {"cached_tokens": cached_toks}
            return Result(
                text=text,
                provider=self.name,
                model=target_model,
                usage=usage,
                finish_reason=finish_reason,
                elapsed=elapsed,
            )
        except Exception as exc:
            raise self._translate_error(exc) from exc

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати фрагменти відповіді в міру надходження."""
        target_model = request.model or self.model
        try:
            response_stream = self.client.chat.completions.create(
                model=target_model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            raise self._translate_error(exc) from exc

        try:
            for item in response_stream:
                choices = item.choices
                delta_text = ""
                finish_reason = None
                if choices:
                    c = choices[0]
                    finish_reason = c.finish_reason
                    if c.delta and c.delta.content:
                        delta_text = c.delta.content
                usage = None
                if item.usage:
                    usage = {
                        "prompt_tokens": item.usage.prompt_tokens,
                        "completion_tokens": item.usage.completion_tokens,
                        "total_tokens": item.usage.total_tokens,
                    }
                    if hasattr(item.usage, "prompt_tokens_details") and item.usage.prompt_tokens_details:
                        cached_toks = getattr(item.usage.prompt_tokens_details, "cached_tokens", 0)
                        if cached_toks:
                            usage["prompt_tokens_details"] = {"cached_tokens": cached_toks}
                yield Chunk(text=delta_text, usage=usage, finish_reason=finish_reason)
        except Exception as exc:
            raise self._translate_error(exc) from exc
        finally:
            if hasattr(response_stream, "close"):
                try:
                    response_stream.close()
                except Exception:
                    pass
