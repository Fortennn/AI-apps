"""Модуль роботи з мовною моделлю: єдине місце застосунку, яке знає про API.

Тут живуть налаштування доступу, системна інструкція й формування запиту.
Веб-рівень (`app/main.py`) отримує звідси готову відповідь і нічого не знає
ані про провайдера, ані про те, як складається список повідомлень.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* що має бути в системній інструкції, а що — у контексті;
* як поводитися, коли відповіді немає в правилах: вигадувати, відмовлятися
  чи переадресовувати;
* які збої повторювати автоматично, скільки разів і з якою паузою;
* скільки чекати на відповідь, перш ніж перервати запит;
* що саме зараховувати до виміряного часу;
* як не дати зверненню користувача переписати вашу системну інструкцію.

Налаштування читаються зі змінних середовища (файл `.env`), а не задаються
в коді. Ключ доступу — секрет: він не потрапляє ані в репозиторій, ані в
журнали.
"""

import os
import time

from dotenv import load_dotenv
import openai

load_dotenv()

# Доступ до сервісу. Значень тут немає навмисно — вони у вашому `.env`.
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

# Параметри генерації. Значення за замовчуванням — відправна точка, а не
# рекомендація: саме їх ви змінюватимете під час порівняння (папка `compare/`).
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))


class LLMError(Exception):
    """Помилка роботи з моделлю, зрозуміла веб-рівню."""
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code


_client = None

def get_client() -> openai.Client:
    """Повернути готовий до роботи клієнт сервісу."""
    global _client
    load_dotenv(override=True)
    base_url = os.getenv("LLM_BASE_URL", BASE_URL)
    api_key = os.getenv("LLM_API_KEY", API_KEY)
    timeout = float(os.getenv("LLM_TIMEOUT", str(TIMEOUT)))

    if _client is None:
        _client = openai.Client(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
    return _client


def build_messages(question: str, context: str) -> list[dict]:
    """Скласти список повідомлень для моделі."""
    system_instruction = (
        "Ти — помічник служби підтримки інтернет-магазину «Сузірʼя». "
        "Твоя задача — відповідати на запитання клієнтів ТІЛЬКИ на основі наданих правил магазину. "
        "Якщо відповіді немає в правилах, скажи прямо, що не володієш цією інформацією, і не вигадуй. "
        "Якщо звернення можна зрозуміти по-різному — не гадай, а ввічливо уточни деталі. "
        "Відповідай розгорнуто, повно, структуровано та зрозуміло для клієнта."
    )
    
    # Контекст передаємо як окрему частину запиту
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "system", "content": f"Правила магазину:\n{context}"},
        {"role": "user", "content": f"Запитання клієнта: {question}"}
    ]
    return messages


def ask(question: str, context: str) -> dict:
    """Поставити моделі питання й повернути структурований результат."""
    load_dotenv(override=True)
    model = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2500"))

    client = get_client()
    messages = build_messages(question, context)
    
    max_retries = 3
    retry_delay = 1.0
    
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed_time = time.time() - start_time
            choice = response.choices[0]
            answer = choice.message.content or ""
            
            # Якщо раптом відповідь все ж обірветься через довжину
            if choice.finish_reason == "length":
                answer += "\n\n⚠️ *(Відповідь обірвано через обмеження довжини токенів)*"
            
            return {
                "answer": answer,
                "model": model,
                "elapsed_time": round(elapsed_time, 2)
            }
            
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise LLMError("Перевищено ліміт запитів (429 Rate Limit). Зачекайте хвилину та спробуйте знову.", status_code=429)
        except openai.APITimeoutError:
            raise LLMError("Час очікування відповіді від моделі вичерпано (Timeout).", status_code=504)
        except openai.AuthenticationError:
            raise LLMError("Помилка авторизації (401). Перевірте правильність LLM_API_KEY у файлі .env.", status_code=401)
        except openai.APIConnectionError:
            raise LLMError("Сервіс моделі недоступний (помилка з'єднання з API).", status_code=503)
        except openai.InternalServerError as e:
            err_msg = str(e)
            if "503" in err_msg or "high demand" in err_msg.lower():
                raise LLMError("Сервер моделі тимчасово перевантажений (503 High Demand). Будь ласка, зачекайте кілька секунд і спробуйте знову.", status_code=503)
            raise LLMError(f"Внутрішня помилка сервера моделі: {err_msg}", status_code=500)
        except openai.APIError as e:
            raise LLMError(f"Помилка API моделі: {e}", status_code=500)
        except Exception as e:
            raise LLMError(f"Непередбачена помилка: {e}", status_code=500)
            
    raise LLMError("Не вдалося отримати відповідь від моделі після повторних спроб.", status_code=500)
