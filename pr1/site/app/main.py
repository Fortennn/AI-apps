"""Веб-рівень застосунку: сторінка з формою і JSON-ендпоінт.

Цей файл не знає про `requests`, адреси сервісів і коди їхніх відповідей.
Він приймає винятки з `app/weather.py` і повертає правильні HTTP-статуси з JSON.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from . import weather

app = FastAPI(title="Погода — ПР1")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку з формою вводу міста."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/weather")
def api_weather(city: str = Query(..., description="Назва міста для пошуку погоди")):
    """Повернути поточну погоду в місті у форматі JSON."""
    try:
        return weather.get_current_weather(city)
    except weather.CityNotFoundError as err:
        # 404 Not Found (місто не знайдено) або 400 Bad Request (порожній ввід)
        status = 400 if not city or not city.strip() else 404
        raise HTTPException(status_code=status, detail=str(err)) from err
    except weather.ApiUnavailableError as err:
        # 502 Bad Gateway / 504 Timeout для зовнішнього API
        raise HTTPException(status_code=502, detail=str(err)) from err
    except weather.InvalidApiResponseError as err:
        # 502 Bad Gateway - некоректний формат відповіді від зовнішнього сервісу
        raise HTTPException(status_code=502, detail=str(err)) from err
    except weather.WeatherError as err:
        # Загальні помилки модуля погоди
        raise HTTPException(status_code=500, detail=str(err)) from err
