"""Модуль інтеграції із зовнішнім API погоди Open-Meteo.

Це єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON. Веб-рівень (`app/main.py`) отримує
звідси готовий результат або зрозумілу помилку і нічого не знає про
`requests`.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT = 5.0  # Секунди


class WeatherError(Exception):
    """Базовий виняток для помилок модуля погоди."""

    pass


class CityNotFoundError(WeatherError):
    """Виняток: місто не знайдено або вказано порожню назву."""

    pass


class ApiUnavailableError(WeatherError):
    """Виняток: зовнішній сервіс недоступний, таймаут або 5xx помилка."""

    pass


class InvalidApiResponseError(WeatherError):
    """Виняток: некоректна або неочікувана структура JSON від API."""

    pass


def find_city(name: str) -> dict:
    """Знайти координати міста за його назвою через Open-Meteo Geocoding API.

    Returns:
        dict із ключами "name", "country", "latitude", "longitude".

    Raises:
        CityNotFoundError: якщо назва порожня або місто не знайдено.
        ApiUnavailableError: якщо стався мережевий збій чи таймаут.
        InvalidApiResponseError: якщо відповідь має несподіваний формат.
    """
    clean_name = name.strip() if name else ""
    if not clean_name:
        raise CityNotFoundError("Будь ласка, введіть назву міста")

    params = {
        "name": clean_name,
        "count": 1,
        "language": "uk",
        "format": "json",
    }

    try:
        response = requests.get(GEOCODING_URL, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.Timeout as exc:
        raise ApiUnavailableError("Час очікування відповіді сервісу геокодування вичерпано") from exc
    except requests.RequestException as exc:
        raise ApiUnavailableError("Не вдалося з'єднатися із сервісом геокодування") from exc

    if response.status_code >= 500:
        raise ApiUnavailableError(f"Сервіс геокодування тимчасово недоступний (HTTP {response.status_code})")
    if response.status_code >= 400:
        raise InvalidApiResponseError(f"Помилка запиту геокодування (HTTP {response.status_code})")

    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidApiResponseError("Некоректний JSON у відповіді сервісу геокодування") from exc

    results = data.get("results")
    if not results or not isinstance(results, list):
        raise CityNotFoundError(f"Місто '{clean_name}' не знайдено")

    first_match = results[0]
    lat = first_match.get("latitude")
    lon = first_match.get("longitude")

    if lat is None or lon is None:
        raise InvalidApiResponseError("Сервіс геокодування не повернув координати міста")

    return {
        "name": first_match.get("name", clean_name),
        "country": first_match.get("country", ""),
        "latitude": lat,
        "longitude": lon,
    }


def get_current_weather(city: str) -> dict:
    """Отримати поточну погоду (температуру й швидкість вітру) для міста.

    Returns:
        dict з інформацією про місто, температуру та вітер.

    Raises:
        WeatherError (або її підкласи): при будь-якій помилці виконання.
    """
    city_info = find_city(city)

    params = {
        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],
        "current": ["temperature_2m", "wind_speed_10m"],
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.Timeout as exc:
        raise ApiUnavailableError("Час очікування відповіді сервісу прогнозу погоди вичерпано") from exc
    except requests.RequestException as exc:
        raise ApiUnavailableError("Не вдалося з'єднатися із сервісом прогнозу погоди") from exc

    if response.status_code >= 500:
        raise ApiUnavailableError(f"Сервіс прогнозу погоди тимчасово недоступний (HTTP {response.status_code})")
    if response.status_code >= 400:
        raise InvalidApiResponseError(f"Помилка запиту прогнозу погоди (HTTP {response.status_code})")

    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidApiResponseError("Некоректний JSON у відповіді сервісу прогнозу погоди") from exc

    current = data.get("current")
    if not isinstance(current, dict):
        raise InvalidApiResponseError("У відповіді сервісу прогнозу відсутній блок 'current'")

    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")

    if temp is None or wind is None:
        raise InvalidApiResponseError("У відповіді сервісу відсутні потрібні показники погоди")

    units = data.get("current_units", {})
    temp_unit = units.get("temperature_2m", "°C")
    wind_unit = units.get("wind_speed_10m", "км/год")

    return {
        "city": city_info["name"],
        "country": city_info["country"],
        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],
        "temperature": temp,
        "temperature_unit": temp_unit,
        "wind_speed": wind,
        "wind_speed_unit": wind_unit,
    }
