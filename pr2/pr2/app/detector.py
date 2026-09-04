"""Модуль inference: єдине місце застосунку, яке знає про модель YOLOv8.

Тут живуть ваги, поріг упевненості й формат результату детекції.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає про `ultralytics`.
"""

import io
import time
from typing import Dict, List, Optional
from PIL import Image
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

# Глобальне сховище для синглтон-екземпляра моделі
_model: Optional[YOLO] = None


class DetectionError(Exception):
    """Базовий виняток для помилок модуля детекції."""

    pass


class InvalidImageError(DetectionError):
    """Виняток: невалідний, пошкоджений або порожній файл зображення."""

    pass


class ModelInferenceError(DetectionError):
    """Виняток: помилка виконання обчислень моделі."""

    pass


def load_model() -> YOLO:
    """Завантажити модель у пам'ять один раз за життя застосунку."""
    global _model
    if _model is None:
        try:
            _model = YOLO(WEIGHTS)
        except Exception as exc:
            raise ModelInferenceError(f"Не вдалося завантажити ваги моделі '{WEIGHTS}': {exc}") from exc
    return _model


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE) -> Dict:
    """Запустити детекцію об'єктів на зображенні.

    Args:
        image_bytes: Байти файлу зображення.
        confidence: Поріг упевненості (0.01 - 1.0).

    Returns:
        Словник із результатами детекції, кількістю об'єктів та часом виконання.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise InvalidImageError("Файл порожній або не був переданий")

    # Декодування зображення через PIL
    try:
        raw_img = Image.open(io.BytesIO(image_bytes))
        image = raw_img.convert("RGB")
    except Exception as exc:
        raise InvalidImageError("Файл не є підтримуваним або валідним зображенням") from exc

    width, height = image.size
    if width == 0 or height == 0:
        raise InvalidImageError("Зображення має нульові розміри")

    model = load_model()

    # Замір часу виконання (inference time)
    start_time = time.perf_counter()
    try:
        results = model(image, conf=confidence, verbose=False)
    except Exception as exc:
        raise ModelInferenceError(f"Помилка під час обчислень моделі: {exc}") from exc
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    objects: List[Dict] = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                class_name = model.names.get(cls_id, f"class_{cls_id}")

                # Відносні координати у % для малювання у браузері
                left_pct = round((x1 / width) * 100, 2)
                top_pct = round((y1 / height) * 100, 2)
                width_pct = round(((x2 - x1) / width) * 100, 2)
                height_pct = round(((y2 - y1) / height) * 100, 2)

                objects.append({
                    "id": idx + 1,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf_val, 4),
                    "confidence_percent": round(conf_val * 100, 1),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_normalized": {
                        "left_pct": left_pct,
                        "top_pct": top_pct,
                        "width_pct": width_pct,
                        "height_pct": height_pct,
                    },
                })

    return {
        "objects": objects,
        "count": len(objects),
        "image_width": width,
        "image_height": height,
        "inference_time_ms": elapsed_ms,
        "confidence_threshold": confidence,
    }
