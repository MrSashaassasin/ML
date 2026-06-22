from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Delivery Scoring Service API",
    description="Сервис для оценки вероятности своевременной доставки заказов",
    version="1.0.0"
)

# Путь к модели
MODEL_PATH = "model/delivery_model.joblib"
model = None


# Загрузка модели при старте приложения
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("--- Модель успешно загружена! ---")
    else:
        raise FileNotFoundError(
            f"Файл модели не найден по пути {MODEL_PATH}. "
            f"Пожалуйста, запустите сначала train.py для обучения модели."
        )


# Валидация входных данных
class OrderFeatures(BaseModel):
    store_id: int = Field(..., description="ID магазина/ресторана", ge=1, le=100)
    items_count: int = Field(..., description="Количество товаров в заказе", ge=1, le=50)
    order_price: float = Field(..., description="Общая стоимость заказа в рублях", ge=0.0)
    delivery_distance: float = Field(..., description="Дистанция доставки в километрах", ge=0.1, le=50.0)
    planned_prep_time: int = Field(..., description="Запланированное время приготовления в минутах", ge=5, le=120)

    # Пример для автодокументации Swagger
    model_config = {
        "json_schema_extra": {
            "example": {
                "store_id": 5,
                "items_count": 3,
                "order_price": 1250.50,
                "delivery_distance": 3.2,
                "planned_prep_time": 30
            }
        }
    }


# 1. Эндпоинт проверки работоспособности
@app.get("/health", summary="Проверка работоспособности сервиса")
def health():
    return {"status": "ok"}


# 2. Эндпоинт инференса (предсказания)
@app.post("/predict", summary="Расчет вероятности доставки вовремя")
def predict(order: OrderFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена на сервере.")

    try:
        # Преобразуем входящие данные в DataFrame (какие ожидает sklearn Pipeline)
        input_data = pd.DataFrame([{
            'store_id': order.store_id,
            'items_count': order.items_count,
            'order_price': order.order_price,
            'delivery_distance': order.delivery_distance,
            'planned_prep_time': order.planned_prep_time
        }])

        # Получаем предсказание класса и вероятностей
        prediction_class = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        # Вероятность класса 1 (доставка вовремя)
        probability_on_time = float(probabilities[1])

        return {
            "probability_on_time": round(probability_on_time, 4),
            "class": prediction_class
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса модели: {str(e)}")