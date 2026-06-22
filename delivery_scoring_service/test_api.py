import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("Проверка эндпоинта /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Код ответа: {response.status_code}")
    print(f"Тело ответа: {response.json()}\n")

def test_predict_success():
    print("Отправка корректных данных на /predict...")
    payload = {
        "store_id": 3,
        "items_count": 4,
        "order_price": 1500.0,
        "delivery_distance": 2.5,
        "planned_prep_time": 40
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Код ответа: {response.status_code}")
    print(f"Тело ответа: {response.json()}\n")

def test_predict_validation_error():
    print("Отправка некорректных данных для проверки валидации (отрицательная дистанция)...")
    payload = {
        "store_id": 3,
        "items_count": 4,
        "order_price": 1500.0,
        "delivery_distance": -5.0,  # Недопустимо по правилам валидации
        "planned_prep_time": 40
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Код ответа: {response.status_code} (Ожидается 422)")
    print(f"Тело ответа: {json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")

if __name__ == "__main__":
    # Убедитесь, что ваш FastAPI сервис запущен перед запуском тестов!
    try:
        test_health()
        test_predict_success()
        test_predict_validation_error()
    except requests.exceptions.ConnectionError:
        print("Ошибка: FastAPI сервер не запущен. Пожалуйста, запустите сначала 'uvicorn main:app --reload'")