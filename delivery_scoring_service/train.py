import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib


# 1. Генерация реалистичного датасета
def generate_data(num_samples=5000):
    np.random.seed(42)

    # Признаки
    store_id = np.random.randint(1, 11, size=num_samples)  # 10 разных ресторанов
    items_count = np.random.randint(1, 15, size=num_samples)  # Кол-во товаров в чеке
    order_price = items_count * np.random.uniform(300, 800, size=num_samples) + np.random.uniform(50, 150,
                                                                                                  size=num_samples)
    delivery_distance = np.random.uniform(0.5, 12.0, size=num_samples)  # Дистанция в км
    planned_prep_time = np.random.randint(15, 60, size=num_samples)  # Плановое время на готовку (мин)

    # Логика целевой переменной: prepared_on_time (1 - вовремя, 0 - опоздание)
    # На задержку влияет: высокая дистанция, много товаров и мало времени на готовку
    delay_score = (delivery_distance * 3) + (items_count * 1.5) - (planned_prep_time * 0.5)
    probability_on_time = 1 / (1 + np.exp(delay_score - 2))  # Сигмоида для вероятности

    prepared_on_time = (np.random.random(num_samples) < probability_on_time).astype(int)

    df = pd.DataFrame({
        'store_id': store_id,
        'items_count': items_count,
        'order_price': order_price,
        'delivery_distance': delivery_distance,
        'planned_prep_time': planned_prep_time,
        'prepared_on_time': prepared_on_time
    })
    return df


print("Шаг 1: Генерация синтетического датасета...")
df = generate_data()

# 2. Разбиение на признаки и таргет, разделение выборки
X = df.drop(columns=['prepared_on_time'])
y = df['prepared_on_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Препроцессинг данных
numeric_features = ['items_count', 'order_price', 'delivery_distance', 'planned_prep_time']
categorical_features = ['store_id']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Создание Pipeline с моделью RandomForest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Обучение и тюнинг гиперпараметров
print("Шаг 2: Настройка сетки параметров и обучение (GridSearchCV)...")
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Лучшие параметры: {grid_search.best_params_}")

# 6. Оценка качества модели
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nОтчет о классификации на тестовой выборке:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 7. Сохранение модели
os.makedirs('model', exist_ok=True)
model_path = 'model/delivery_model.joblib'
joblib.dump(best_model, model_path)
print(f"\nМодель успешно сохранена в файл: {model_path}")