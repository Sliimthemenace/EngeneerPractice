import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow as mf
import mlflow.catboost
from mlflow.tracking import MlflowClient

mf.set_tracking_uri('http://127.0.0.1:5000')
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'TRUE'

experiment_id = mf.set_experiment('test')

# Загрузка данных
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.select_dtypes(exclude=['object', 'category'])
    return data

# Разделение данных на обучающую и тестовую выборки
def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Обучение модели CatBoost
def train_catboost(X_train, y_train):
    model = CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=6, verbose=False)
    model.fit(X_train, y_train)
    return model

# Оценка модели
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Основной пайплайн
def main():
    file_path = "data/train.csv"
    target_column = "Survived"

    data = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    # Начало MLflow эксперимента
    with mlflow.start_run():
        model = train_catboost(X_train, y_train)
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

        # Логирование метрик в MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Логирование параметров модели
        mlflow.log_params({
            "model": "CatBoost",
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 6
        })

        # Сохранение модели в MLflow
        mlflow.catboost.log_model(model, "catboost_model")

        print(f"Metrics logged in MLflow: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
