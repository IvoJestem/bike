#!/bin/bash
set -e

MODEL_FILE="models/bike_model.joblib"

echo "Startujemy system..."

echo "KROK 1: Uruchamiam usługi pomocnicze (MLflow & API)..."
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

if [ -f "$MODEL_FILE" ]; then
    echo "Znaleziono gotowy model ($MODEL_FILE). Pomijam trening."
else
    echo "Brak modelu. Uruchamiam pełną procedurę (to zajmie chwilę)..."
    echo "-> ETL..."
    python src/etl/workflow.py
    echo "-> Migracja bazy MLflow..."
    mlflow db upgrade sqlite:///mlflow.db
    echo "-> Trening..."
    python src/models/train.py
fi

echo "KROK 2: Odpalam Dashboard Streamlit..."
streamlit run src/api/dashboard.py --server.port 8501 --server.address 0.0.0.0