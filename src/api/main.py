from fastapi import FastAPI
import pandas as pd
import joblib
import os
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Bike Sharing Prediction API | XGBoost Edition")

MODEL_PATH = os.path.join("models", "bike_model.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Sukces: Model XGBoost załadowany z {MODEL_PATH}")
else:
    print(f"OSTRZEŻENIE: Nie znaleziono modelu w {MODEL_PATH}")
    model = None

class BikePredictionInput(BaseModel):
    hr: int
    temp: float
    hum: float
    windspeed: float
    workingday: int
    is_rush_hour: int
    season: int

@app.get("/")
def home():
    return {
        "message": "API działa!", 
        "model_status": "loaded" if model else "not_found",
        "engine": "XGBoost"
    }

@app.post("/predict")
def predict(input_data: BikePredictionInput):
    if model is None:
        return {"error": "Model nie jest załadowany na serwerze."}

    data = pd.DataFrame([input_data.model_dump()])

    data['temp_hum_idx'] = data['temp'] * data['hum']
    

    data['cnt_lag_1h'] = 150.0 
    data['cnt_lag_2h'] = 140.0
    data['rolling_3h'] = 145.0


    cols_order = [
        'hr', 'temp', 'hum', 'windspeed', 'workingday', 'is_rush_hour', 'season',
        'cnt_lag_1h', 'cnt_lag_2h', 'rolling_3h', 'temp_hum_idx'
    ]
    
    final_data = data[cols_order]

    prediction = model.predict(final_data)[0]

    safe_prediction = max(0, int(round(prediction)))
    
    return {
        "predicted_bike_count": safe_prediction,
        "engine": "XGBoost_V3",
        "features_used": len(cols_order)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)