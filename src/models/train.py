import pandas as pd
import mlflow
import mlflow.xgboost
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os
from sqlalchemy import create_engine

DB_URL = os.getenv("DB_URL", "postgresql://admin:secretpassword@localhost:5432/bike_db")

engine = create_engine(DB_URL)

df = pd.read_sql("SELECT * FROM processed_bikeshare", engine)

df = df.sort_values(['dteday', 'hr']) if 'dteday' in df.columns else df


df['cnt_lag_1h'] = df['cnt'].shift(1)
df['cnt_lag_2h'] = df['cnt'].shift(2)
df['rolling_3h'] = df['cnt'].shift(1).rolling(window=3).mean()

df['temp_hum_idx'] = df['temp'] * df['hum']

df = df.dropna()

features = ['hr', 'temp', 'hum', 'windspeed', 'workingday', 'is_rush_hour', 'season', 
            'cnt_lag_1h', 'cnt_lag_2h', 'rolling_3h', 'temp_hum_idx']
X = df[features]
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

mlflow.set_experiment("Bike_Sharing_Demand_XGBoost")

with mlflow.start_run():
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds) # To podbije Twoje "AI Confidence"
    
    print(f"XGBoost Ready! RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.xgboost.log_model(model, "model")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/bike_model.joblib")