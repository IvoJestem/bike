from prefect import task, flow
import pandas as pd
from sqlalchemy import create_engine
import requests
import zipfile
import io

DB_URL = "postgresql://admin:secretpassword@db:5432/bike_db"

@task(retries=3, retry_delay_seconds=10)
def extract_data():
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        df = pd.read_csv(zip_ref.open("hour.csv"))
    return df

@task
def transform_data(df):
    df['is_rush_hour'] = df.apply(
        lambda x: 1 if x['workingday'] == 1 and (7 <= x['hr'] <= 9 or 16 <= x['hr'] <= 19) else 0, 
        axis=1
    )
    weather_map = {1: 'Clear', 2: 'Mist', 3: 'Light_Rain_Snow', 4: 'Heavy_Rain_Ice'}
    df['weather_desc'] = df['weathersit'].map(weather_map)
    
    return df

@task
def load_data(df, table_name):

    engine = create_engine(DB_URL)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    return f"Załadowano {len(df)} wierszy do {table_name}"

@flow(name="Bike Sharing ETL Pipeline")
def bike_sharing_flow():
    raw_df = extract_data()
    clean_df = transform_data(raw_df)
    status = load_data(clean_df, "processed_bikeshare")
    print(status)

if __name__ == "__main__":
    bike_sharing_flow()