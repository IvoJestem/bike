import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql://admin:secretpassword@db:5432/bike_db"

def load_data():
    try:
        engine = create_engine(DB_URL)
        
        print("Wczytywanie danych z pliku...")
        df = pd.read_csv("data/hour.csv")
        
        print("Wysyłanie danych do bazy PostgreSQL w Dockerze...")
        df.to_sql('raw_bikeshare_hour', engine, if_exists='replace', index=False)
        
        print("Sukces! Dane zostały załadowane do tabeli 'raw_bikeshare_hour'.")
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    load_data()