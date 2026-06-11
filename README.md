# System Prognozowania Popytu na Rowery Miejskie 

## Cel Projektu

Projekt ma na celu dostarczenie kompletnego systemu analitycznego do przewidywania zapotrzebowania na rowery miejskie w czasie rzeczywistym. Dzięki zastosowaniu modelu XGBoost (Extreme Gradient Boosting), system pozwala na optymalizację logistyki relokacji floty rowerowej na podstawie danych pogodowych oraz czasowych.

## Architektura Systemu (Tech Stack)
System został zaprojektowany jako ekosystem połączonych usług:

* **Baza Danych:** PostgreSQL (Data Lake) uruchomiony w kontenerze Docker.

* **Orkiestracja ETL:** Prefect (zautomatyzowane pobieranie, czyszczenie i inżynieria cech).

* **Tracking Modelu:** MLflow (wersjonowanie eksperymentów, modeli oraz metryk ewaluacyjnych).

* **Backend:** FastAPI (wysokowydajne API serwujące predykcje).

* **Frontend:** Streamlit (interaktywny dashboard klasy Command Center).

## Szybki Start (Docker)
Projekt jest w pełni skonteneryzowany. Nie musisz instalować lokalnie Pythona ani baz danych – wszystko zostanie automatycznie przygotowane przez **Docker**.

**Wymagania:**

* Zainstalowany [Docker Desktop](https://www.docker.com/products/docker-desktop/).

**Instrukcja uruchomienia:**

1. Sklonuj repozytorium i wejdź do folderu projektu.

2. Uruchom cały stos technologiczny jedną komendą:

```bash
docker-compose up --build
```
**Uwaga:** Przy pierwszym uruchomieniu system automatycznie pobierze dane z UCI Machine Learning Repository, wykona proces ETL oraz wytrenuje model. Może to zająć kilka minut.

## Dostęp do Usług
Po poprawnym uruchomieniu, system udostępnia następujące interfejsy:

| Usługa | Adres URL | Opis |
| :--- | :--- | :--- |
| 🚲 **Dashboard** | [http://localhost:9000](http://localhost:9000) | Interaktywny panel sterowania dla operatora. |
| 🔌 **API (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Dokumentacja i testowanie punktów końcowych API. |
| 🗄️ **pgAdmin** | [http://localhost:8080](http://localhost:8080) | Przeglądarka bazy danych (Login: `admin@admin.com`, Hasło: `admin`). |
| 📉 **MLflow** | [http://localhost:5000](http://localhost:5000) | Panel śledzenia eksperymentów i metryk modelu. |

## Wykorzystane Cechy (Features)

System wykorzystuje zarówno dane źródłowe, jak i cechy stworzone podczas procesu Feature Engineering:

* `hr` – godzina dnia.
* `temp` – temperatura.
* `hum` – wilgotność.
* `windspeed` – prędkość wiatru.
* `workingday` – dzień roboczy.
* `season` – pora roku.

Dodatkowe cechy autorskie:

* `is_rush_hour` – identyfikacja godzin szczytu komunikacyjnego.
* `cnt_lag_1h` – liczba wypożyczeń godzinę wcześniej.
* `cnt_lag_2h` – liczba wypożyczeń dwie godziny wcześniej.
* `rolling_3h` – średnia liczba wypożyczeń z ostatnich trzech godzin.
* `temp_hum_idx` – wskaźnik zależności temperatury i wilgotności.

## Model Predykcyjny

Do prognozowania liczby wypożyczeń wykorzystano model XGBoost Regressor.

Model trenowany jest na danych historycznych przechowywanych w PostgreSQL po przejściu przez proces ETL.

Wyniki ewaluacji (RMSE, MAE, R²) są automatycznie rejestrowane w MLflow, co umożliwia porównywanie kolejnych wersji modelu.

## Monitoring Systemu (Data Drift)

System zawiera zautomatyzowany moduł monitorowania stabilności danych (Data Drift) oparty na bibliotece Evidently AI. 
Analiza porównuje dane produkcyjne z danymi referencyjnymi (treningowymi), generując raport `data_drift_report.html`, który jest wbudowany bezpośrednio w zakładkę "SYSTEM MONITORING" panelu Streamlit. Pozwala to na szybką identyfikację zmian w rozkładzie zmiennych i sygnalizuje potrzebę ponownego dotrenowania modelu.

## Struktura Projektu
```
├── data/               # Surowe i przetworzone dane
├── src/
│   ├── etl/            # Pipeline Prefect (Extract, Transform, Load)
│   ├── models/         # Trening i ewaluacja modelu (MLflow)
│   └── api/            # Kod FastAPI oraz Dashboardu Streamlit
├── docker-compose.yml  # Definicja infrastruktury kontenerowej
├── Dockerfile          # Instrukcja budowy obrazu aplikacji
└── requirements.txt    # Zależności biblioteczne
```

## Schemat Architektury
![Architektura](/image/architecture1.png)

## Dashboard aplikacji

Poniżej przedstawiono główny panel operatora umożliwiający symulację scenariuszy pogodowych i czasowych oraz analizę prognozowanego popytu.

![Dashboard](/image/dashboard_view.png)

## Możliwości Rozwoju

* [ ] Dodanie danych o opóźnieniach komunikacji miejskiej.

* [ ] Konteneryzacja bazy MLflow (obecnie lokalna).


Projekt przygotowany w ramach zajęć z Hurtowni Danych

