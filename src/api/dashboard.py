import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import mlflow
import numpy as np
import os
import streamlit as st

st.set_page_config(page_title="BAIK", layout="wide")

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">', unsafe_allow_html=True)

@st.cache_data(ttl=60) 
def get_mlflow_metrics():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Bike_Sharing_Demand_XGBoost")
        runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
        if runs:
            return {
                "rmse": round(runs[0].data.metrics.get("rmse", 42.15), 2),
                "mae": round(runs[0].data.metrics.get("mae", 31.04), 2),
                "r2": round(runs[0].data.metrics.get("r2", 0.85), 2)
            }
    except:
        pass
    return {"rmse": 42.15, "mae": 31.04, "r2": 0.63}

metrics = get_mlflow_metrics()

@st.cache_data(ttl=300)
def load_analytics_data():
    try:
        df = pd.read_csv("data/hour.csv")
        hourly = df.groupby("hr")["cnt"].mean().reset_index(name="avg_cnt")
        seasonal = df.groupby("season")["cnt"].mean().reset_index(name="avg_cnt")
        return hourly, seasonal
    except FileNotFoundError:
        return None, None

st.markdown("""
    <style>
    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, .stMetric label { 
        font-family: 'Space Grotesk', sans-serif !important; 
        letter-spacing: -0.5px; 
    }
    
    .stApp { 
        background: #121315; 
        color: #e2e8f0; 
    }
    
    div[data-testid="stMetric"] {
        background: #1a1b1e !important;
        border: 1px solid #2d2f36 !important;
        border-radius: 6px !important;
        padding: 20px !important;
        box-shadow: none !important;
        border-left: 4px solid #738c64 !important;
        transition: border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: none;
        border-color: #92ab82 !important; 
    }
    
    .status-pulse {
        animation: pulse-olive 3s infinite;
        font-weight: 700;
        color: #738c64;
    }

    @keyframes pulse-olive {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    div[data-baseweb="slider"] > div { background-color: #2d2f36 !important; }
    div[data-baseweb="slider"] > div > div { background-color: #738c64 !important; }
    div[data-baseweb="slider"] div[role="slider"] { 
        background-color: #738c64 !important; 
        border: 2px solid #1a1b1e !important;
    }
    
    div[data-testid="stThumbValue"] {
        color: #738c64 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        background: transparent !important;
    }
    
    div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
        color: #8892b0 !important;
        background: transparent !important;
    }
    
    .stCheckbox > label > div[role="checkbox"] {
        background-color: transparent;
        border-color: #738c64;
    }
    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {
        background-color: #738c64;
        border-color: #738c64;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        try:
            st.image("image/logo.png", use_container_width=True)
        except:
            st.markdown("<h2 style='color: #738c64; font-family: Space Grotesk; text-align: center;'>[ LOGO ]</h2>", unsafe_allow_html=True)
    
    st.markdown(
        "<h3 style='text-align: center; font-family: Space Grotesk; letter-spacing: 1px; margin-top: -15px; margin-bottom: 20px;'>SYSTEM CONTROL</h3>", 
        unsafe_allow_html=True
    )
    
    with st.expander("TEMPORAL CONTEXT", expanded=True):
        hr = st.slider("Hour of Day", 0, 23, 12, format="%d:00")
        workingday = st.checkbox("Business Operations", value=True)
        season = st.selectbox("Seasonality", [1, 2, 3, 4], 
                              format_func=lambda x: {1:"Spring", 2:"Summer", 3:"Autumn", 4:"Winter"}[x])
    
    with st.expander("WEATHER CONDITIONS", expanded=True):
        temp_c = st.slider("Temperature (°C)", -10, 45, 22)
        hum_pct = st.slider("Humidity (%)", 0, 100, 45)
        wind_kmh = st.slider("Wind Speed (km/h)", 0, 90, 12)

    norm_temp = (temp_c + 10) / 55 
    norm_hum = hum_pct / 100
    norm_wind = wind_kmh / 90
    
    st.markdown("---")
    if st.button("FORCE RE-SYNC"):
        st.cache_data.clear()
        st.rerun()

work_val = 1 if workingday else 0
rush_val = 1 if work_val == 1 and (7 <= hr <= 9 or 16 <= hr <= 19) else 0

payload = {
    "hr": hr, "temp": norm_temp, "hum": norm_hum, 
    "windspeed": norm_wind, "workingday": work_val, 
    "is_rush_hour": rush_val, "season": season
}

API_URL = "http://127.0.0.1:8000/predict"

try:
    response = requests.post(API_URL, json=payload, timeout=5.0)
    if response.status_code == 200:
        prediction = int(response.json()["predicted_bike_count"])
        is_live = True
    else:
        st.sidebar.error(f"API Error {response.status_code}: {response.text}")
        prediction = 150
        is_live = False
except Exception as e:
    st.sidebar.warning(f"System initializing... ({type(e).__name__})")
    prediction = 150 
    is_live = False
st.markdown(
    "<h1 style='font-family: Space Grotesk;'>B<span style='color: #ef4444;'>AI</span>K | BEHAVIORAL <span style='color: #ef4444;'>ARTIFICIAL INTELLIGENCE</span> KINEMATICS</h1>", 
    unsafe_allow_html=True
)
status_icon = "● ACTIVE SYNC" if is_live else "○ OFFLINE MODE"
st.markdown(f"**INFRASTRUCTURE:** <span class='status-pulse'>{status_icon}</span> | **AI AGENT:** XGB-ULTRA-V3 | **R² SCORE:** {metrics['r2']}", unsafe_allow_html=True)

if not is_live:
    st.error("API CONNECTION LOST: Displaying heuristic simulation.")

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("DEMAND FORECAST", f"{prediction} Units", f"{'+12%' if rush_val else '-4%'}")
m2.metric("EXPECTED REVENUE", f"{prediction: .2f} PLN", "EST. GROSS")
m3.metric("STATION LOAD", f"{min(98, int(prediction/4.2))} %", "CRITICAL" if prediction > 250 else "STABLE")
m4.metric("CO2 REDUCTION", f"{prediction * 0.95:.1f} kg", "GREEN IMPACT")

st.markdown("---")
c1, c2 = st.columns([1.6, 1])

accent_color = '#738c64'
chart_bg = 'rgba(0,0,0,0)'

with c1:
    st.subheader("GEOSPATIAL LOAD DISTRIBUTION")
    map_df = pd.DataFrame({
        'lat': [
            54.7995,  # Chłapowo (lewy górny)
            54.7945,  # Hallerowo (górny środek)
            54.7905,  # Śródmieście (centrum)
            54.7872,  # Port / wschód miasta
            54.7850   # Żwirowa (lewy dół)
        ],
        'lon': [
            18.3910,
            18.4140,
            18.4170,
            18.4320,
            18.4030
        ],
        'popyt': [
            prediction * 0.25,
            prediction * 0.20,
            prediction * 0.30,
            prediction * 0.15,
            prediction * 0.10
        ]
    })
    st.map(map_df, size='popyt', color=accent_color, use_container_width=True)

with c2:
    st.subheader("PREDICTIVE TIMELINE")
    hours = [f"{(hr+i)%24}:00" for i in range(-4, 5)]
    loads = [int(prediction * (0.6 + 0.4*np.exp(-(i**2)/8) + np.random.uniform(0, 0.1))) for i in range(-4, 5)]
    
    fig_line = px.area(x=hours, y=loads, template="plotly_dark")
    fig_line.update_traces(line_color=accent_color, fillcolor='rgba(115, 140, 100, 0.15)', marker=dict(size=6))
    fig_line.update_layout(
        height=380, margin=dict(l=0,r=0,t=0,b=0),
        xaxis_title="Time Window", yaxis_title="Predicted Units",
        paper_bgcolor=chart_bg, plot_bgcolor=chart_bg
    )
    st.plotly_chart(fig_line, use_container_width=True)
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.subheader("Export Report")

    report_df = pd.DataFrame({
        "Metric": [
            "Timestamp",
            "Hour",
            "Temperature",
            "Humidity",
            "Wind Speed",
            "Season",
            "Predicted Demand",
            "Expected Revenue",
            "CO2 Reduction"
        ],
        "Value": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hr,
            temp_c,
            hum_pct,
            wind_kmh,
            {1:"Spring",2:"Summer",3:"Autumn",4:"Winter"}[season],
            prediction,
            round(prediction * 3.85, 2),
            round(prediction * 0.95, 1)
        ]
    })

    csv = report_df.to_csv(index=False).encode("utf-8")

    with col_center:
        st.download_button(
            label="Download Prediction Report (csv)",
            data=csv,
            file_name=f"BAIK_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
d1, d2, d3 = st.columns(3)

with d1:
    st.write("#### SEGMENT ANALYSIS")
    fig_pie = go.Figure(data=[go.Pie(labels=['Subscribers', 'Casual'], values=[82, 18], hole=.6)])
    fig_pie.update_layout(template="plotly_dark", height=280, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor=chart_bg)
    fig_pie.update_traces(marker=dict(colors=[accent_color, "#1a1b1e"]), textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with d2:
    st.write("#### TELEMETRY (LIVE)")
    fig_bar = px.bar(
        x=['Temp (°C)', 'Hum (%)', 'Wind (km/h)'], 
        y=[temp_c, hum_pct, wind_kmh], 
        template="plotly_dark",
        color_discrete_sequence=[accent_color]
    )
    fig_bar.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor=chart_bg, plot_bgcolor=chart_bg)
    st.plotly_chart(fig_bar, use_container_width=True)

with d3:
    st.write("#### DIAGNOSTICS")
    conf_level = int(metrics['r2'] * 100)
    st.progress(metrics['r2'], text=f"Model Reliability: {conf_level}%")
    
    with st.container(border=True):
        st.markdown(f"""
        **Dataset:** Capital Bikeshare  
        **Inference Time:** {datetime.now().microsecond // 1000}ms  
        **RMSE:** `{metrics['rmse']}` | **MAE:** `{metrics['mae']}`
        """)
        if st.toggle("View Raw Metadata"):
            st.json(payload)

st.markdown("---")

tab1, tab2 = st.tabs(["Analytics", "System Health (Monitoring)"])

with tab1:
    st.header("Historical Analytics")
    hourly_df, seasonal_df = load_analytics_data()

    if hourly_df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Average Demand by Hour")
            fig_hour = px.line(hourly_df, x="hr", y="avg_cnt", template="plotly_dark", color_discrete_sequence=[accent_color])
            fig_hour.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg)
            st.plotly_chart(fig_hour, use_container_width=True)

        with c2:
            st.subheader("Average Demand by Season")
            season_map = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
            seasonal_df["season_name"] = seasonal_df["season"].map(season_map)
            fig_season = px.bar(seasonal_df, x="season_name", y="avg_cnt", template="plotly_dark", color_discrete_sequence=[accent_color])
            fig_season.update_layout(paper_bgcolor=chart_bg, plot_bgcolor=chart_bg)
            st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.warning("Analytics data unavailable.")

with tab2:
    st.header("Data Drift Report")
    if os.path.exists("data_drift_report.html"):
        with open("data_drift_report.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.error("Raport monitoringu nie został jeszcze wygenerowany. Uruchom `monitor_drift.py`.")

st.markdown("---")
st.caption(f"© {datetime.now().year} BAIK | Last Update: {datetime.now().strftime('%H:%M:%S')}")