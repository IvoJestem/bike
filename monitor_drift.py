import pandas as pd
from evidently import Report          
from evidently.presets import DataDriftPreset 

df = pd.read_csv("data/hour.csv")

reference_data = df.iloc[:10000] 
current_data = df.iloc[10000:]   

features_to_monitor = [
    'season', 'hr', 'holiday', 'weekday', 'workingday', 
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 
    'casual', 'registered', 'cnt'
]

ref_filtered = reference_data[features_to_monitor]
curr_filtered = current_data[features_to_monitor]

drift_report = Report(metrics=[DataDriftPreset()])

my_eval = drift_report.run(reference_data=ref_filtered, current_data=curr_filtered)

my_eval.save_html("data_drift_report.html")

print("Oczyszczony raport Evidently AI wygenerowany pomyślnie!")