import pandas as pd
from evidently import Report                  
from evidently.presets import DataDriftPreset 

df = pd.read_csv("data/hour.csv")
reference_data = df.iloc[:10000] 
current_data = df.iloc[10000:]   

drift_report = Report(metrics=[DataDriftPreset()])

my_eval = drift_report.run(reference_data=reference_data, current_data=current_data)

my_eval.save_html("data_drift_report.html")

print("Raport Evidently AI wygenerowany pomyślnie!")