import requests
import pandas as pd
import json

# URL model yang sedang serve
url = "http://127.0.0.1:5000/invocations"

# Data uji 
data = pd.DataFrame([{
    "Year": 2016.0,
    "Genre": 5,
    "Publisher": 143,
    "North America": 0.0,
    "Europe": 0.0,
    "Japan": 0.0,
    "Rest of World": 0.0
}])

headers = {"Content-Type": "application/json"}

response = requests.post(
    url,
    data=json.dumps({"dataframe_records": data.to_dict(orient="records")}),
    headers=headers
)

print("Prediction:", response.json())
