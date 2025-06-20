from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import requests
import pandas as pd
import json

REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
REQUEST_LATENCY = Summary('inference_request_latency_seconds', 'Latency of inference requests')
REQUEST_SUCCESS = Counter('inference_success_total', 'Total number of successful inferences')
REQUEST_FAILURE = Counter('inference_failure_total', 'Total number of failed inferences')
LAST_INFERENCE_TIME = Gauge('last_inference_time_seconds', 'Timestamp of last inference request')

url = "http://127.0.0.1:5000/invocations"
headers = {"Content-Type": "application/json"}

data = pd.DataFrame([{
    "Year": 2016.0,
    "Genre": 5,
    "Publisher": 143,
    "North America": 0.0,
    "Europe": 0.0,
    "Japan": 0.0,
    "Rest of World": 0.0
}])

@REQUEST_LATENCY.time()
def make_prediction():
    REQUEST_COUNT.inc()
    try:
        response = requests.post(
            url,
            data=json.dumps({"dataframe_records": data.to_dict(orient="records")}),
            headers=headers
        )
        if response.status_code == 200:
            REQUEST_SUCCESS.inc()
        else:
            REQUEST_FAILURE.inc()
        print("Prediction:", response.json())
    except Exception as e:
        REQUEST_FAILURE.inc()
        print("Error:", e)
    LAST_INFERENCE_TIME.set_to_current_time()

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        make_prediction()
        time.sleep(10)
