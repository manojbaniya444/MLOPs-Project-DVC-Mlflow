import json
import requests

base_url = "http://localhost:8000"

x_new = dict (
    ph = 10.20,
    Hardness = 500.01,
    Solids = 20000.01,
    Chloramines = 7.5,
    Sulfate = 368.01,
    Conductivity = 654.12,
    Organic_carbon = 10.3,
    Trihalomethanes = 86.1,
    Turbidity = 2.9,
)

x_new_json = json.dumps(x_new)

response = requests.post(url=f"{base_url}/predict", json=x_new)
print("Response: ", response.text)
print("Status Code: ", response.status_code)