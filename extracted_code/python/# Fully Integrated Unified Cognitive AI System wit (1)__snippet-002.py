import requests

response = requests.get("http://localhost:5000/status")
metrics = response.json()

print("System health:", metrics["health"])
