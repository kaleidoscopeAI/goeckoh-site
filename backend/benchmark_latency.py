import requests
import time
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/health"

def ping():
    start = time.perf_counter()
    try:
        response = requests.get(URL)
        end = time.perf_counter()
        return end - start, response.status_code
    except Exception as e:
        return None, str(e)

print(f"Benchmarking {URL}...")
results = []
for _ in range(10):
    latency, status = ping()
    if latency:
        results.append(latency)
        print(f"Status: {status}, Latency: {latency:.4f}s")
    else:
        print(f"Error: {status}")

if results:
    print(f"\nAverage Latency: {sum(results)/len(results):.4f}s")
