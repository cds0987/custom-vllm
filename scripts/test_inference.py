import time
import requests

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
}

start = time.time()
resp = requests.post(URL, json=payload, timeout=120)
elapsed = time.time() - start

resp.raise_for_status()
data = resp.json()

print("=== vLLM smoke test ===")
print(f"HTTP status: {resp.status_code}")
print(f"Latency: {elapsed:.2f}s")
print(f"Response: {data['choices'][0]['message']['content']}")
print(f"Usage: {data.get('usage')}")
print("=== PASSED ===")
