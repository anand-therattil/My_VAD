import requests

url = "http://localhost:8001/vad"

file_path = "data/mixed_10db/mix_0000.wav"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
