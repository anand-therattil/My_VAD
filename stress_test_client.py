import requests
import time
import concurrent.futures
import statistics

URL = "http://localhost:8001/vad"
FILE_PATH = "data/mixed_10db/mix_0000.wav"

NUM_REQUESTS = 100        # total requests
CONCURRENT_WORKERS = 50  # parallel threads


def send_request():

    with open(FILE_PATH, "rb") as f:
        files = {"file": f}

        start = time.time()
        response = requests.post(URL, files=files)
        end = time.time()

        latency = end - start

        if response.status_code != 200:
            return None

        return latency


def main():

    print(f"Sending {NUM_REQUESTS} requests "
          f"with {CONCURRENT_WORKERS} concurrent workers...")

    latencies = []

    start_total = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = [executor.submit(send_request) for _ in range(NUM_REQUESTS)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                latencies.append(result)

    end_total = time.time()

    if not latencies:
        print("All requests failed.")
        return

    print("\n--- Stress Test Results ---")
    print(f"Total time: {end_total - start_total:.2f} sec")
    print(f"Successful requests: {len(latencies)}")
    print(f"Average latency: {statistics.mean(latencies):.4f} sec")
    print(f"Median latency: {statistics.median(latencies):.4f} sec")
    print(f"Min latency: {min(latencies):.4f} sec")
    print(f"Max latency: {max(latencies):.4f} sec")


if __name__ == "__main__":
    main()
