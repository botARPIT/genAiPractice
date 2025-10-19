# This works same as structure based text splitting but has additional functionality of splitting documents based on their characteristics for example if a document is a piece of pyhton code, the it splits it based on certain keywords such as class, def etc which are python specific keywords and then performs normal text splitting, it does the same for other type of documents as well first splits them on the basis of keywords specific to that language and then splits them on the basis of text structure

from langchain_text_splitters import PythonCodeTextSplitter


text = '''
import time
import random
from datetime import datetime

class APIMonitor:
    """
    A simple backend monitoring tool that simulates API requests,
    records their response times, and logs the results.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.logs = []

    def simulate_request(self, method: str):
        """Simulate an API request with random latency and status."""
        start_time = time.time()
        time.sleep(random.uniform(0.05, 0.3))  # Simulate network latency
        status_code = random.choice([200, 201, 400, 404, 500])
        duration = round((time.time() - start_time) * 1000, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = {
            "method": method,
            "endpoint": self.endpoint,
            "status_code": status_code,
            "response_time_ms": duration,
            "timestamp": timestamp
        }

        self.logs.append(log_entry)
        return log_entry

    def run_monitor(self, iterations: int = 10):
        """Run multiple simulated requests and store their logs."""
        print(f"Monitoring endpoint: {self.endpoint}\n")
        for i in range(iterations):
            method = random.choice(["GET", "POST", "PUT", "DELETE"])
            log = self.simulate_request(method)
            print(f"[{log['timestamp']}] {method} {self.endpoint} → "
                  f"{log['status_code']} ({log['response_time_ms']} ms)")
        print("\nMonitoring complete.\n")

    def export_logs(self, filename="api_logs.csv"):
        """Export collected logs to a CSV file."""
        import csv
        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.logs[0].keys())
            writer.writeheader()
            writer.writerows(self.logs)
        print(f"Logs exported successfully → {filename}")


# Example usage
if __name__ == "__main__":
    monitor = APIMonitor("/api/books")
    monitor.run_monitor(iterations=5)
    monitor.export_logs()

'''

splitter = PythonCodeTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0
)

docs = splitter.split_text(text)
print(len(docs))
print(docs[10])