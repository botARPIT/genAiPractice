# Has the same working as document based text splitter but specialized to work for markdown files
from langchain_text_splitters import MarkdownTextSplitter

text = '''
# Mock API Response Testing Report

## Overview
The concept of **backend testing** continues to evolve as systems grow in complexity and scale.  
Modern infrastructures rely on well-structured APIs that connect services, transfer data, and ensure stability across environments.  
Testing these APIs effectively requires a combination of automation, validation, and monitoring.

This document provides a **sample Markdown text** that simulates a technical report used during API performance analysis.  
It can be used to test markdown parsing, rendering engines, or text ingestion pipelines.

---

## Objectives
- Validate the correctness of API responses  
- Measure latency and success rates  
- Identify patterns in error codes  
- Generate reproducible datasets for regression testing  

---

## Data Summary
The dataset includes mock responses for `/api/books`, representing typical backend activity.

| request_id | endpoint   | method | status_code | response_time_ms | timestamp           |
|-------------|-------------|---------|--------------|------------------|--------------------|
| req_0001    | /api/books  | GET     | 200          | 132              | 2025-10-19 12:05:32 |
| req_0002    | /api/books  | POST    | 201          | 287              | 2025-10-19 12:03:11 |
| req_0003    | /api/books  | GET     | 404          | 511              | 2025-10-19 11:58:46 |
| req_0004    | /api/books  | DELETE  | 403          | 729              | 2025-10-19 11:45:17 |
| req_0005    | /api/books  | PUT     | 500          | 1156             | 2025-10-19 11:22:58 |

> *Note: The data above is synthetic and created solely for backend test simulation.*

---

## Observations
1. The majority of requests completed successfully (`200` and `201` status codes).  
2. Average latency for successful responses remained below 400 ms.  
3. Failed requests (`4xx` or `5xx`) showed higher response times, suggesting possible retries or processing bottlenecks.  
4. Response distribution appears balanced across request methods — though **DELETE** and **PUT** show occasional spikes.  

---

## Insights
The analysis indicates a strong correlation between **response time** and **status code severity**.  
Under load, endpoints may experience queueing delays, especially for write-heavy operations.  
Implementing **asynchronous handlers**, **bulk request batching**, or **caching layers** could significantly reduce this latency.

Developers should also:
- Log both request and response metadata.
- Track request IDs end-to-end for observability.
- Monitor 5xx error clusters for infrastructure-level issues.

---

## Next Steps
- Integrate response metrics into Prometheus + Grafana dashboards.  
- Expand dataset to include `/api/users` and `/api/orders`.  
- Run automated regression tests against new API versions.  
- Refactor error handling logic for better resilience.

---

## Conclusion
Backend testing forms the foundation of reliable software delivery.  
By combining well-structured test data with automated analysis pipelines, teams can ensure both performance and correctness.  
The insights gathered from even synthetic datasets like this one guide architecture decisions, improve error visibility, and promote continuous optimization.

---

*Generated for backend testing and documentation simulation — October 2025.*

'''

splitter = MarkdownTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
)

docs = splitter.split_text(text)
print(len(docs))
print(docs[20])
print(docs[21])