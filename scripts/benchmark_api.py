from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure VelaCL HTTP latency on named hardware")
    parser.add_argument("--url", default="http://localhost:8000/v1/predict")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    payload = json.dumps({"texts": ["hello, I need help"] * args.batch_size}).encode()
    latencies = []
    started = time.perf_counter()
    for _ in range(args.requests):
        request = urllib.request.Request(
            args.url, data=payload, headers={"content-type": "application/json"}
        )
        before = time.perf_counter()
        with urllib.request.urlopen(request) as response:
            response.read()
        latencies.append((time.perf_counter() - before) * 1000)
    duration = time.perf_counter() - started
    ordered = sorted(latencies)

    def percentile(value: float) -> float:
        return ordered[min(int(value * len(ordered)), len(ordered) - 1)]

    print(
        json.dumps(
            {
                "requests": args.requests,
                "batch_size": args.batch_size,
                "requests_per_second": args.requests / duration,
                "mean_ms": statistics.mean(latencies),
                "p50_ms": percentile(0.50),
                "p95_ms": percentile(0.95),
                "p99_ms": percentile(0.99),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
