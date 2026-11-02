"""HTTP load generator for the running API (end-to-end latency + Grafana traffic).

Sends real test-set reports to ``POST /predict`` from several threads and reports
client-side latency percentiles. A small share of invalid payloads can be mixed
in so the error-rate panels in Grafana have something to show.

Usage:
    python -m scripts.load_test --n 2000 --concurrency 8
    python -m scripts.load_test --duration 300 --error-ratio 0.03   # continuous traffic

Writes ``reports/http_latency_<backend>_c<concurrency>.json``.
"""

import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from src.config import get_settings
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

_FALLBACK_TEXTS = [
    "Acute myocardial infarction with ST elevation and severe chest pain.",
    "Ischemic stroke with left hemiparesis and aphasia, onset two hours ago.",
    "Colon adenocarcinoma staging after resection, no distant metastasis.",
    "Chronic gastritis with mild mucosal inflammation on endoscopy.",
    "Wound healing assessment in soft tissue after minor surgery.",
]
_INVALID_PAYLOADS = [{"text": ""}, {"text": "!!!"}, {"wrong_field": "x"}]


def load_texts(limit: int = 2000) -> list[str]:
    """Real test-set reports when available, otherwise a few built-in examples."""
    path = get_settings().data_dir / "processed" / "test.csv"
    if not path.exists():
        return _FALLBACK_TEXTS
    import pandas as pd

    return pd.read_csv(path)["text"].head(limit).tolist()


def run(url: str, n: int, duration: float, concurrency: int, error_ratio: float) -> dict:
    texts = load_texts()
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    lock = threading.Lock()
    deadline = time.monotonic() + duration if duration else None
    counter = iter(range(10**9))

    def worker() -> None:
        with httpx.Client(base_url=url, timeout=10) as client:
            while True:
                i = next(counter)
                if deadline is None and i >= n:
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    return
                invalid = random.random() < error_ratio
                payload = (
                    random.choice(_INVALID_PAYLOADS) if invalid else {"text": random.choice(texts)}
                )
                start = time.perf_counter()
                response = client.post("/predict", json=payload)
                elapsed = (time.perf_counter() - start) * 1000
                with lock:
                    statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
                    if response.status_code == 200:
                        latencies.append(elapsed)

    backend = httpx.get(f"{url}/health", timeout=10).json().get("backend", "unknown")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(worker)
    wall = time.perf_counter() - started

    values = np.asarray(latencies)
    result = {
        "backend": backend,
        "requests": int(sum(statuses.values())),
        "status_codes": {str(k): v for k, v in sorted(statuses.items())},
        "concurrency": concurrency,
        "throughput_rps": round(sum(statuses.values()) / wall, 1),
        "mean_ms": round(float(values.mean()), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "p99_ms": round(float(np.percentile(values, 99)), 3),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first (+~40 ms per call).
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--n", type=int, default=2000, help="total requests (ignored with --duration)"
    )
    parser.add_argument("--duration", type=float, default=0, help="seconds of continuous traffic")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--error-ratio", type=float, default=0.0)
    args = parser.parse_args()

    result = run(args.url, args.n, args.duration, args.concurrency, args.error_ratio)
    reports_dir = get_settings().reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / f"http_latency_{result['backend']}_c{args.concurrency}.json"
    target.write_text(json.dumps(result, indent=2))
    logger.info("HTTP load test: %s (saved to %s)", result, target)


if __name__ == "__main__":
    setup_logging()
    main()
