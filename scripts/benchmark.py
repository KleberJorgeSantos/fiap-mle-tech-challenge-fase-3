"""Offline latency benchmark: original sklearn model vs ONNX Runtime.

Measures only the model call (no HTTP), on real test-set reports:

* single-report latency (N calls, batch of 1) -> mean, p50, p95, p99
* batch latency (batches of 32)               -> p50, p95, throughput
* artifact size, load time and macro-F1 of each backend

Usage:
    python scripts/benchmark.py [--n 1000] [--batch-size 32]

Writes ``reports/latency.json`` and ``reports/latency.md``.
"""

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import (
    LABELS,
    ONNX_MODEL_FILE,
    SKLEARN_MODEL_FILE,
    TARGET_COLUMN,
    TEXT_COLUMN,
    get_settings,
)
from src.logging_config import setup_logging
from src.models.predictor import Predictor, load_predictor

logger = logging.getLogger(__name__)

ARTIFACTS = {"sklearn": SKLEARN_MODEL_FILE, "onnx": ONNX_MODEL_FILE}


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms)
    return {
        "mean_ms": round(float(values.mean()), 4),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "p99_ms": round(float(np.percentile(values, 99)), 4),
    }


def _time_calls(predictor: Predictor, batches: list[list[str]]) -> list[float]:
    samples = []
    for batch in batches:
        start = time.perf_counter()
        predictor.predict_proba(batch)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def benchmark_backend(
    backend: str, model_dir: Path, texts: list[str], labels: list[str], n: int, batch_size: int
) -> dict:
    start = time.perf_counter()
    predictor = load_predictor(backend, model_dir)
    load_ms = (time.perf_counter() - start) * 1000

    rng = np.random.default_rng(42)
    predictor.warmup(50)

    singles = [[texts[i]] for i in rng.integers(0, len(texts), size=n)]
    single = _percentiles(_time_calls(predictor, singles))
    single["throughput_rps"] = round(1000 / single["mean_ms"], 1)

    n_batches = max(n // batch_size, 20)
    batches = [list(rng.choice(texts, size=batch_size)) for _ in range(n_batches)]
    batch = _percentiles(_time_calls(predictor, batches))
    batch["throughput_rps"] = round(batch_size * 1000 / batch["mean_ms"], 1)

    macro_f1 = f1_score(labels, predictor.predict(texts), labels=LABELS, average="macro")
    result = {
        "backend": backend,
        "artifact_mb": round((model_dir / ARTIFACTS[backend]).stat().st_size / 1e6, 2),
        "load_ms": round(load_ms, 1),
        "macro_f1": round(float(macro_f1), 4),
        "single": single,
        f"batch_{batch_size}": batch,
    }
    logger.info("%s: %s", backend, result)
    return result


def to_markdown(results: dict) -> str:
    sk, ox = results["sklearn"], results["onnx"]
    bkey = f"batch_{results['batch_size']}"

    def speedup(metric: str, section: str) -> str:
        return f"{sk[section][metric] / ox[section][metric]:.1f}x"

    rows = [
        ("Latência média (1 laudo)", "single", "mean_ms"),
        ("p50 (1 laudo)", "single", "p50_ms"),
        ("p95 (1 laudo)", "single", "p95_ms"),
        ("p99 (1 laudo)", "single", "p99_ms"),
        (f"p95 (lote de {results['batch_size']})", bkey, "p95_ms"),
    ]
    lines = [
        "# Benchmark de latência: scikit-learn vs ONNX Runtime",
        "",
        f"- Ambiente: {results['environment']}",
        f"- Amostras: {results['n']} inferências unitárias + lotes de {results['batch_size']}"
        " (laudos reais do conjunto de teste, após 50 inferências de warm-up)",
        "- Medição: apenas a chamada ao modelo (sem HTTP)",
        "",
        "| Métrica | scikit-learn (original) | ONNX Runtime (otimizado) | Ganho |",
        "|---|---:|---:|---:|",
    ]
    for title, section, metric in rows:
        lines.append(
            f"| {title} | {sk[section][metric]:.3f} ms | {ox[section][metric]:.3f} ms "
            f"| **{speedup(metric, section)}** |"
        )
    lines += [
        f"| Throughput (1 laudo/chamada) | {sk['single']['throughput_rps']:,.0f} req/s "
        f"| {ox['single']['throughput_rps']:,.0f} req/s "
        f"| **{ox['single']['throughput_rps'] / sk['single']['throughput_rps']:.1f}x** |",
        f"| Throughput (lote) | {sk[bkey]['throughput_rps']:,.0f} laudos/s "
        f"| {ox[bkey]['throughput_rps']:,.0f} laudos/s "
        f"| **{ox[bkey]['throughput_rps'] / sk[bkey]['throughput_rps']:.1f}x** |",
        f"| Tamanho do artefato | {sk['artifact_mb']} MB | {ox['artifact_mb']} MB | "
        f"{sk['artifact_mb'] / ox['artifact_mb']:.1f}x menor |",
        f"| Tempo de carga | {sk['load_ms']:.0f} ms | {ox['load_ms']:.0f} ms | |",
        f"| Macro-F1 (teste) | {sk['macro_f1']:.4f} | {ox['macro_f1']:.4f} | "
        f"{'idêntico' if sk['macro_f1'] == ox['macro_f1'] else 'Δ ' + format(ox['macro_f1'] - sk['macro_f1'], '+.4f')} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    settings = get_settings()
    test = pd.read_csv(settings.data_dir / "processed" / "test.csv")
    texts, labels = test[TEXT_COLUMN].tolist(), test[TARGET_COLUMN].tolist()

    results = {
        "environment": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "n": args.n,
        "batch_size": args.batch_size,
    }
    for backend in ARTIFACTS:
        results[backend] = benchmark_backend(
            backend, settings.model_dir, texts, labels, args.n, args.batch_size
        )

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "latency.json").write_text(json.dumps(results, indent=2))
    markdown = to_markdown(results)
    (settings.reports_dir / "latency.md").write_text(markdown, encoding="utf-8")
    logger.info("Report written to %s\n%s", settings.reports_dir / "latency.md", markdown)


if __name__ == "__main__":
    setup_logging()
    main()
