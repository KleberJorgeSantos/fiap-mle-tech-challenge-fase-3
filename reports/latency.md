# Benchmark de latência: scikit-learn vs ONNX Runtime

- Ambiente: Windows AMD64, Python 3.11.9
- Amostras: 1000 inferências unitárias + lotes de 32 (laudos reais do conjunto de teste, após 50 inferências de warm-up)
- Medição: apenas a chamada ao modelo (sem HTTP)

| Métrica | scikit-learn (original) | ONNX Runtime (otimizado) | Ganho |
|---|---:|---:|---:|
| Latência média (1 laudo) | 2.096 ms | 0.392 ms | **5.3x** |
| p50 (1 laudo) | 1.893 ms | 0.365 ms | **5.2x** |
| p95 (1 laudo) | 3.419 ms | 0.606 ms | **5.6x** |
| p99 (1 laudo) | 3.994 ms | 0.784 ms | **5.1x** |
| p95 (lote de 32) | 15.196 ms | 13.259 ms | **1.1x** |
| Throughput (1 laudo/chamada) | 477 req/s | 2,551 req/s | **5.3x** |
| Throughput (lote) | 2,641 laudos/s | 2,809 laudos/s | **1.1x** |
| Tamanho do artefato | 3.26 MB | 2.08 MB | 1.6x menor |
| Tempo de carga | 486 ms | 253 ms | |
| Macro-F1 (teste) | 0.7271 | 0.7271 | idêntico |
