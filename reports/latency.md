# Benchmark de latência: scikit-learn vs ONNX Runtime

- Ambiente: Windows AMD64, Python 3.11.9
- Amostras: 2000 inferências unitárias + lotes de 32 (laudos reais do conjunto de teste, após 50 inferências de warm-up)
- Medição: apenas a chamada ao modelo (sem HTTP)

| Métrica | scikit-learn (original) | ONNX Runtime (otimizado) | Ganho |
|---|---:|---:|---:|
| Latência média (1 laudo) | 1.365 ms | 0.384 ms | **3.6x** |
| p50 (1 laudo) | 1.324 ms | 0.352 ms | **3.8x** |
| p95 (1 laudo) | 1.779 ms | 0.606 ms | **2.9x** |
| p99 (1 laudo) | 2.213 ms | 0.898 ms | **2.5x** |
| p95 (lote de 32) | 17.540 ms | 11.581 ms | **1.5x** |
| Throughput (1 laudo/chamada) | 732 req/s | 2,604 req/s | **3.6x** |
| Throughput (lote) | 2,661 laudos/s | 3,048 laudos/s | **1.1x** |
| Tamanho do artefato | 3.26 MB | 2.08 MB | 1.6x menor |
| Tempo de carga | 411 ms | 231 ms | |
| Macro-F1 (teste) | 0.7271 | 0.7271 | idêntico |
