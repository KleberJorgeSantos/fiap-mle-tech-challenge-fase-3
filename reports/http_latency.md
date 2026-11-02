# Latência HTTP fim a fim (API em Docker): scikit-learn vs ONNX Runtime

- Medição no cliente (`scripts/load_test.py`), API em container (`docker compose`), 1 worker uvicorn
- 3000 requisições `POST /predict` com laudos reais do conjunto de teste
- Inclui rede Docker Desktop (Windows), validação Pydantic, serialização JSON e métricas Prometheus

| Cenário | Backend | Throughput | Média | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|
| 1 cliente (latência pura) | scikit-learn | 136.9 req/s | 7.14 ms | 6.86 ms | 9.87 ms | 12.80 ms |
| 1 cliente (latência pura) | **ONNX Runtime** | **164.1 req/s** | **5.92 ms** | **5.68 ms** | **7.97 ms** | **10.90 ms** |
| 8 clientes concorrentes | scikit-learn | 199.6 req/s | 36.60 ms | 35.16 ms | 49.59 ms | 69.10 ms |
| 8 clientes concorrentes | **ONNX Runtime** | **256.6 req/s** | **27.46 ms** | **26.25 ms** | **41.37 ms** | **54.59 ms** |

**Leitura:** com 1 cliente o custo fixo de HTTP/serialização (~5 ms) domina e o ganho do ONNX é de ~17%
na média. Sob concorrência, onde o tempo de CPU do modelo vira gargalo, o ONNX entrega **+29% de
throughput** e reduz o p95 em ~17%. O ganho isolado do modelo (sem HTTP) é de **5,3x** — ver
[`latency.md`](latency.md).

Os JSONs brutos estão em `http_latency_<backend>_c<concorrência>.json`.
