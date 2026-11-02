# Latência HTTP fim a fim (API em Docker): scikit-learn vs ONNX Runtime

- Medição no cliente (`scripts/load_test.py`), API em container (`docker compose`), 1 worker uvicorn
- 3000 requisições `POST /predict` por cenário, com laudos reais do conjunto de teste
- Inclui rede do Docker Desktop (Windows), validação Pydantic, serialização JSON e métricas Prometheus
- As quatro medições foram feitas em sequência, nas mesmas condições (sem Airflow/MLflow rodando)

| Cenário | Backend | Throughput | Média | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|
| 1 cliente (latência pura) | scikit-learn | 69,2 req/s | 14,29 ms | 13,71 ms | 19,42 ms | 25,38 ms |
| 1 cliente (latência pura) | **ONNX Runtime** | **74,7 req/s** | **13,24 ms** | **12,86 ms** | **17,34 ms** | **21,32 ms** |
| 8 clientes concorrentes | scikit-learn | 162,3 req/s | 46,38 ms | 44,70 ms | 65,75 ms | 83,68 ms |
| 8 clientes concorrentes | **ONNX Runtime** | **204,5 req/s** | **36,26 ms** | **34,78 ms** | **52,53 ms** | **68,40 ms** |

**Leitura:** com 1 cliente, o custo fixo de HTTP e serialização domina (o modelo leva ~1 ms dentro da
API), e o ganho do ONNX fica em ~7%. Sob concorrência, quando o tempo de CPU do modelo vira gargalo, o
ONNX entrega **+26% de throughput** e reduz o p95 em **20%**. O ganho isolado do modelo (sem HTTP) é de
**3,6x** — ver [`latency.md`](latency.md).

**Sobre os valores absolutos:** eles dependem do estado da máquina. Ao longo do desenvolvimento, o
mesmo código mediu de ~6 ms a ~13 ms no p50 com 1 cliente — inclusive com a API rodando fora do Docker
—, conforme a carga de outros processos (antivírus inspecionando conexões, navegador, etc.). Por isso
as comparações são sempre feitas na mesma rodada; a conclusão (ganho grande no modelo, ganho que cresce
com a concorrência na API) se repetiu em todas as rodadas.

Os JSONs brutos estão em `http_latency_<backend>_c<concorrência>.json`.
