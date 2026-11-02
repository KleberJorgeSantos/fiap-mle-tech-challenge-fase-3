.PHONY: help install lint format test data train benchmark api up down logs load-test traffic \
        airflow-up airflow-down clean

UV ?= uv run
AIRFLOW_COMPOSE = docker compose -f orchestration/docker-compose.airflow.yml

# Training logs to the MLflow server of the compose stack (skipped if it is not running).
export MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000

help:           ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Ambiente e qualidade ─────────────────────────────────────────────────────

install:        ## Instala todas as dependências (serving + train + dev)
	uv sync --all-extras

lint:           ## Ruff: lint + checagem de formatação
	$(UV) ruff check src tests scripts dags
	$(UV) ruff format --check src tests scripts dags

format:         ## Formata o código com Ruff
	$(UV) ruff check --fix src tests scripts dags
	$(UV) ruff format src tests scripts dags

test:           ## Testes com cobertura (offline, dataset sintético)
	$(UV) pytest

# ── Modelo ───────────────────────────────────────────────────────────────────

models/model.onnx:
	$(UV) python -m src.pipelines.train

train:          ## Pipeline completo: download → validação → treino → gate → ONNX → publish → MLflow
	$(UV) python -m src.pipelines.train

benchmark: models/model.onnx  ## Latência offline sklearn vs ONNX → reports/latency.md
	$(UV) python -m scripts.benchmark

# ── API e stack de monitoramento ─────────────────────────────────────────────

api: models/model.onnx  ## API local com reload em http://localhost:8000/docs
	$(UV) uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

up: models/model.onnx   ## Sobe nginx + API + Prometheus + Grafana + MLflow (http://localhost)
	docker compose up -d --build

down:           ## Derruba a stack
	docker compose down

logs:           ## Logs da API
	docker compose logs -f api

load-test:      ## Mede latência HTTP fim a fim (3000 req, 8 clientes)
	$(UV) python -m scripts.load_test --n 3000 --concurrency 8

traffic:        ## Tráfego contínuo por 5 min com 3% de erros (alimenta o Grafana)
	$(UV) python -m scripts.load_test --duration 300 --concurrency 4 --error-ratio 0.03

# ── Airflow ──────────────────────────────────────────────────────────────────

airflow-up:     ## Sobe o Airflow local em http://localhost:8080 (airflow/airflow)
	@test -f orchestration/.env || cp orchestration/.env.example orchestration/.env
	$(AIRFLOW_COMPOSE) up -d --build

airflow-down:   ## Derruba o Airflow
	$(AIRFLOW_COMPOSE) down

clean:          ## Remove caches e artefatos gerados
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml data/processed data/staging
