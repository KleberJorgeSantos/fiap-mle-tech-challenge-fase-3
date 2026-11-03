.PHONY: help install lint format test train benchmark api up down logs load-test traffic \
        mlops-up mlops-down infra-up infra-outputs infra-down clean

UV ?= uv run
MLOPS_COMPOSE = docker compose -f orchestration/docker-compose.yml

# Training logs to the MLflow server of the MLOps platform (skipped if it is not running).
export MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000

# AWS stack (infra/cloudformation.yml)
STACK        ?= triage-api
AWS_REGION   ?= us-east-1
GITHUB_REPO  ?=
BUDGET_EMAIL ?=
ELASTIC_IP   ?= false   # true: fixed public IP (evaluation period)
SCHEDULE     ?= true    # false: instance on 24/7

help:           ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Ambiente e qualidade ─────────────────────────────────────────────────────

install:        ## Instala todas as dependências (serving + train + dev)
	uv sync --all-extras

lint:           ## Ruff (lint + formatação) e cfn-lint (CloudFormation)
	$(UV) ruff check src tests scripts dags
	$(UV) ruff format --check src tests scripts dags
	$(UV) cfn-lint infra/cloudformation.yml

format:         ## Formata o código com Ruff
	$(UV) ruff check --fix src tests scripts dags
	$(UV) ruff format src tests scripts dags

test:           ## Testes com cobertura (offline, dataset sintético)
	$(UV) pytest

# ── Modelo ───────────────────────────────────────────────────────────────────

models/model.onnx:
	$(UV) python -m src.pipelines.train

train:          ## Pipeline completo: dados → treino → gate → ONNX → publish → MLflow → S3
	$(UV) python -m src.pipelines.train

benchmark: models/model.onnx  ## Latência offline sklearn vs ONNX → reports/latency.md
	$(UV) python -m scripts.benchmark

# ── Serving: API + monitoramento (local) ─────────────────────────────────────

api: models/model.onnx  ## API local com reload em http://localhost:8000/docs
	$(UV) uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

up: models/model.onnx   ## Sobe nginx + API + Prometheus + Grafana (http://localhost)
	docker compose up -d --build

down:           ## Derruba a stack de serving
	docker compose down

logs:           ## Logs da API
	docker compose logs -f api

load-test:      ## Mede latência HTTP fim a fim (3000 req, 8 clientes)
	$(UV) python -m scripts.load_test --n 3000 --concurrency 8

traffic:        ## Tráfego contínuo por 5 min com 3% de erros (alimenta o Grafana)
	$(UV) python -m scripts.load_test --duration 300 --concurrency 4 --error-ratio 0.03

# ── Plataforma de MLOps: Airflow + MLflow (local) ────────────────────────────

mlops-up:       ## Sobe Airflow (:8080, airflow/airflow) + MLflow (:5000)
	@test -f orchestration/.env || cp orchestration/.env.example orchestration/.env
	$(MLOPS_COMPOSE) up -d --build

mlops-down:     ## Derruba Airflow + MLflow
	$(MLOPS_COMPOSE) down

# ── Infraestrutura AWS (CloudFormation) ──────────────────────────────────────

infra-up:       ## Cria/atualiza o stack AWS (GITHUB_REPO=dono/repo [BUDGET_EMAIL=] [ELASTIC_IP=true] [SCHEDULE=false])
	@test -n "$(GITHUB_REPO)" || (echo "Uso: make infra-up GITHUB_REPO=dono/repo [BUDGET_EMAIL=voce@email]" && exit 1)
	aws cloudformation deploy --stack-name $(STACK) --region $(AWS_REGION) \
		--template-file infra/cloudformation.yml --capabilities CAPABILITY_IAM \
		--parameter-overrides GitHubRepository=$(GITHUB_REPO) BudgetEmail=$(BUDGET_EMAIL) \
		UseElasticIp=$(ELASTIC_IP) EnableSchedule=$(SCHEDULE)
	@"$(MAKE)" --no-print-directory infra-outputs

infra-outputs:  ## Mostra os valores para as variáveis do GitHub e do orchestration/.env
	aws cloudformation describe-stacks --stack-name $(STACK) --region $(AWS_REGION) \
		--query "Stacks[0].Outputs[].[Description,OutputValue]" --output table

infra-down:     ## Apaga TODO o stack AWS (esvazia o bucket antes) — custo volta a zero
	$(eval BUCKET := $(shell aws cloudformation describe-stacks --stack-name $(STACK) --region $(AWS_REGION) \
		--query "Stacks[0].Outputs[?OutputKey=='ModelsBucketName'].OutputValue" --output text))
	$(UV) python -m scripts.empty_bucket $(BUCKET)
	aws cloudformation delete-stack --stack-name $(STACK) --region $(AWS_REGION)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK) --region $(AWS_REGION)
	@echo "Stack $(STACK) removido."

clean:          ## Remove caches e artefatos gerados
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml data/processed data/staging
