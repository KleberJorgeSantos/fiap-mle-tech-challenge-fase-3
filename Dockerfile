# ============================================================
# Medical Report Triage API -- inference image
#
#   make train                                  # generates models/
#   docker build -t triage-api .
#   docker run -p 8000:8000 triage-api          # ONNX backend (default)
#   docker run -p 8000:8000 -e MODEL_BACKEND=sklearn triage-api
# ============================================================

# ── Stage 1: resolve dependencies with uv (serving deps only) ──
FROM python:3.11-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: slim runtime ──
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    MODEL_BACKEND=onnx

# Non-root user. The app only READS its files, so they stay owned by root: no
# `chown -R` (which duplicated the whole .venv in an extra layer) and the process
# cannot modify its own code or model.
RUN useradd --create-home --uid 1001 appuser

COPY --from=builder /app/.venv /app/.venv
COPY params.yaml ./
COPY src/ ./src/
COPY models/ ./models/

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
