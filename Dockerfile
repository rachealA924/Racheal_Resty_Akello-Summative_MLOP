# Cassava leaf disease API + web UI
# Works on Hugging Face Spaces (Docker SDK, port 7860) and locally with docker compose.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000, so the app folder must belong to it
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY api ./api
COPY ui ./ui
COPY models ./models
COPY data/train ./data/train
COPY data/test ./data/test
RUN mkdir -p data/uploads && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:${PORT}/health || exit 1

# One worker per container: scaling is done by running more containers
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
