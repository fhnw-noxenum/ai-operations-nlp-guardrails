FROM python:3.11-slim

WORKDIR /app

# System deps for some NeMo Guardrails extras (e.g. embeddings)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/ /app/backend/
COPY configs/ /app/configs/

ENV PYTHONUNBUFFERED=1
ENV CONFIGS_DIR=/app/configs

EXPOSE 8000

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
