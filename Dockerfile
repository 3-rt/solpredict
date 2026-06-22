FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY solpredict ./solpredict
RUN pip install --no-cache-dir ".[api]"

COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY api ./api
COPY models ./models

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
