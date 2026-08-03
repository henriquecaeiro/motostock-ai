# Runtime image for the FastAPI service. Ollama is intentionally external by default.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_BACKEND=sqlite \
    DATABASE_PATH=/app/storage/motostock.db \
    AUTO_IMPORT_CSV=true \
    RAG_KNOWLEDGE_BASE_PATH=/app/knowledge_base \
    RAG_STORAGE_PATH=/app/storage/rag \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime inputs and code. Notebooks, tests and reports stay out of the image.
COPY api ./api
COPY src ./src
COPY scripts ./scripts
COPY artifacts/models ./artifacts/models
COPY data/raw ./data/raw
COPY knowledge_base ./knowledge_base
COPY LICENSE README.md .env.example ./

RUN mkdir -p /app/data/processed /app/storage/rag \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# The preparation/import steps are idempotent. They make a fresh container usable
# from the tracked raw CSV while keeping generated data in the persistent volume.
CMD ["sh", "-c", "python -m scripts.prepare_data && python -m scripts.init_database && python -m scripts.import_csv_data && exec uvicorn api.main:app --host 0.0.0.0 --port 8000"]
