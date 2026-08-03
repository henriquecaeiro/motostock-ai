# MotoStock AI

AI-powered demand forecasting and stock replenishment system for motorcycle and delivery gear retail stores.

## Project objective

MotoStock AI helps small retail stores estimate future product demand and decide when to reorder stock. The FastAPI API serves predictions and replenishment recommendations from a saved XGBoost model, with SQLite as the operational backend and CSV retained for repeatable imports and development.

## Current ML model

- **Production model:** XGBoost pipeline saved at `artifacts/models/xgboost_model.pkl`
- **Input features:** unit price, calendar features, lag/rolling demand features, and product name
- **Forecast horizon:** 1 to 30 days (default: 14)
- **Business rule:** daily negative predictions are set to zero; product totals use `ceil(sum of non-negative daily predictions)`

## API architecture

```text
api/
  main.py                 # FastAPI app and startup lifecycle
  routes/                 # HTTP endpoints
  schemas/                # Pydantic request/response models
  services/               # Forecasting and recommendation logic
  repositories/           # CSV and SQLite data access behind one contract
src/
  forecasting.py          # Reusable recursive forecasting logic
  recommendation.py       # Stock recommendation business rules
```

The route layer stays thin. Services orchestrate forecasting and recommendations. `DataRepository` keeps the API independent from whether data came from CSV or SQLite.

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements-dev.txt
```

`requirements.txt` contains the runtime dependencies. The development file
adds pytest, notebooks and visualisation packages. The version ranges keep the
FastAPI/Pydantic v2 and scientific Python stack compatible without pretending
to be a full lock file.

## Running the API

From the repository root:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

For the operational SQLite mode, configure the backend before starting the API:

```env
DATA_BACKEND=sqlite
DATABASE_PATH=storage/motostock.db
AUTO_IMPORT_CSV=true
```

When the configured SQLite file is empty and the processed CSV files exist, the
application imports them once at startup. The database file is ignored by Git.

### SQLite data layer

The database schema is versioned and includes products, suppliers, sales,
inventory snapshots, modeling rows, recommendation runs, persisted stock
recommendations, and model metadata. Foreign keys, check constraints and
indexes are enabled at connection time.

Initialize and import explicitly when a controlled data step is preferred:

```bash
# Prepare ignored serving CSVs from the tracked raw dataset
.venv\\Scripts\\python.exe -m scripts.prepare_data

# Windows
.venv\\Scripts\\python.exe -m scripts.init_database
.venv\\Scripts\\python.exe -m scripts.import_csv_data

# Linux/macOS
python -m scripts.prepare_data
python -m scripts.init_database
python -m scripts.import_csv_data
```

The import is transactional and idempotent. Historical zero-demand days are
preserved; new operational sales still require a positive quantity and a
non-negative price. Set `DATA_BACKEND=csv` to use the read-only CSV repository.

Run the same refresh flow from a shell when an HTTP call is not needed:

```bash
.venv\\Scripts\\python.exe -m scripts.refresh_recommendations
```

Model training is deliberately separate from serving. Train a candidate,
evaluate it, then promote or roll it back explicitly:

```bash
.venv\\Scripts\\python.exe -m scripts.train_model --model xgboost
.venv\\Scripts\\python.exe -m scripts.evaluate_candidate --artifact artifacts/models/candidates/<version>.pkl
.venv\\Scripts\\python.exe -m scripts.promote_model <version>
.venv\\Scripts\\python.exe -m scripts.rollback_model <production-version>
```

Candidates are stored outside the production artifact path. The registry keeps
status, checksum, parameters, metrics, data interval and parent version.

### Assistant evaluation

The versioned evaluation cases live under `evaluation/`. Run the deterministic
checks without Ollama:

```bash
.venv\\Scripts\\python.exe -m scripts.evaluate_assistant
```

To measure the real local assistant, including retrieval latency and source
provenance, opt in explicitly:

```bash
.venv\\Scripts\\python.exe -m scripts.evaluate_assistant --live
```

Both modes write Markdown, CSV and JSON reports under `evaluation/results/`,
which is intentionally ignored. Deterministic checks cover tool choice,
arguments, invalid products/horizons, source catalog presence, missing index
and prompt-injection handling. Live LLM answer quality and numerical
fidelity remain manual review items rather than a fabricated accuracy score.

## Running tests

```bash
# Fast unit-oriented checks
python -m pytest -m "not integration and not e2e and not manual" -q

# SQLite/API integration checks
python -m pytest -m integration -q

# Complete workflow with a temporary SQLite database
python -m pytest -m e2e -q

# Full local suite; the real Ollama check remains skipped by default
python -m pytest -q
```

The opt-in real Ollama check is kept outside the quick suite:

```powershell
$env:RUN_OLLAMA_TESTS="1"
python -m pytest -m manual -q
python -m scripts.check_ollama_integration
```

On Linux/macOS, use `RUN_OLLAMA_TESTS=1 python -m pytest -m manual -q`.

## Docker

The API image uses Python 3.11, installs only runtime dependencies, prepares
the ignored serving CSVs from the tracked raw dataset, imports SQLite data and
then starts Uvicorn. SQLite and the generated RAG index are stored in the
`motostock_storage` named volume.

```bash
docker compose build api
docker compose up api
curl http://127.0.0.1:8000/health
```

By default the API container reaches Ollama on the host at
`http://host.docker.internal:11434`. Start Ollama on the host and set
`OLLAMA_MODEL` and `OLLAMA_EMBEDDING_MODEL` as needed. The Docker healthcheck
only checks the FastAPI process; `/assistant/health` reports Ollama/model
availability separately.

If your local `.env` still sets `OLLAMA_BASE_URL=http://localhost:11434`,
override it for the container with
`OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose up api`.

Ollama can also run in an optional container. It requires no GPU configuration
in this compose file, but model downloads are still explicit:

```powershell
$env:OLLAMA_BASE_URL="http://ollama:11434"
docker compose --profile ollama up --build
docker compose exec ollama ollama pull qwen3:4b
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec api python -m scripts.index_knowledge_base
```

On Linux/macOS, use `OLLAMA_BASE_URL=http://ollama:11434 docker compose --profile ollama up --build`.
The optional Ollama model volume is separate from the API storage volume.

## Local AI Assistant

The MotoStock AI API includes a local AI assistant powered by [Ollama](https://ollama.com/). The assistant uses the **qwen3:4b** model, which runs locally on your machine through the Ollama runtime.

### Install the model

```bash
ollama pull qwen3:4b
```

Confirm the model is available:

```bash
ollama list
```

### Start Ollama

```bash
ollama serve
```

On some Windows installations, the Ollama desktop app may start the service automatically. Do not run `ollama serve` if another Ollama instance is already listening on the same port.

### Environment variables

Configure the assistant in your local `.env` file (see `.env.example`):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT_SECONDS=60
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
RAG_KNOWLEDGE_BASE_PATH=knowledge_base
RAG_STORAGE_PATH=storage/rag
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_DEFAULT_TOP_K=4
RAG_MAX_TOP_K=10
RAG_MIN_SCORE=0.40
RAG_MAX_CONTEXT_CHARS=6000
```

Do not commit your real `.env` file.

### Run the API

```bash
uvicorn api.main:app --reload
```

Interactive documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`

### Assistant health endpoint

```bash
curl http://127.0.0.1:8000/assistant/health
```

Example success response:

```json
{
  "status": "ok",
  "ollama_available": true,
  "model": "qwen3:4b",
  "model_available": true
}
```

Possible `status` values:

- `ok` — Ollama is reachable and the configured model is installed
- `ollama_unavailable` — the Ollama server could not be reached
- `model_unavailable` — Ollama is reachable, but the configured model is missing

### Assistant chat endpoint

```bash
curl -X POST http://127.0.0.1:8000/assistant/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"What is MotoStock AI?\"}"
```

Example response:

```json
{
  "answer": "MotoStock AI...",
  "model": "qwen3:4b",
  "tools_used": [],
  "sources": [
    {
      "source": "model_evaluation.md",
      "section": "Model Selection",
      "chunk_id": "model-evaluation-model-selection-001",
      "score": 0.5889
    }
  ]
}
```

### Current limitations

The current assistant is connected to the local LLM, can retrieve static project documentation through a local NumPy vector index, and can call four read-only business tools.

In the current version:

- `tools_used` lists the allowlisted tool used for current-data questions
- `sources` contains retrieved document metadata when the RAG index is available
- the assistant can answer grounded conceptual questions about forecasting, inventory, and recommendations
- current values are returned directly from the internal application services
- it must not invent current business values

Automated assistant tests use mocks and do not require Ollama to be running. For a real integration check, run this command from the project root:

```bash
python -m scripts.check_ollama_integration
```

On Windows, you can also use:

```bash
.venv\Scripts\python.exe -m scripts.check_ollama_integration
```

The script configures `stdout` and `stderr` as UTF-8 when supported by Python, which helps avoid encoding errors on Windows terminals.

### RAG knowledge base and embeddings

The curated knowledge base lives in `knowledge_base/` and is loaded by `KnowledgeBaseService`, which prepares deterministic `KnowledgeChunk` objects using the chunker in `api/rag/chunking.py`. This step does not generate embeddings or perform vector search.

Install the embedding model separately from the chat model:

```bash
ollama pull qwen3-embedding:0.6b
```

| Setting | Purpose | Example |
|---------|---------|---------|
| `OLLAMA_MODEL` | Chat model | `qwen3:4b` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `qwen3-embedding:0.6b` |
| `RAG_CHUNK_SIZE` | Chunk size in characters | `1000` |
| `RAG_CHUNK_OVERLAP` | Chunk overlap in characters | `150` |
| `RAG_DEFAULT_TOP_K` | Default number of retrieved chunks | `4` |
| `RAG_MAX_TOP_K` | Maximum number of retrieved chunks | `10` |
| `RAG_MIN_SCORE` | Minimum cosine score used by retrieval | `0.40` |
| `RAG_MAX_CONTEXT_CHARS` | Maximum context sent to the chat model | `6000` |

`EmbeddingService` sends embeddings in batches through the shared `OllamaService`, normalizes the vectors, and validates their dimensions. `VectorStoreService` persists `embeddings.npz`, `metadata.json`, and `index_info.json` under `storage/rag/`; these generated files are ignored by Git.

Build the index after installing the embedding model:

```bash
# Windows
.venv\Scripts\python.exe -m scripts.index_knowledge_base

# Linux/macOS
python -m scripts.index_knowledge_base
```

The indexing command is idempotent and replaces the previous index. Check representative English, Portuguese, and out-of-domain queries with:

```bash
python -m scripts.check_rag_retrieval
```

### Read-only assistant tools

The assistant uses deterministic intent rules and an explicit allowlist instead of allowing the language model to execute arbitrary tool names or code. The available tools are:

- `list_products`
- `forecast_product`
- `get_recommendations`
- `get_recommendation_summary`

Tool arguments are validated with Pydantic, horizons are limited to 1–30 days, product names are checked by the repository, and tool results are rendered as exact JSON returned by the application services. Tool requests do not make HTTP calls back into this API.

Run the manual tool check with:

```bash
# Windows
.venv\Scripts\python.exe -m scripts.check_assistant_tools

# Linux/macOS
python -m scripts.check_assistant_tools
```

## Swagger documentation

After starting the API, open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Endpoint overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API and model availability check |
| GET | `/products` | List known products |
| POST | `/predict` | Forecast demand for one product |
| GET | `/recommendations` | Generate stock recommendations for all products |
| GET | `/recommendations/latest` | Read the latest persisted recommendation run |
| GET | `/recommendations/summary` | Summary counts from the current recommendations |
| POST | `/recommendations/refresh` | Rebuild features and persist current recommendations |
| POST | `/sales` | Insert one idempotent operational sale |
| POST | `/sales/batch` | Insert a transactional batch of sales |
| GET | `/models` | List registered model candidates and production versions |
| POST | `/models/{version}/promote` | Explicitly promote a candidate after the gate |
| POST | `/models/{version}/rollback` | Restore the selected production model's parent |
| GET | `/assistant/health` | Check Ollama and configured model availability |
| POST | `/assistant/chat` | Send a message to the local AI assistant |

## Example requests

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### List products

```bash
curl http://127.0.0.1:8000/products
```

### Predict demand

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"product_name\": \"Bag Delivery 45L\", \"horizon_days\": 14}"
```

### Get recommendations

```bash
curl "http://127.0.0.1:8000/recommendations?horizon_days=14&stock_status=critical"
```

### Get recommendation summary

```bash
curl http://127.0.0.1:8000/recommendations/summary
```

## Current data source

The default example configuration uses SQLite at `storage/motostock.db`. The
processed CSV files remain the reproducible import source:

- `data/processed/modeling_dataset.csv`
- `data/processed/daily_product_sales.csv`

The current branch includes the database foundation and persisted
recommendation runs. `POST /sales` and `POST /sales/batch` accept idempotent
operational writes, and `POST /recommendations/refresh` rebuilds the feature
table and recommendation result under a simple in-process lock.
