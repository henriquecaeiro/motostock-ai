# MotoStock AI

AI-powered demand forecasting and stock replenishment system for motorcycle and delivery gear retail stores.

## Project objective

MotoStock AI helps small retail stores estimate future product demand and decide when to reorder stock. The current phase exposes a FastAPI inference API that serves predictions and replenishment recommendations from processed CSV data and a saved XGBoost model.

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
  repositories/           # CSV data access (replaceable later)
src/
  forecasting.py          # Reusable recursive forecasting logic
  recommendation.py       # Stock recommendation business rules
```

The route layer stays thin. Services orchestrate forecasting and recommendations. The CSV repository can later be replaced by a SQLite repository without rewriting the API routes.

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the API

From the repository root:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Running tests

```bash
python -m pytest tests/test_assistant.py -q
python -m pytest -q
```

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
  "sources": []
}
```

### Current limitations

The current assistant is connected to the local LLM, but RAG and live-data tools have not been implemented yet.

In the current version:

- `tools_used` remains empty
- `sources` remains empty
- the assistant can answer conceptual questions about forecasting, inventory, and recommendations
- it does not query current stock levels
- it does not query current forecasts
- it does not query current recommendations
- it must not invent current business values

Automated assistant tests use mocks and do not require Ollama to be running. For a real integration check, use:

```bash
python scripts/check_ollama_integration.py
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
| GET | `/recommendations/summary` | Summary counts from the current recommendations |
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

The API currently reads from CSV files:

- `data/processed/modeling_dataset.csv`
- `data/processed/daily_product_sales.csv`

SQLite integration, sales ingestion endpoints, and model retraining will be implemented in a later phase.
