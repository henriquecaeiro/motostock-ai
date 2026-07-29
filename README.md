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
python -m pytest -q
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
