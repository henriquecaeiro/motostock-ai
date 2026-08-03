# MotoStock AI System Architecture

## Overview

MotoStock AI separates forecasting, stock recommendation, operational data
access, API delivery, knowledge retrieval and natural-language interaction.
The current default backend is SQLite, while CSV remains the reproducible
source used to prepare and import the demo data.

The responsibilities are explicit:

- XGBoost predicts daily product demand.
- The recommendation engine calculates replenishment quantities and stock
  status.
- SQLite stores current operational data and persisted results.
- RAG provides documented project knowledge.
- Read-only tools provide current business values from application services.
- The local LLM explains documented knowledge and presents verified tool output.
- FastAPI validates requests and exposes the workflow over HTTP.

## Main application flow

```text
Raw CSV
    ↓
Deterministic preparation
    ↓
SQLite repository
    ↓
Forecast and recommendation services
    ↓
FastAPI endpoints
```

The API can also use a read-only CSV repository when `DATA_BACKEND=csv` is
selected. The repository contract keeps storage details out of forecasting and
recommendation code.

## Forecast and recommendation components

### XGBoost forecasting model

The XGBoost pipeline receives calendar, lag, rolling and product features and
returns continuous daily demand predictions. Serving uses recursive forecasting
for the requested horizon. The model does not decide how much stock to buy.

### Recommendation engine

The recommendation engine clips negative demand to zero, aggregates the
horizon, applies 20% safety stock, compares required stock with current stock
and calculates status, purchase quantity, lead-time context and priority.

### SQLite operational layer

SQLite stores products, suppliers, sales, inventory snapshots, modeling rows,
recommendation runs, stock recommendations, model versions and application
runs. Sales writes are validated, transactional and idempotent. Refresh
rebuilds features and persists a completed run under a deterministic key.

## RAG flow

```text
User question
    ↓
EmbeddingService
    ↓
Local NumPy vector search
    ↓
Relevant Markdown chunks and source metadata
    ↓
Bounded, explicitly untrusted context
    ↓
Local Ollama chat model
    ↓
Grounded explanation
```

The curated knowledge base covers the business problem, architecture,
forecasting model, evaluation, recommendation rules, limitations and glossary.
RAG is not a source of current inventory, forecasts or recommendation values.

## Assistant tools

The assistant uses deterministic intent rules and four allowlisted read-only
tools:

- `list_products`;
- `forecast_product`;
- `get_recommendations`;
- `get_recommendation_summary`.

Tool arguments use strict Pydantic validation. Tool output is serialized from
application services and returned with `tools_used`; the LLM does not invent
current business values or execute arbitrary code.

## Model operations

Training is separate from serving. A candidate is trained on a temporal split,
evaluated against the production artifact, checked by global/product gates and
only promoted by an explicit administrative command. Candidate artifacts never
silently replace the production model. Rollback uses the recorded parent
version and a backup.

## API layers

FastAPI validates request and response schemas, maps internal failures to safe
HTTP details and exposes OpenAPI/Swagger. Routes call services; services call
repositories and model/LLM adapters. This keeps database, forecast, stock-rule
and assistant behavior testable without putting business logic in route
functions.
