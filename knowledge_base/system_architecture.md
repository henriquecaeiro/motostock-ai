# MotoStock AI System Architecture

## Overview

MotoStock AI separates forecasting, stock recommendation, data access, API delivery, knowledge retrieval, and natural-language interaction into distinct responsibilities.

This separation keeps the system easier to understand, test, maintain, and extend.

## Main Components

### XGBoost Forecasting Model

The XGBoost model predicts future product demand.

Its responsibilities are to:

- receive prepared forecasting features;
- generate daily demand predictions;
- support recursive multi-day forecasting;
- return continuous forecast values.

The forecasting model does not decide how much stock should be purchased.

### Recommendation Engine

The recommendation engine converts forecast results into stock replenishment information.

Its responsibilities are to combine:

- forecasted demand;
- safety stock;
- current stock;
- supplier lead time;
- stock classification rules;
- priority rules.

It calculates required stock, recommended purchase quantity, stock status, and priority score.

### FastAPI

FastAPI exposes the system through HTTP endpoints.

Its responsibilities are to:

- validate request data;
- call application services;
- return structured responses;
- map internal failures to safe HTTP responses;
- document the API through OpenAPI and Swagger.

FastAPI does not perform model training and should not contain forecasting or recommendation business logic.

### RAG

Retrieval-Augmented Generation, or RAG, retrieves relevant information from the curated MotoStock AI knowledge base.

Its responsibility is to provide static documented knowledge about:

- the business problem;
- system architecture;
- forecasting behavior;
- model evaluation;
- stock recommendation rules;
- known limitations;
- technical terminology.

RAG must not be used as a source of current inventory, current forecasts, or current recommendations.

### Local LLM

The local language model explains system concepts and interacts with users in natural language.

Its responsibilities are to:

- answer conceptual questions;
- explain retrieved documentation;
- communicate model and business limitations;
- present tool results in a user-friendly format when tools are available.

The LLM must not invent current business data.

## Current Application Architecture

The current application uses CSV-backed data access:

```text
CSV files
    ↓
Repository
    ↓
Services
    ↓
FastAPI
```

### Repository

The repository reads and validates the CSV-backed datasets.

It isolates file access from forecasting and recommendation logic.

### Services

The service layer contains application behavior and coordinates model inference, forecasting, stock recommendation, and Ollama communication.

Examples include:

- ModelService;
- ForecastService;
- RecommendationService;
- OllamaService;
- EmbeddingService;
- VectorStoreService;
- RagService.

### FastAPI Endpoints

The API exposes products, forecasts, stock recommendations, assistant health, and assistant chat functionality.

## Knowledge Interaction Flow

The RAG flow is:

```text
User question
    ↓
EmbeddingService
    ↓
VectorStoreService and cosine search
    ↓
Relevant document chunks and source metadata
    ↓
System prompt, retrieved context and user question
    ↓
Local LLM
    ↓
Grounded explanation
```

## Future SQLite Migration

The current repository reads operational data from CSV files.

A future version may replace CSV-backed persistence with SQLite:

```text
SQLite
    ↓
Repository
    ↓
Services
    ↓
FastAPI
```

The repository abstraction allows the storage implementation to change without moving database logic into the forecasting, recommendation, or API layers.

SQLite is a planned persistence improvement. It is not part of the current CSV-backed architecture unless explicitly implemented.
