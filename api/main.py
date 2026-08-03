"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.config import load_settings
from api.repositories.factory import create_repository
from api.routes import (
    assistant,
    health,
    inventory,
    models,
    predictions,
    products,
    recommendations,
    sales,
)
from api.services.embedding_service import EmbeddingService
from api.services.forecast_service import ForecastService
from api.services.model_service import ModelService
from api.services.rag_service import RagService
from api.services.ollama_service import OllamaService
from api.services.recommendation_service import RecommendationService
from api.services.refresh_service import RefreshService
from api.services.tool_service import ToolService
from api.services.vector_store_service import (
    VectorStoreCorruptedError,
    VectorStoreNotFoundError,
    VectorStoreService,
)

logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load application dependencies once at startup."""

    settings = load_settings()
    repository = create_repository(settings)
    model_service = ModelService()
    ollama_service = OllamaService(settings=settings)

    try:
        model_service.load()
        repository.load_daily_sales()
        repository.load_modeling_data()
    except Exception:
        logger.exception("Startup initialization encountered an error")

    forecast_service = ForecastService(
        repository=repository,
        model_service=model_service,
    )

    recommendation_service = RecommendationService(
        repository=repository,
        forecast_service=forecast_service,
        model_service=model_service,
    )

    vector_store_service = VectorStoreService(settings=settings)

    try:
        vector_store_service.load_index()
    except VectorStoreNotFoundError:
        logger.warning("RAG index is unavailable; assistant retrieval is disabled.")
    except VectorStoreCorruptedError:
        logger.error("RAG index is invalid; assistant retrieval is disabled.")

    embedding_service = EmbeddingService(
        ollama_service,
        settings=settings,
    )
    rag_service = RagService(
        embedding_service,
        vector_store_service,
        settings=settings,
    )
    tool_service = ToolService(
        repository=repository,
        forecast_service=forecast_service,
        recommendation_service=recommendation_service,
    )
    refresh_service = RefreshService(
        repository=repository,
        recommendation_service=recommendation_service,
        model_service=model_service,
    )

    app.state.settings = settings
    app.state.repository = repository
    app.state.model_service = model_service
    app.state.forecast_service = forecast_service
    app.state.recommendation_service = recommendation_service
    app.state.ollama_service = ollama_service
    app.state.embedding_service = embedding_service
    app.state.vector_store_service = vector_store_service
    app.state.rag_service = rag_service
    app.state.tool_service = tool_service
    app.state.refresh_service = refresh_service

    try:
        yield
    finally:
        await ollama_service.close()


app = FastAPI(
    title="MotoStock AI API",
    description=(
        "Demand forecasting and stock replenishment API for motorcycle and "
        "delivery gear retail stores. The API supports a SQLite operational "
        "backend with CSV import compatibility and a saved XGBoost model."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(products.router)
app.include_router(predictions.router)
app.include_router(recommendations.router)
app.include_router(sales.router)
app.include_router(inventory.router)
app.include_router(models.router)
app.include_router(assistant.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "message": "MotoStock AI API",
        "docs": "/docs",
        "health": "/health",
    }
