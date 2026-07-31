"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.repositories.csv_repository import CsvRepository
from api.routes import assistant, health, predictions, products, recommendations
from api.services.forecast_service import ForecastService
from api.services.model_service import ModelService
from api.services.ollama_service import OllamaService
from api.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load application dependencies once at startup."""

    repository = CsvRepository()
    model_service = ModelService()
    ollama_service = OllamaService()

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

    app.state.repository = repository
    app.state.model_service = model_service
    app.state.forecast_service = forecast_service
    app.state.recommendation_service = recommendation_service
    app.state.ollama_service = ollama_service

    try:
        yield
    finally:
        await ollama_service.close()


app = FastAPI(
    title="MotoStock AI API",
    description=(
        "Demand forecasting and stock replenishment API for motorcycle and "
        "delivery gear retail stores. This first version serves predictions "
        "and recommendations from CSV-backed data and a saved XGBoost model."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(products.router)
app.include_router(predictions.router)
app.include_router(recommendations.router)
app.include_router(assistant.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "message": "MotoStock AI API",
        "docs": "/docs",
        "health": "/health",
    }
