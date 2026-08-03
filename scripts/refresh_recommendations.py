"""Run one operational refresh without starting an HTTP server."""

from __future__ import annotations

import json
import sys

from api.config import load_settings
from api.repositories.factory import create_repository
from api.services.forecast_service import ForecastService
from api.services.model_service import ModelService
from api.services.recommendation_service import RecommendationService
from api.services.refresh_service import RefreshService


def main() -> int:
    settings = load_settings()
    repository = create_repository(settings)
    model_service = ModelService()
    model_service.load()
    forecast_service = ForecastService(repository, model_service)
    recommendation_service = RecommendationService(
        repository,
        forecast_service,
        model_service,
    )
    refresh_service = RefreshService(
        repository,
        recommendation_service,
        model_service,
    )
    result = refresh_service.refresh(horizon_days=14)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
