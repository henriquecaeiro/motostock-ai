"""Incremental, idempotent operational refresh orchestration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import HTTPException, status

from api.exceptions import ServiceUnavailableError
from api.repositories.protocol import DataRepository
from api.services.model_service import ModelService
from api.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)


class RefreshService:
    """Run one refresh at a time inside the current API process."""

    def __init__(
        self,
        repository: DataRepository,
        recommendation_service: RecommendationService,
        model_service: ModelService,
    ) -> None:
        self.repository = repository
        self.recommendation_service = recommendation_service
        self.model_service = model_service
        self._lock = Lock()

    def refresh(self, *, horizon_days: int) -> dict[str, Any]:
        """Rebuild features and persist a current recommendation result."""

        if not hasattr(self.repository, "database_path"):
            raise ServiceUnavailableError(
                "Operational refresh requires DATA_BACKEND=sqlite."
            )

        if not self._lock.acquire(blocking=False):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Another operational refresh is already running.",
            )

        started = datetime.now(timezone.utc)
        run_key: str | None = None

        try:
            if not self.model_service.is_loaded:
                raise ServiceUnavailableError("Forecasting model is unavailable.")

            last_date = self.repository.get_last_historical_date()
            run_key = (
                f"refresh:{self.model_service.selected_model}:"
                f"{horizon_days}:{last_date.date().isoformat()}"
            )
            self.repository.record_application_run(
                run_key=run_key,
                job_name="recommendation_refresh",
                status="running",
                started_at=started.isoformat(),
                details={"horizon_days": horizon_days},
            )

            feature_summary = self.repository.rebuild_modeling_data()
            recommendation_payload = self.recommendation_service.get_recommendations(
                horizon_days=horizon_days
            )
            refreshed_last_date = self.repository.get_last_historical_date()
            finished = datetime.now(timezone.utc)
            details = {
                "features": feature_summary,
                "recommendations_count": recommendation_payload["count"],
                "last_historical_date": refreshed_last_date.date().isoformat(),
            }
            self.repository.record_application_run(
                run_key=run_key,
                job_name="recommendation_refresh",
                status="completed",
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                details=details,
            )

            return {
                "status": "completed",
                "run_key": run_key,
                "selected_model": recommendation_payload["selected_model"],
                "horizon_days": horizon_days,
                "features_rows": int(feature_summary["rows"]),
                "recommendations_count": int(recommendation_payload["count"]),
                "last_historical_date": refreshed_last_date.to_pydatetime(),
                "started_at": started,
                "finished_at": finished,
            }
        except Exception as exc:
            if run_key is not None:
                self._record_failure(run_key, started, exc)
            if isinstance(exc, HTTPException):
                raise
            logger.exception("Operational refresh failed")
            raise ServiceUnavailableError("Operational refresh failed.") from exc
        finally:
            self._lock.release()

    def _record_failure(
        self,
        run_key: str,
        started: datetime,
        error: Exception,
    ) -> None:
        try:
            self.repository.record_application_run(
                run_key=run_key,
                job_name="recommendation_refresh",
                status="failed",
                started_at=started.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                error_message=str(error),
            )
        except Exception:
            logger.exception("Could not record failed refresh")
