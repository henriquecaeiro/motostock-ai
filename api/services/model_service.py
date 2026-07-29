"""Model loading and lifecycle management."""

from __future__ import annotations

import logging

import joblib

from api.config import MODEL_PATH, SELECTED_MODEL_NAME
from api.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class ModelService:
    """Load and expose the production forecasting model."""

    def __init__(self, model_path=MODEL_PATH) -> None:
        self.model_path = model_path
        self.selected_model = SELECTED_MODEL_NAME
        self._model = None
        self._load_error: str | None = None

    def load(self) -> None:
        """Load the model artifact once at application startup."""

        if not self.model_path.exists():
            message = f"Model artifact is unavailable: {self.model_path.name}"
            logger.error(message)
            self._load_error = message
            return

        try:
            self._model = joblib.load(self.model_path)
            self._load_error = None
            logger.info("Loaded model from %s", self.model_path)
        except Exception as exc:
            message = "Failed to load the forecasting model."
            logger.exception(message)
            self._load_error = message
            raise ServiceUnavailableError(message) from exc

    @property
    def is_loaded(self) -> bool:
        """Return whether the model was loaded successfully."""

        return self._model is not None

    @property
    def model(self):
        """Return the loaded model or raise a service error."""

        if self._model is None:
            detail = self._load_error or "Forecasting model is not available."
            raise ServiceUnavailableError(detail)
        return self._model
