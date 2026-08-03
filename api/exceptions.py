"""Application-specific exceptions."""

from fastapi import HTTPException, status


class ProductNotFoundError(HTTPException):
    """Raised when a product name is not found in the dataset."""

    def __init__(self, product_name: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_name}' was not found.",
        )


class ModelVersionNotFoundError(HTTPException):
    """Raised when an administrative model version is not registered."""

    def __init__(self, version: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version}' was not found.",
        )


class ServiceUnavailableError(HTTPException):
    """Raised when an essential artifact or dataset is unavailable."""

    def __init__(self, message: str) -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=message,
        )


class ForecastingHTTPError(HTTPException):
    """Raised when forecasting fails for a known product."""

    def __init__(self, message: str) -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )


class RecommendationHTTPError(HTTPException):
    """Raised when recommendation generation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )
