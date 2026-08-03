"""Stock recommendation service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.recommendation import generate_stock_recommendations

from api.config import DEFAULT_LEAD_TIME_DAYS, SELECTED_MODEL_NAME, VALID_STOCK_STATUSES
from api.exceptions import ProductNotFoundError, RecommendationHTTPError
from api.repositories.protocol import DataRepository
from api.services.forecast_service import ForecastService
from api.services.model_service import ModelService

logger = logging.getLogger(__name__)


class RecommendationService:
    """Build stock recommendations from forecasts and business state."""

    def __init__(
        self,
        repository: DataRepository,
        forecast_service: ForecastService,
        model_service: ModelService,
    ) -> None:
        self.repository = repository
        self.forecast_service = forecast_service
        self.model_service = model_service

    def _build_recommendations_df(self, horizon_days: int) -> pd.DataFrame:
        forecast_by_product = self.forecast_service.forecast_all_products(
            horizon_days=horizon_days
        )
        business_state = self.repository.get_latest_business_state()

        current_stock_df = business_state[["product_name", "current_stock"]].copy()
        lead_time_df = business_state[
            ["product_name", "supplier_lead_time_days"]
        ].copy()

        try:
            return generate_stock_recommendations(
                forecast_by_product_df=forecast_by_product,
                current_stock_df=current_stock_df,
                lead_time_df=lead_time_df,
                default_lead_time_days=DEFAULT_LEAD_TIME_DAYS,
            )
        except Exception as exc:
            logger.exception("Recommendation generation failed")
            raise RecommendationHTTPError(
                "Failed to generate stock recommendations."
            ) from exc

    def get_recommendations(
        self,
        horizon_days: int = 14,
        stock_status: str | None = None,
        product_name: str | None = None,
    ) -> dict:
        """Return filtered stock recommendations."""

        if product_name and not self.repository.product_exists(product_name):
            raise ProductNotFoundError(product_name)

        recommendations_df = self._build_recommendations_df(horizon_days)

        if product_name:
            recommendations_df = recommendations_df[
                recommendations_df["product_name"] == product_name
            ]

        if stock_status:
            if stock_status not in VALID_STOCK_STATUSES:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Invalid stock_status '{stock_status}'. "
                        f"Allowed values: {', '.join(VALID_STOCK_STATUSES)}"
                    ),
                )

            recommendations_df = recommendations_df[
                recommendations_df["stock_status"].astype(str) == stock_status
            ]

        recommendations = self._dataframe_to_records(recommendations_df)

        response = {
            "generated_at": datetime.now(timezone.utc),
            "selected_model": self.model_service.selected_model,
            "horizon_days": horizon_days,
            "count": len(recommendations),
            "recommendations": recommendations,
        }
        self.repository.save_recommendations(response)
        return response

    def get_summary(self, horizon_days: int = 14) -> dict:
        """Return summary counts derived from the current recommendations."""

        recommendations_df = self._build_recommendations_df(horizon_days)

        status_counts = (
            recommendations_df["stock_status"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        highest_priority_product = None
        if not recommendations_df.empty:
            top_row = recommendations_df.sort_values(
                by=["priority_score", "recommended_purchase_quantity"],
                ascending=[False, False],
            ).iloc[0]
            highest_priority_product = str(top_row["product_name"])

        return {
            "selected_model": self.model_service.selected_model,
            "forecast_horizon_days": horizon_days,
            "total_products": int(len(recommendations_df)),
            "critical_products": int(status_counts.get("critical", 0)),
            "warning_products": int(status_counts.get("warning", 0)),
            "healthy_products": int(status_counts.get("healthy", 0)),
            "overstock_products": int(status_counts.get("overstock", 0)),
            "total_recommended_purchase_units": int(
                recommendations_df["recommended_purchase_quantity"].sum()
            ),
            "highest_priority_product": highest_priority_product,
        }

    @staticmethod
    def _dataframe_to_records(df: pd.DataFrame) -> list[dict]:
        records: list[dict] = []

        for _, row in df.iterrows():
            records.append(
                {
                    "product_name": str(row["product_name"]),
                    "forecast_horizon_days": int(row["forecast_horizon_days"]),
                    "forecasted_demand_raw": float(row["forecasted_demand_raw"]),
                    "forecasted_demand_non_negative": float(
                        row["forecasted_demand_non_negative"]
                    ),
                    "forecasted_demand_units": int(row["forecasted_demand_units"]),
                    "current_stock": int(row["current_stock"]),
                    "safety_stock": int(row["safety_stock"]),
                    "required_stock": int(row["required_stock"]),
                    "recommended_purchase_quantity": int(
                        row["recommended_purchase_quantity"]
                    ),
                    "supplier_lead_time_days": int(row["supplier_lead_time_days"]),
                    "stock_status": str(row["stock_status"]),
                    "priority_score": int(row["priority_score"]),
                }
            )

        return records
