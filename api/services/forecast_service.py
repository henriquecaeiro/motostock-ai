"""Forecasting service built on the reusable forecasting module."""

from __future__ import annotations

import logging

import pandas as pd

from src.forecasting import (
    ForecastingError,
    aggregate_product_forecast,
    forecast_all_products as build_all_product_forecasts,
    forecast_product_demand,
)

from api.exceptions import ForecastingHTTPError, ProductNotFoundError
from api.repositories.protocol import DataRepository
from api.services.model_service import ModelService

logger = logging.getLogger(__name__)


class ForecastService:
    """Generate demand forecasts for one or all products."""

    def __init__(
        self,
        repository: DataRepository,
        model_service: ModelService,
    ) -> None:
        self.repository = repository
        self.model_service = model_service

    def predict_product(self, product_name: str, horizon_days: int) -> dict:
        """Forecast demand for a single product."""

        if not self.repository.product_exists(product_name):
            raise ProductNotFoundError(product_name)

        product_history = self.repository.get_product_history(product_name)
        latest_prices = self.repository.get_latest_prices()

        if product_name not in latest_prices.index:
            raise ForecastingHTTPError(
                f"Latest unit price not found for '{product_name}'."
            )

        try:
            daily_forecast = forecast_product_demand(
                model=self.model_service.model,
                product_name=product_name,
                product_history=product_history,
                latest_unit_price=float(latest_prices.loc[product_name]),
                last_historical_date=self.repository.get_last_historical_date(),
                horizon_days=horizon_days,
            )
            aggregated = aggregate_product_forecast(daily_forecast, horizon_days)
        except ForecastingError as exc:
            logger.exception("Forecast failed for product '%s'", product_name)
            raise ForecastingHTTPError(str(exc)) from exc

        return {
            "product_name": product_name,
            "horizon_days": horizon_days,
            **aggregated,
            "daily_forecast": [
                {
                    "date": row["sale_date"].date(),
                    "forecast_raw": float(row["forecast_raw"]),
                    "forecast_non_negative": float(row["forecast_non_negative"]),
                }
                for _, row in daily_forecast.iterrows()
            ],
        }

    def forecast_all_products(self, horizon_days: int):
        """Forecast demand for every known product."""

        daily_sales_df = self.repository.load_daily_sales()

        if daily_sales_df.empty:
            return pd.DataFrame(
                columns=[
                    "product_name",
                    "forecast_horizon_days",
                    "forecasted_demand_raw",
                    "forecasted_demand_non_negative",
                    "forecasted_demand_units",
                ]
            )

        try:
            daily_forecast = build_all_product_forecasts(
                model=self.model_service.model,
                daily_sales_df=daily_sales_df,
                latest_prices=self.repository.get_latest_prices(),
                last_historical_date=self.repository.get_last_historical_date(),
                horizon_days=horizon_days,
            )
        except ForecastingError as exc:
            logger.exception("Batch forecast failed")
            raise ForecastingHTTPError(str(exc)) from exc

        forecast_by_product = (
            daily_forecast.groupby("product_name", as_index=False)
            .agg(
                forecast_horizon_days=("sale_date", "nunique"),
                forecasted_demand_raw=("forecast_raw", "sum"),
                forecasted_demand_non_negative=("forecast_non_negative", "sum"),
            )
        )

        import math

        forecast_by_product["forecasted_demand_units"] = forecast_by_product[
            "forecasted_demand_non_negative"
        ].apply(lambda value: int(math.ceil(value)))

        return forecast_by_product

    def get_forecast_summary(
        self,
        horizon_days: int = 14,
        limit: int = 5,
    ) -> dict:
        """Return a deterministic, JSON-native summary for all products."""

        if not 1 <= horizon_days <= 30:
            raise ForecastingHTTPError("horizon_days must be between 1 and 30")

        if not 1 <= limit <= 20:
            raise ForecastingHTTPError("limit must be between 1 and 20")

        forecast_by_product = self.forecast_all_products(horizon_days)
        ordered = forecast_by_product.sort_values(
            by=["forecasted_demand_units", "product_name"],
            ascending=[False, True],
            kind="mergesort",
        )

        top_products = [
            {
                "product_name": str(row["product_name"]),
                "forecasted_demand_units": int(
                    row["forecasted_demand_units"]
                ),
                "forecasted_demand_raw": float(row["forecasted_demand_raw"]),
                "forecasted_demand_non_negative": float(
                    row["forecasted_demand_non_negative"]
                ),
            }
            for _, row in ordered.head(limit).iterrows()
        ]

        return {
            "selected_model": self.model_service.selected_model,
            "horizon_days": int(horizon_days),
            "total_products": int(len(forecast_by_product)),
            "total_forecasted_demand_units": int(
                forecast_by_product["forecasted_demand_units"].sum()
            ),
            "top_products": top_products,
        }
