"""Recursive demand forecasting for future dates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HORIZON_DAYS = 14
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 30
MIN_HISTORY_DAYS = 14


class ForecastingError(Exception):
    """Raised when forecasting cannot be completed."""


def get_model_feature_columns(model: Any) -> list[str]:
    """Return model feature names in the order expected by the pipeline."""

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ForecastingError("Model does not expose feature_names_in_")

    return list(feature_names)


def _build_future_row(
    future_date: pd.Timestamp,
    product_name: str,
    unit_price: float,
    demand_history: list[float],
) -> dict[str, Any]:
    """Build one feature row for a single future date."""

    return {
        "unit_price_1d": unit_price,
        "day_of_week": future_date.dayofweek,
        "day_of_month": future_date.day,
        "month": future_date.month,
        "week_of_year": int(future_date.isocalendar().week),
        "is_weekend": int(future_date.dayofweek in [5, 6]),
        "quantity_1d": demand_history[-1],
        "quantity_7d": demand_history[-7],
        "quantity_14d": demand_history[-14],
        "rolling_1d": float(np.mean(demand_history[-1:])),
        "rolling_7d": float(np.mean(demand_history[-7:])),
        "rolling_14d": float(np.mean(demand_history[-14:])),
        "product_name": product_name,
    }


def forecast_product_demand(
    model: Any,
    product_name: str,
    product_history: pd.DataFrame,
    latest_unit_price: float,
    last_historical_date: pd.Timestamp,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Forecast demand for one product over a future horizon."""

    if horizon_days < MIN_HORIZON_DAYS or horizon_days > MAX_HORIZON_DAYS:
        raise ForecastingError(
            f"horizon_days must be between {MIN_HORIZON_DAYS} and {MAX_HORIZON_DAYS}"
        )

    if product_history.empty:
        raise ForecastingError(f"No historical data available for '{product_name}'")

    history = product_history.sort_values("sale_date")
    demand_history = history["quantity_sold"].astype(float).tolist()

    if len(demand_history) < MIN_HISTORY_DAYS:
        raise ForecastingError(
            f"Insufficient history for '{product_name}': "
            f"need at least {MIN_HISTORY_DAYS} days, found {len(demand_history)}"
        )

    feature_columns = get_model_feature_columns(model)
    future_dates = pd.date_range(
        last_historical_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    future_rows: list[dict[str, Any]] = []

    for future_date in future_dates:
        row = _build_future_row(
            future_date=future_date,
            product_name=product_name,
            unit_price=latest_unit_price,
            demand_history=demand_history,
        )

        try:
            forecast_raw = float(
                model.predict(pd.DataFrame([row])[feature_columns])[0]
            )
        except Exception as exc:
            raise ForecastingError(
                f"Prediction failed for '{product_name}' on {future_date.date()}"
            ) from exc

        forecast_non_negative = max(forecast_raw, 0.0)

        future_rows.append(
            {
                "product_name": product_name,
                "sale_date": future_date,
                "forecast_raw": forecast_raw,
                "forecast_non_negative": forecast_non_negative,
            }
        )
        demand_history.append(forecast_non_negative)

    return pd.DataFrame(future_rows)


def aggregate_product_forecast(
    daily_forecast_df: pd.DataFrame,
    horizon_days: int,
) -> dict[str, Any]:
    """Aggregate daily forecasts into product-level totals."""

    forecasted_demand_raw = float(daily_forecast_df["forecast_raw"].sum())
    forecasted_demand_non_negative = float(
        daily_forecast_df["forecast_non_negative"].sum()
    )
    forecasted_demand_units = int(
        math.ceil(forecasted_demand_non_negative)
    )

    return {
        "forecast_horizon_days": horizon_days,
        "forecasted_demand_raw": forecasted_demand_raw,
        "forecasted_demand_non_negative": forecasted_demand_non_negative,
        "forecasted_demand_units": forecasted_demand_units,
        "forecast_start_date": daily_forecast_df["sale_date"].min().date(),
        "forecast_end_date": daily_forecast_df["sale_date"].max().date(),
    }


def forecast_all_products(
    model: Any,
    daily_sales_df: pd.DataFrame,
    latest_prices: pd.Series,
    last_historical_date: pd.Timestamp,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Forecast demand for every product in the daily sales dataset."""

    product_forecasts: list[pd.DataFrame] = []

    for product_name, product_history in daily_sales_df.groupby("product_name"):
        if product_name not in latest_prices.index:
            raise ForecastingError(
                f"Latest unit price not found for '{product_name}'"
            )

        daily_forecast = forecast_product_demand(
            model=model,
            product_name=product_name,
            product_history=product_history,
            latest_unit_price=float(latest_prices.loc[product_name]),
            last_historical_date=last_historical_date,
            horizon_days=horizon_days,
        )
        product_forecasts.append(daily_forecast)

    return pd.concat(product_forecasts, ignore_index=True)
