"""Unit tests for aggregate forecast summaries."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from api.exceptions import ForecastingHTTPError
from api.services.forecast_service import ForecastService


class EmptyRepository:
    def load_daily_sales(self):
        return pd.DataFrame()


class StubModelService:
    selected_model = "fake-model"


def make_service() -> ForecastService:
    return ForecastService(
        repository=EmptyRepository(),
        model_service=StubModelService(),
    )


def test_forecast_summary_sorts_ties_and_applies_limit(monkeypatch) -> None:
    service = make_service()
    forecast_by_product = pd.DataFrame(
        [
            {
                "product_name": "Gamma",
                "forecasted_demand_units": 7,
                "forecasted_demand_raw": 6.5,
                "forecasted_demand_non_negative": 7.1,
            },
            {
                "product_name": "Beta",
                "forecasted_demand_units": 10,
                "forecasted_demand_raw": 9.4,
                "forecasted_demand_non_negative": 10.2,
            },
            {
                "product_name": "Alpha",
                "forecasted_demand_units": 10,
                "forecasted_demand_raw": 9.6,
                "forecasted_demand_non_negative": 10.1,
            },
        ]
    )
    monkeypatch.setattr(
        service,
        "forecast_all_products",
        lambda horizon_days: forecast_by_product,
    )

    summary = service.get_forecast_summary(horizon_days=14, limit=2)

    assert summary["selected_model"] == "fake-model"
    assert summary["horizon_days"] == 14
    assert summary["total_products"] == 3
    assert summary["total_forecasted_demand_units"] == 27
    assert [item["product_name"] for item in summary["top_products"]] == [
        "Alpha",
        "Beta",
    ]
    json.dumps(summary)


def test_forecast_summary_sums_final_units_without_double_rounding(monkeypatch) -> None:
    service = make_service()
    forecast_by_product = pd.DataFrame(
        [
            {
                "product_name": "Rounded A",
                "forecasted_demand_units": 2,
                "forecasted_demand_raw": 1.1,
                "forecasted_demand_non_negative": 1.1,
            },
            {
                "product_name": "Rounded B",
                "forecasted_demand_units": 2,
                "forecasted_demand_raw": 1.1,
                "forecasted_demand_non_negative": 1.1,
            },
            {
                "product_name": "Negative",
                "forecasted_demand_units": 0,
                "forecasted_demand_raw": -4.0,
                "forecasted_demand_non_negative": 0.0,
            },
        ]
    )
    monkeypatch.setattr(
        service,
        "forecast_all_products",
        lambda horizon_days: forecast_by_product,
    )

    summary = service.get_forecast_summary()

    assert summary["total_forecasted_demand_units"] == 4
    assert summary["top_products"][-1]["forecasted_demand_units"] == 0


def test_forecast_summary_handles_empty_dataframe() -> None:
    summary = make_service().get_forecast_summary()

    assert summary == {
        "selected_model": "fake-model",
        "horizon_days": 14,
        "total_products": 0,
        "total_forecasted_demand_units": 0,
        "top_products": [],
    }


def test_forecast_summary_propagates_forecast_errors(monkeypatch) -> None:
    service = make_service()

    def raise_error(horizon_days):
        raise ForecastingHTTPError("forecast failed")

    monkeypatch.setattr(service, "forecast_all_products", raise_error)

    with pytest.raises(ForecastingHTTPError, match="forecast failed"):
        service.get_forecast_summary()
