"""Real SQLite repository and import integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from api.database import SCHEMA_VERSION, check_database_integrity
from api.exceptions import ProductNotFoundError, ServiceUnavailableError
from api.repositories.sqlite_import import import_csv_files
from api.repositories.sqlite_repository import SqliteRepository


@pytest.fixture
def sqlite_repository(tmp_path: Path) -> tuple[SqliteRepository, dict]:
    daily_path = tmp_path / "daily_product_sales.csv"
    modeling_path = tmp_path / "modeling_dataset.csv"
    database_path = tmp_path / "motostock.db"

    pd.DataFrame(
        [
            {
                "product_name": "Widget A",
                "sale_date": "2026-01-01",
                "quantity_sold": 2,
                "total_revenue_brl": 20.0,
                "estimated_profit_brl": 8.0,
                "discount_pct": 0.0,
                "unit_price_brl": 10.0,
                "current_stock_snapshot": 5,
                "supplier_lead_time_days": 3,
                "product_category": "Test",
                "weather_condition": "Clear",
            },
            {
                "product_name": "Widget A",
                "sale_date": "2026-01-02",
                "quantity_sold": 0,
                "total_revenue_brl": 0.0,
                "estimated_profit_brl": 0.0,
                "discount_pct": 0.0,
                "unit_price_brl": None,
                "current_stock_snapshot": 4,
                "supplier_lead_time_days": 3,
                "product_category": "Test",
                "weather_condition": "Clear",
            },
            {
                "product_name": "Widget B",
                "sale_date": "2026-01-01",
                "quantity_sold": 1,
                "total_revenue_brl": 20.0,
                "estimated_profit_brl": 7.0,
                "discount_pct": 0.0,
                "unit_price_brl": 20.0,
                "current_stock_snapshot": 2,
                "supplier_lead_time_days": 5,
                "product_category": "Test",
                "weather_condition": "Cloudy",
            },
        ]
    ).to_csv(daily_path, index=False)

    pd.DataFrame(
        [
            {
                "sale_date": "2026-01-02",
                "quantity_sold": 0,
                "product_name": "Widget A",
                "unit_price_1d": None,
            },
            {
                "sale_date": "2026-01-01",
                "quantity_sold": 1,
                "product_name": "Widget B",
                "unit_price_1d": 20.0,
            },
        ]
    ).to_csv(modeling_path, index=False)

    summary = import_csv_files(
        database_path,
        daily_sales_path=daily_path,
        modeling_data_path=modeling_path,
    )
    return SqliteRepository(database_path), summary


def test_import_is_idempotent_and_preserves_zero_demand_days(sqlite_repository) -> None:
    repository, summary = sqlite_repository

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["products"] == 2
    assert summary["sales"] == 3
    assert summary["inventory"] == 3
    assert summary["modeling_rows"] == 2

    daily = repository.load_daily_sales()
    assert len(daily) == 3
    assert daily.loc[
        (daily["product_name"] == "Widget A")
        & (daily["sale_date"] == pd.Timestamp("2026-01-02")),
        "quantity_sold",
    ].iloc[0] == 0

    second_summary = import_csv_files(
        repository.database_path,
        daily_sales_path=Path(sqlite_repository[0].database_path).parent
        / "daily_product_sales.csv",
        modeling_data_path=Path(sqlite_repository[0].database_path).parent
        / "modeling_dataset.csv",
    )
    assert second_summary == summary


def test_sqlite_repository_matches_forecasting_repository_surface(sqlite_repository) -> None:
    repository, _ = sqlite_repository

    assert repository.list_products() == ["Widget A", "Widget B"]
    assert repository.product_exists("Widget A")
    assert not repository.product_exists("Widget' OR 1=1 --")
    assert len(repository.get_product_history("Widget A")) == 2
    assert repository.get_latest_prices().to_dict() == {
        "Widget A": 10.0,
        "Widget B": 20.0,
    }
    assert repository.get_last_historical_date() == pd.Timestamp("2026-01-02")

    state = repository.get_latest_business_state().set_index("product_name")
    assert state.loc["Widget A", "current_stock"] == 4
    assert state.loc["Widget A", "supplier_lead_time_days"] == 3


def test_sales_insert_is_transactional_and_idempotent(sqlite_repository) -> None:
    repository, _ = sqlite_repository
    sale = {
        "product_name": "Widget A",
        "sale_date": "2026-01-03",
        "quantity_sold": 3,
        "unit_price_brl": 12.0,
        "external_id": "order-1",
        "current_stock_snapshot": 7,
        "supplier_lead_time_days": 4,
    }

    first = repository.insert_sales([sale])
    second = repository.insert_sales([sale])

    assert first["inserted"] == 1
    assert second == {"inserted": 0, "skipped": 1, "sale_ids": []}
    assert len(repository.load_daily_sales()) == 4
    state = repository.get_latest_business_state().set_index("product_name")
    assert state.loc["Widget A", "current_stock"] == 7

    with pytest.raises(ValueError, match="positive"):
        repository.insert_sales(
            [
                {
                    "product_name": "Widget A",
                    "sale_date": "2026-01-04",
                    "quantity_sold": 0,
                    "unit_price_brl": 12,
                }
            ]
        )

    with pytest.raises(ProductNotFoundError):
        repository.insert_sales(
            [
                {
                    "product_name": "Unknown",
                    "sale_date": "2026-01-04",
                    "quantity_sold": 1,
                    "unit_price_brl": 12,
                }
            ]
        )


def test_recommendations_are_persisted_and_read_back(sqlite_repository) -> None:
    repository, _ = sqlite_repository
    payload = {
        "generated_at": "2026-01-02T12:00:00+00:00",
        "selected_model": "xgboost",
        "horizon_days": 14,
        "recommendations": [
            {
                "product_name": "Widget A",
                "forecast_horizon_days": 14,
                "forecasted_demand_raw": 10.5,
                "forecasted_demand_non_negative": 10.5,
                "forecasted_demand_units": 11,
                "current_stock": 4,
                "safety_stock": 3,
                "required_stock": 14,
                "recommended_purchase_quantity": 10,
                "supplier_lead_time_days": 3,
                "stock_status": "critical",
                "priority_score": 100,
            }
        ],
    }

    repository.save_recommendations(payload)
    stored = repository.get_latest_recommendations(14)

    assert stored is not None
    assert stored["count"] == 1
    assert stored["recommendations"][0]["product_name"] == "Widget A"
    assert check_database_integrity(repository.database_path)


def test_corrupt_database_is_reported_as_unavailable(tmp_path: Path) -> None:
    database_path = tmp_path / "corrupt.db"
    database_path.write_bytes(b"not a sqlite database")

    with pytest.raises(ServiceUnavailableError, match="SQLite database"):
        SqliteRepository(database_path)
