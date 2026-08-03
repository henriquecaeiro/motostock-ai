"""Unit tests for the clean-clone data preparation path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import api.repositories.factory as repository_factory
from api.config import Settings
from api.repositories.factory import create_repository
from src.data_preparation import prepare_processed_data


pytestmark = pytest.mark.unit


def test_prepare_processed_data_builds_complete_daily_and_modeling_files(tmp_path):
    raw_path = tmp_path / "raw.csv"
    daily_path = tmp_path / "processed" / "daily.csv"
    modeling_path = tmp_path / "processed" / "modeling.csv"

    records = []
    for product_name, category, lead_time in (
        ("A", "Gear", 3),
        ("B", "Parts", 4),
    ):
        for day in range(16):
            quantity = (day % 3) + 1
            records.append(
                {
                    "sale_date": f"2025-01-{day + 1:02d}",
                    "sale_time": "10:00",
                    "product_name": product_name,
                    "product_category": category,
                    "quantity_sold": quantity,
                    "total_revenue_brl": quantity * 10,
                    "estimated_profit_brl": quantity * 5,
                    "discount_pct": 0,
                    "unit_price_brl": 10,
                    "current_stock_snapshot": 20 - day,
                    "supplier_lead_time_days": lead_time,
                    "weather_condition": "Sunny",
                }
            )
    pd.DataFrame(records).to_csv(raw_path, index=False)

    summary = prepare_processed_data(raw_path, daily_path, modeling_path)

    daily = pd.read_csv(daily_path)
    modeling = pd.read_csv(modeling_path)

    assert summary["products"] == 2
    assert len(daily) == 32
    assert len(modeling) == 4
    assert daily[["product_name", "sale_date"]].drop_duplicates().shape[0] == 32
    assert set(modeling.columns) >= {"quantity_sold", "product_name"}


def test_empty_sqlite_bootstraps_processed_data_from_raw_csv(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    raw_path = root / "data" / "raw" / "motoretail.csv"
    daily_path = tmp_path / "daily_product_sales.csv"
    modeling_path = tmp_path / "modeling_dataset.csv"
    database_path = tmp_path / "bootstrap.db"

    monkeypatch.setattr(repository_factory, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(repository_factory, "DAILY_SALES_PATH", daily_path)
    monkeypatch.setattr(repository_factory, "MODELING_DATA_PATH", modeling_path)

    repository = create_repository(
        Settings(
            data_backend="sqlite",
            database_path=database_path,
            auto_import_csv=True,
        )
    )

    assert daily_path.exists()
    assert modeling_path.exists()
    assert len(repository.list_products()) == 12
