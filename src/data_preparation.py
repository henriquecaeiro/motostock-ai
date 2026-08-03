"""Deterministic preparation of the tracked raw dataset for serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.training.features import build_modeling_dataset


RAW_REQUIRED_COLUMNS = {
    "sale_date",
    "sale_time",
    "product_name",
    "product_category",
    "quantity_sold",
    "total_revenue_brl",
    "estimated_profit_brl",
    "discount_pct",
    "unit_price_brl",
    "current_stock_snapshot",
    "supplier_lead_time_days",
    "weather_condition",
}


def prepare_processed_data(
    raw_path: str | Path,
    daily_output_path: str | Path,
    modeling_output_path: str | Path,
) -> dict[str, Any]:
    """Build the daily and modeling CSVs from the raw transaction dataset.

    The transformations mirror the preparation and feature-engineering
    notebooks, but are executable from a clean clone and do not depend on
    notebook state.
    """

    raw_path = Path(raw_path)
    daily_output_path = Path(daily_output_path)
    modeling_output_path = Path(modeling_output_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset is unavailable: {raw_path}")

    frame = pd.read_csv(raw_path)
    missing = sorted(RAW_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "Raw dataset is missing required columns: " + ", ".join(missing)
        )

    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="raise")
    frame["sale_timestamp"] = pd.to_datetime(
        frame["sale_date"].dt.strftime("%Y-%m-%d") + " " + frame["sale_time"],
        errors="raise",
    )
    frame = frame.sort_values(["product_name", "sale_timestamp"]).copy()

    def most_frequent_value(values: pd.Series) -> Any:
        modes = values.mode().sort_values()
        if modes.empty:
            return None
        return modes.iloc[0]

    weather_by_date = (
        frame.groupby("sale_date")["weather_condition"]
        .agg(most_frequent_value)
        .rename("weather_condition")
        .reset_index()
    )

    daily_sales = (
        frame.groupby(
            ["product_name", "sale_date"],
            as_index=False,
            observed=True,
        )
        .agg(
            quantity_sold=("quantity_sold", "sum"),
            total_revenue_brl=("total_revenue_brl", "sum"),
            estimated_profit_brl=("estimated_profit_brl", "sum"),
            discount_pct=("discount_pct", "mean"),
            unit_price_brl=("unit_price_brl", "mean"),
            current_stock_snapshot=("current_stock_snapshot", "last"),
            supplier_lead_time_days=("supplier_lead_time_days", "last"),
        )
    )

    product_category = (
        frame.groupby("product_name")["product_category"]
        .agg(most_frequent_value)
        .rename("product_category")
        .reset_index()
    )

    full_grid = pd.MultiIndex.from_product(
        [
            sorted(frame["product_name"].unique()),
            pd.date_range(frame["sale_date"].min(), frame["sale_date"].max(), freq="D"),
        ],
        names=["product_name", "sale_date"],
    ).to_frame(index=False)

    daily = (
        full_grid.merge(daily_sales, on=["product_name", "sale_date"], how="left")
        .merge(product_category, on="product_name", how="left")
        .merge(weather_by_date, on="sale_date", how="left")
        .sort_values(["product_name", "sale_date"])
        .reset_index(drop=True)
    )

    flow_columns = ["quantity_sold", "total_revenue_brl", "estimated_profit_brl"]
    daily[flow_columns] = daily[flow_columns].fillna(0)
    daily["discount_pct"] = daily["discount_pct"].fillna(0)

    snapshot_columns = [
        "unit_price_brl",
        "current_stock_snapshot",
        "supplier_lead_time_days",
    ]
    daily[snapshot_columns] = daily.groupby("product_name", sort=False)[
        snapshot_columns
    ].ffill()
    daily["quantity_sold"] = daily["quantity_sold"].astype(int)

    key_columns = [
        "product_name",
        "sale_date",
        "quantity_sold",
        "product_category",
        "weather_condition",
    ]
    if daily[key_columns].isna().any().any():
        raise ValueError("Prepared daily data contains missing key values.")
    if (daily["quantity_sold"] < 0).any():
        raise ValueError("Prepared daily data contains negative quantities.")

    modeling = build_modeling_dataset(daily)
    if modeling.empty:
        raise ValueError("Prepared modeling data is empty.")

    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    modeling_output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_output_path, index=False, encoding="utf-8")
    modeling.to_csv(modeling_output_path, index=False, encoding="utf-8")

    return {
        "raw_path": str(raw_path.resolve()),
        "daily_output_path": str(daily_output_path.resolve()),
        "modeling_output_path": str(modeling_output_path.resolve()),
        "daily_rows": int(len(daily)),
        "modeling_rows": int(len(modeling)),
        "products": int(daily["product_name"].nunique()),
        "date_start": daily["sale_date"].min().date().isoformat(),
        "date_end": daily["sale_date"].max().date().isoformat(),
    }
