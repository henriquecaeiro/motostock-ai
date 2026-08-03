"""Leakage-safe daily feature engineering shared by training and operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "quantity_sold"
NUMERIC_FEATURES = [
    "unit_price_1d",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "quantity_1d",
    "quantity_7d",
    "quantity_14d",
    "rolling_1d",
    "rolling_7d",
    "rolling_14d",
]
CATEGORICAL_FEATURES = ["product_name"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class TemporalSplit:
    """A date-disjoint train/test split and its audit metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def build_modeling_dataset(daily_sales: pd.DataFrame) -> pd.DataFrame:
    """Build one product/day row using only information available before the day.

    Missing dates are explicit zero-demand days. Lag and rolling features are
    calculated after shifting demand, so the target for a date never enters
    its own features.
    """

    required = {"product_name", "sale_date", "quantity_sold", "unit_price_brl"}
    missing = sorted(required.difference(daily_sales.columns))
    if missing:
        raise ValueError(f"Daily sales data is missing columns: {', '.join(missing)}")

    source = daily_sales.copy()
    source["sale_date"] = pd.to_datetime(source["sale_date"], errors="coerce")
    if source["sale_date"].isna().any():
        raise ValueError("Daily sales data contains invalid sale_date values.")

    source["quantity_sold"] = pd.to_numeric(
        source["quantity_sold"], errors="raise"
    ).astype(float)
    source["unit_price_brl"] = pd.to_numeric(
        source["unit_price_brl"], errors="coerce"
    )

    daily_rows: list[pd.DataFrame] = []
    for product_name, group in source.groupby("product_name", sort=True):
        grouped = (
            group.groupby("sale_date", as_index=True)
            .agg(
                quantity_sold=("quantity_sold", "sum"),
                unit_price_brl=("unit_price_brl", "mean"),
            )
            .sort_index()
        )
        full_dates = pd.date_range(
            grouped.index.min(), grouped.index.max(), freq="D"
        )
        quantity = grouped["quantity_sold"].reindex(full_dates).fillna(0.0)
        price = grouped["unit_price_brl"].reindex(full_dates)
        previous_quantity = quantity.shift(1)
        previous_price = price.shift(1)

        frame = pd.DataFrame(
            {
                "sale_date": full_dates,
                "quantity_sold": quantity.to_numpy(),
                "unit_price_1d": previous_price.to_numpy(),
                "day_of_week": full_dates.dayofweek,
                "day_of_month": full_dates.day,
                "month": full_dates.month,
                "week_of_year": [int(value) for value in full_dates.isocalendar().week],
                "is_weekend": [int(value in (5, 6)) for value in full_dates.dayofweek],
                "quantity_1d": previous_quantity.to_numpy(),
                "quantity_7d": quantity.shift(7).to_numpy(),
                "quantity_14d": quantity.shift(14).to_numpy(),
                "rolling_1d": previous_quantity.rolling(1).mean().to_numpy(),
                "rolling_7d": previous_quantity.rolling(7).mean().to_numpy(),
                "rolling_14d": previous_quantity.rolling(14).mean().to_numpy(),
                "product_name": str(product_name),
            }
        )
        daily_rows.append(frame.dropna(subset=FEATURE_COLUMNS))

    if not daily_rows:
        return pd.DataFrame(columns=["sale_date", TARGET_COLUMN, *FEATURE_COLUMNS])

    result = pd.concat(daily_rows, ignore_index=True)
    return result[
        ["sale_date", TARGET_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    ].sort_values(["sale_date", "product_name"]).reset_index(drop=True)


def temporal_split(
    modeling_data: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
) -> TemporalSplit:
    """Split by unique dates, never by random rows."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one.")
    if modeling_data.empty:
        raise ValueError("Cannot split an empty modeling dataset.")

    dates = sorted(pd.to_datetime(modeling_data["sale_date"]).unique())
    split_index = max(1, min(len(dates) - 1, int(len(dates) * (1 - test_fraction))))
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]
    if not train_dates or not test_dates:
        raise ValueError("Temporal split needs at least one train and one test date.")

    train = modeling_data[modeling_data["sale_date"].isin(train_dates)].copy()
    test = modeling_data[modeling_data["sale_date"].isin(test_dates)].copy()
    if set(train["sale_date"]).intersection(set(test["sale_date"])):
        raise AssertionError("Temporal split leaked dates between train and test.")

    return TemporalSplit(
        train=train,
        test=test,
        train_start=pd.Timestamp(train_dates[0]).date().isoformat(),
        train_end=pd.Timestamp(train_dates[-1]).date().isoformat(),
        test_start=pd.Timestamp(test_dates[0]).date().isoformat(),
        test_end=pd.Timestamp(test_dates[-1]).date().isoformat(),
    )
