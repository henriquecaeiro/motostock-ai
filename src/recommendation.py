import numpy as np
import pandas as pd


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Check whether the DataFrame contains all required columns."""

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def calculate_safety_stock(
    df: pd.DataFrame, safety_stock_pct: float = 0.20
) -> pd.DataFrame:
    """Calculate safety stock as a percentage of forecasted demand."""

    _validate_columns(df, ["forecasted_demand_units"])

    result = df.copy()

    if safety_stock_pct < 0:
        raise ValueError("safety_stock_pct must be non-negative")

    result["forecasted_demand_units"] = (
        pd.to_numeric(result["forecasted_demand_units"], errors="coerce")
        .fillna(0).clip(lower=0).astype(int)
    )
    result["safety_stock"] = np.ceil(
        result["forecasted_demand_units"] * safety_stock_pct
    ).astype(int)

    return result


def calculate_recommended_purchase(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate required stock and recommended purchase quantity."""

    _validate_columns(df, ["forecasted_demand_units", "safety_stock", "current_stock"])

    result = df.copy()

    result["forecasted_demand_units"] = pd.to_numeric(
        result["forecasted_demand_units"], errors="coerce"
    ).fillna(0).clip(lower=0).astype(int)
    result["safety_stock"] = pd.to_numeric(
        result["safety_stock"], errors="coerce"
    ).fillna(0).clip(lower=0).astype(int)
    result["current_stock"] = pd.to_numeric(
        result["current_stock"], errors="coerce"
    ).fillna(0).clip(lower=0).round().astype(int)

    result["required_stock"] = (
        result["forecasted_demand_units"] + result["safety_stock"]
    ).astype(int)

    result["recommended_purchase_quantity"] = (
        (result["required_stock"] - result["current_stock"]).clip(lower=0).astype(int)
    )

    return result


def classify_stock_status(
    df: pd.DataFrame, overstock_multiplier: float = 1.50
) -> pd.DataFrame:
    """Classify each product according to its stock risk."""

    _validate_columns(
        df, ["forecasted_demand_units", "required_stock", "current_stock"]
    )

    result = df.copy()
    result["forecasted_demand_units"] = pd.to_numeric(
        result["forecasted_demand_units"], errors="coerce"
    ).fillna(0).clip(lower=0)
    result["required_stock"] = pd.to_numeric(
        result["required_stock"], errors="coerce"
    ).fillna(0).clip(lower=0)
    result["current_stock"] = pd.to_numeric(
        result["current_stock"], errors="coerce"
    ).fillna(0).clip(lower=0)

    critical_condition = result["current_stock"] < result["forecasted_demand_units"]

    warning_condition = (
        result["current_stock"] >= result["forecasted_demand_units"]
    ) & (result["current_stock"] < result["required_stock"])

    overstock_condition = (
        (result["required_stock"] > 0)
        & (result["current_stock"] > result["required_stock"] * overstock_multiplier)
    ) | ((result["required_stock"] == 0) & (result["current_stock"] > 0))

    result["stock_status"] = np.select(
        [critical_condition, warning_condition, overstock_condition],
        ["critical", "warning", "overstock"],
        default="healthy",
    )

    return result


def calculate_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create a priority score from 0 to 100."""

    _validate_columns(
        df, ["recommended_purchase_quantity", "supplier_lead_time_days", "stock_status"]
    )

    result = df.copy()
    result["recommended_purchase_quantity"] = pd.to_numeric(
        result["recommended_purchase_quantity"], errors="coerce"
    ).fillna(0).clip(lower=0)
    result["supplier_lead_time_days"] = pd.to_numeric(
        result["supplier_lead_time_days"], errors="coerce"
    ).fillna(0).clip(lower=0)

    risk_weights = {
        "critical": 1.50,
        "warning": 1.20,
        "healthy": 1.00,
        "overstock": 0.00,
    }

    risk_weight = result["stock_status"].map(risk_weights).fillna(1.00)

    priority_score_raw = (
        result["recommended_purchase_quantity"]
        * result["supplier_lead_time_days"]
        * risk_weight
    )

    max_priority = priority_score_raw.max()

    if max_priority > 0:
        result["priority_score"] = (
            (priority_score_raw / max_priority * 100).round().astype(int)
        )
    else:
        result["priority_score"] = 0

    return result


def generate_stock_recommendations(
    forecast_by_product_df: pd.DataFrame,
    current_stock_df: pd.DataFrame,
    lead_time_df: pd.DataFrame | None = None,
    default_lead_time_days: int = 7,
    safety_stock_pct: float = 0.20,
    overstock_multiplier: float = 1.50,
) -> pd.DataFrame:
    """Generate the complete stock recommendation table."""

    _validate_columns(
        forecast_by_product_df,
        [
            "product_name",
            "forecast_horizon_days",
            "forecasted_demand_raw",
            "forecasted_demand_non_negative",
            "forecasted_demand_units",
        ],
    )

    _validate_columns(current_stock_df, ["product_name", "current_stock"])

    stock_df = current_stock_df.drop_duplicates(subset="product_name", keep="last")

    result = forecast_by_product_df.merge(stock_df, on="product_name", how="left")

    result["current_stock"] = (
        result["current_stock"].fillna(0).clip(lower=0).round().astype(int)
    )

    if lead_time_df is not None:
        _validate_columns(lead_time_df, ["product_name", "supplier_lead_time_days"])

        lead_time_df = lead_time_df.drop_duplicates(subset="product_name", keep="last")

        result = result.merge(lead_time_df, on="product_name", how="left")
    else:
        result["supplier_lead_time_days"] = default_lead_time_days

    result["supplier_lead_time_days"] = (
        result["supplier_lead_time_days"]
        .fillna(default_lead_time_days)
        .clip(lower=1)
        .astype(int)
    )

    result = calculate_safety_stock(result, safety_stock_pct=safety_stock_pct)

    result = calculate_recommended_purchase(result)

    result = classify_stock_status(result, overstock_multiplier=overstock_multiplier)

    result = calculate_priority_score(result)

    status_order = ["critical", "warning", "healthy", "overstock"]

    result["stock_status"] = pd.Categorical(
        result["stock_status"], categories=status_order, ordered=True
    )

    result = result.sort_values(
        by=["stock_status", "priority_score", "recommended_purchase_quantity"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    final_columns = [
        "product_name",
        "forecast_horizon_days",
        "forecasted_demand_raw",
        "forecasted_demand_non_negative",
        "forecasted_demand_units",
        "current_stock",
        "safety_stock",
        "required_stock",
        "recommended_purchase_quantity",
        "supplier_lead_time_days",
        "stock_status",
        "priority_score",
    ]

    return result[final_columns]
