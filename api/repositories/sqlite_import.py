"""Transactional import of the project's processed CSV datasets into SQLite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from api.config import DAILY_SALES_PATH, MODELING_DATA_PATH
from api.database import SCHEMA_VERSION, connect_database, initialize_database, utc_now_iso


DAILY_REQUIRED_COLUMNS = {
    "product_name",
    "sale_date",
    "quantity_sold",
    "unit_price_brl",
    "current_stock_snapshot",
    "supplier_lead_time_days",
}
MODELING_REQUIRED_COLUMNS = {
    "sale_date",
    "quantity_sold",
    "product_name",
}


def _as_optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _as_optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _as_date(value: Any) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("sale_date cannot be null")
    return parsed.date().isoformat()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _ensure_product(
    connection,
    product_name: str,
    product_category: str | None,
    timestamp: str,
) -> int:
    connection.execute(
        """
        INSERT INTO products(product_name, product_category, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(product_name) DO UPDATE SET
            product_category = COALESCE(excluded.product_category, products.product_category),
            updated_at = excluded.updated_at
        """,
        (product_name, product_category, timestamp, timestamp),
    )
    row = connection.execute(
        "SELECT product_id FROM products WHERE product_name = ?",
        (product_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Could not resolve imported product: {product_name}")
    return int(row[0])


def import_csv_files(
    database_path: str | Path,
    *,
    daily_sales_path: str | Path = DAILY_SALES_PATH,
    modeling_data_path: str | Path = MODELING_DATA_PATH,
) -> dict[str, Any]:
    """Import both processed CSVs using one transaction and stable source keys."""

    daily_path = Path(daily_sales_path)
    modeling_path = Path(modeling_data_path)

    if not daily_path.exists():
        raise FileNotFoundError(f"Daily sales CSV is unavailable: {daily_path}")
    if not modeling_path.exists():
        raise FileNotFoundError(f"Modeling CSV is unavailable: {modeling_path}")

    daily = pd.read_csv(daily_path)
    modeling = pd.read_csv(modeling_path)
    _require_columns(daily, DAILY_REQUIRED_COLUMNS, "daily_product_sales.csv")
    _require_columns(modeling, MODELING_REQUIRED_COLUMNS, "modeling_dataset.csv")

    initialize_database(database_path)
    timestamp = utc_now_iso()

    with connect_database(database_path) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")

            for row_index, row in daily.iterrows():
                product_name = str(row["product_name"]).strip()
                if not product_name:
                    raise ValueError(f"daily sales row {row_index} has an empty product")

                sale_date = _as_date(row["sale_date"])
                quantity = float(row["quantity_sold"])
                unit_price_value = row["unit_price_brl"]
                unit_price = (
                    0.0
                    if pd.isna(unit_price_value) and quantity == 0
                    else float(unit_price_value)
                )
                if quantity < 0:
                    raise ValueError(
                        f"daily sales row {row_index} has negative quantity_sold"
                    )
                if unit_price < 0:
                    raise ValueError(
                        f"daily sales row {row_index} has a negative unit_price_brl"
                    )

                product_id = _ensure_product(
                    connection,
                    product_name,
                    (
                        None
                        if pd.isna(row.get("product_category"))
                        else str(row.get("product_category")).strip()
                    ),
                    timestamp,
                )
                source_key = f"csv:daily:{product_name}:{sale_date}:{row_index}"

                connection.execute(
                    """
                    INSERT INTO sales(
                        product_id, sale_date, quantity_sold, unit_price_brl,
                        total_revenue_brl, estimated_profit_brl, discount_pct,
                        current_stock_snapshot, supplier_lead_time_days,
                        weather_condition, external_id, idempotency_key, source_key,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_key) DO UPDATE SET
                        product_id = excluded.product_id,
                        sale_date = excluded.sale_date,
                        quantity_sold = excluded.quantity_sold,
                        unit_price_brl = excluded.unit_price_brl,
                        total_revenue_brl = excluded.total_revenue_brl,
                        estimated_profit_brl = excluded.estimated_profit_brl,
                        discount_pct = excluded.discount_pct,
                        current_stock_snapshot = excluded.current_stock_snapshot,
                        supplier_lead_time_days = excluded.supplier_lead_time_days,
                        weather_condition = excluded.weather_condition,
                        updated_at = excluded.updated_at
                    """,
                    (
                        product_id,
                        sale_date,
                        quantity,
                        unit_price,
                        _as_optional_float(row.get("total_revenue_brl")),
                        _as_optional_float(row.get("estimated_profit_brl")),
                        _as_optional_float(row.get("discount_pct")),
                        _as_optional_float(row.get("current_stock_snapshot")),
                        (
                            None
                            if pd.isna(row.get("supplier_lead_time_days"))
                            else max(1, int(row.get("supplier_lead_time_days")))
                        ),
                        (
                            None
                            if pd.isna(row.get("weather_condition"))
                            else str(row.get("weather_condition"))
                        ),
                        source_key,
                        source_key,
                        source_key,
                        timestamp,
                        timestamp,
                    ),
                )

                lead_time = _as_optional_int(row["supplier_lead_time_days"])
                quantity_on_hand = _as_optional_float(row["current_stock_snapshot"])
                if quantity_on_hand is not None:
                    connection.execute(
                        """
                        INSERT INTO inventory(
                            product_id, quantity_on_hand, supplier_lead_time_days,
                            as_of_date, source_key, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(product_id, as_of_date) DO UPDATE SET
                            quantity_on_hand = excluded.quantity_on_hand,
                            supplier_lead_time_days = excluded.supplier_lead_time_days,
                            source_key = excluded.source_key,
                            updated_at = excluded.updated_at
                        """,
                        (
                            product_id,
                            quantity_on_hand,
                            max(1, lead_time or 7),
                            sale_date,
                            f"csv:inventory:{product_name}:{sale_date}",
                            timestamp,
                            timestamp,
                        ),
                    )

            for row_index, row in modeling.iterrows():
                product_name = str(row["product_name"]).strip()
                if not product_name:
                    raise ValueError(f"modeling row {row_index} has an empty product")

                product_id = _ensure_product(connection, product_name, None, timestamp)
                sale_date = _as_date(row["sale_date"])

                connection.execute(
                    """
                    INSERT INTO modeling_data(
                        product_id, sale_date, quantity_sold, unit_price_1d,
                        day_of_week, day_of_month, month, week_of_year, is_weekend,
                        quantity_1d, quantity_7d, quantity_14d, rolling_1d,
                        rolling_7d, rolling_14d, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(product_id, sale_date) DO UPDATE SET
                        quantity_sold = excluded.quantity_sold,
                        unit_price_1d = excluded.unit_price_1d,
                        day_of_week = excluded.day_of_week,
                        day_of_month = excluded.day_of_month,
                        month = excluded.month,
                        week_of_year = excluded.week_of_year,
                        is_weekend = excluded.is_weekend,
                        quantity_1d = excluded.quantity_1d,
                        quantity_7d = excluded.quantity_7d,
                        quantity_14d = excluded.quantity_14d,
                        rolling_1d = excluded.rolling_1d,
                        rolling_7d = excluded.rolling_7d,
                        rolling_14d = excluded.rolling_14d,
                        updated_at = excluded.updated_at
                    """,
                    (
                        product_id,
                        sale_date,
                        float(row["quantity_sold"]),
                        _as_optional_float(row.get("unit_price_1d")),
                        _as_optional_int(row.get("day_of_week")),
                        _as_optional_int(row.get("day_of_month")),
                        _as_optional_int(row.get("month")),
                        _as_optional_int(row.get("week_of_year")),
                        _as_optional_int(row.get("is_weekend")),
                        _as_optional_float(row.get("quantity_1d")),
                        _as_optional_float(row.get("quantity_7d")),
                        _as_optional_float(row.get("quantity_14d")),
                        _as_optional_float(row.get("rolling_1d")),
                        _as_optional_float(row.get("rolling_7d")),
                        _as_optional_float(row.get("rolling_14d")),
                        timestamp,
                        timestamp,
                    ),
                )

            connection.commit()
        except Exception:
            connection.rollback()
            raise

        counts = {
            "products": int(
                connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            ),
            "sales": int(connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0]),
            "inventory": int(
                connection.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
            ),
            "modeling_rows": int(
                connection.execute("SELECT COUNT(*) FROM modeling_data").fetchone()[0]
            ),
        }

    return {
        "database_path": str(Path(database_path).resolve()),
        "schema_version": SCHEMA_VERSION,
        **counts,
    }
