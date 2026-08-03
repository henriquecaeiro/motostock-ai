"""SQLite-backed implementation of the application data repository."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from api.database import (
    DatabaseError,
    connect_database,
    initialize_database,
)
from api.exceptions import ProductNotFoundError, ServiceUnavailableError

logger = logging.getLogger(__name__)


class SqliteRepository:
    """Repository that keeps operational data in a local SQLite file."""

    def __init__(self, database_path: str | Path, *, initialize: bool = True) -> None:
        self.database_path = Path(database_path)

        if initialize:
            try:
                initialize_database(self.database_path)
            except DatabaseError as exc:
                logger.exception("Could not initialize SQLite repository")
                raise ServiceUnavailableError(
                    f"SQLite database is unavailable: {self.database_path.name}"
                ) from exc

    def _read_frame(self, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        try:
            with connect_database(self.database_path) as connection:
                return pd.read_sql_query(
                    query,
                    connection,
                    params=params,
                    parse_dates=["sale_date"] if "sale_date" in query else None,
                )
        except DatabaseError as exc:
            logger.exception("SQLite read failed")
            raise ServiceUnavailableError("SQLite data is unavailable.") from exc

    def _fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        try:
            with connect_database(self.database_path) as connection:
                return connection.execute(query, params).fetchone()
        except DatabaseError as exc:
            logger.exception("SQLite lookup failed")
            raise ServiceUnavailableError("SQLite data is unavailable.") from exc

    def load_modeling_data(self) -> pd.DataFrame:
        """Load the feature dataset using the same column names as the CSV path."""

        frame = self._read_frame(
            """
            SELECT
                m.sale_date,
                m.quantity_sold,
                m.unit_price_1d,
                m.day_of_week,
                m.day_of_month,
                m.month,
                m.week_of_year,
                m.is_weekend,
                m.quantity_1d,
                m.quantity_7d,
                m.quantity_14d,
                m.rolling_1d,
                m.rolling_7d,
                m.rolling_14d,
                p.product_name
            FROM modeling_data AS m
            JOIN products AS p ON p.product_id = m.product_id
            ORDER BY m.sale_date, p.product_name
            """
        )
        return frame

    def load_daily_sales(self) -> pd.DataFrame:
        """Return one aggregated product/day row for forecasting and operations."""

        return self._read_frame(
            """
            SELECT
                p.product_name,
                s.sale_date,
                SUM(s.quantity_sold) AS quantity_sold,
                SUM(COALESCE(s.total_revenue_brl, s.quantity_sold * s.unit_price_brl))
                    AS total_revenue_brl,
                SUM(COALESCE(s.estimated_profit_brl, 0)) AS estimated_profit_brl,
                AVG(s.discount_pct) AS discount_pct,
                SUM(s.quantity_sold * s.unit_price_brl)
                    / NULLIF(SUM(s.quantity_sold), 0) AS unit_price_brl,
                MAX(s.current_stock_snapshot) AS current_stock_snapshot,
                MAX(s.supplier_lead_time_days) AS supplier_lead_time_days,
                p.product_category,
                MAX(s.weather_condition) AS weather_condition
            FROM sales AS s
            JOIN products AS p ON p.product_id = s.product_id
            GROUP BY p.product_id, p.product_name, p.product_category, s.sale_date
            ORDER BY s.sale_date, p.product_name
            """
        )

    def list_products(self) -> list[str]:
        """Return products in deterministic alphabetical order."""

        frame = self._read_frame(
            "SELECT product_name FROM products ORDER BY product_name"
        )
        return frame["product_name"].dropna().astype(str).tolist()

    def product_exists(self, product_name: str) -> bool:
        """Check a product by an exact, parameterized lookup."""

        return self._fetch_one(
            "SELECT 1 FROM products WHERE product_name = ? LIMIT 1",
            (product_name,),
        ) is not None

    def get_product_history(self, product_name: str) -> pd.DataFrame:
        """Return aggregated historical sales for one product."""

        if not self.product_exists(product_name):
            raise ProductNotFoundError(product_name)

        frame = self.load_daily_sales()
        history = frame[frame["product_name"] == product_name].sort_values("sale_date")
        if history.empty:
            raise ProductNotFoundError(product_name)
        return history.reset_index(drop=True)

    @staticmethod
    def _latest_valid_value(series: pd.Series) -> Any:
        values = series.dropna()
        return None if values.empty else values.iloc[-1]

    def get_latest_business_state(self) -> pd.DataFrame:
        """Return inventory snapshots, falling back to the latest sales snapshot."""

        sales = self.load_daily_sales()
        fallback: dict[str, dict[str, Any]] = {}
        for product_name, history in sales.groupby("product_name"):
            history = history.sort_values("sale_date")
            fallback[str(product_name)] = {
                "current_stock": self._latest_valid_value(
                    history["current_stock_snapshot"]
                ),
                "supplier_lead_time_days": self._latest_valid_value(
                    history["supplier_lead_time_days"]
                ),
            }

        inventory = self._read_frame(
            """
            SELECT p.product_name, i.quantity_on_hand, i.supplier_lead_time_days
            FROM inventory AS i
            JOIN products AS p ON p.product_id = i.product_id
            JOIN (
                SELECT product_id, MAX(as_of_date) AS latest_date
                FROM inventory
                GROUP BY product_id
            ) AS latest
                ON latest.product_id = i.product_id
               AND latest.latest_date = i.as_of_date
            """
        )
        latest_by_product = {
            str(row["product_name"]): row for _, row in inventory.iterrows()
        }

        products = self.list_products()
        rows: list[dict[str, Any]] = []
        for product_name in products:
            latest = latest_by_product.get(product_name)
            fallback_values = fallback.get(product_name, {})
            rows.append(
                {
                    "product_name": product_name,
                    "current_stock": (
                        latest["quantity_on_hand"]
                        if latest is not None
                        else fallback_values.get("current_stock")
                    ),
                    "supplier_lead_time_days": (
                        latest["supplier_lead_time_days"]
                        if latest is not None
                        else fallback_values.get("supplier_lead_time_days")
                    ),
                }
            )

        return pd.DataFrame(rows)

    def get_latest_prices(self) -> pd.Series:
        """Return quantity-weighted latest-day prices by product."""

        frame = self._read_frame(
            """
            SELECT p.product_name,
                   SUM(s.quantity_sold * s.unit_price_brl)
                       / NULLIF(SUM(s.quantity_sold), 0) AS unit_price_brl
            FROM sales AS s
            JOIN products AS p ON p.product_id = s.product_id
            JOIN (
                SELECT product_id, MAX(sale_date) AS latest_date
                FROM sales
                WHERE quantity_sold > 0 AND unit_price_brl > 0
                GROUP BY product_id
            ) AS latest
                ON latest.product_id = s.product_id
               AND latest.latest_date = s.sale_date
            GROUP BY p.product_name
            ORDER BY p.product_name
            """
        )
        return pd.Series(
            {
                str(row["product_name"]): float(row["unit_price_brl"])
                for _, row in frame.iterrows()
                if pd.notna(row["unit_price_brl"])
            },
            name="unit_price_brl",
        )

    def get_last_historical_date(self) -> pd.Timestamp:
        """Return the latest date in the sales table."""

        row = self._fetch_one("SELECT MAX(sale_date) AS latest_date FROM sales")
        if row is None or row["latest_date"] is None:
            raise ServiceUnavailableError("No historical sales are available.")
        return pd.Timestamp(row["latest_date"])

    def save_recommendations(self, payload: Mapping[str, Any]) -> None:
        """Upsert a deterministic recommendation run and replace its rows."""

        generated_at = payload.get("generated_at")
        if isinstance(generated_at, datetime):
            generated_at_text = generated_at.astimezone(timezone.utc).isoformat()
        else:
            generated_at_text = str(generated_at or datetime.now(timezone.utc).isoformat())

        selected_model = str(payload.get("selected_model", "unknown"))
        horizon_days = int(payload.get("horizon_days", 14))
        try:
            historical_date = self.get_last_historical_date().date().isoformat()
        except ServiceUnavailableError:
            historical_date = "empty"
        run_key = f"recommendations:{selected_model}:{horizon_days}:{historical_date}"

        try:
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO recommendation_runs(
                        run_key, status, selected_model, horizon_days,
                        started_at, finished_at, error_message
                    ) VALUES (?, 'completed', ?, ?, ?, ?, NULL)
                    ON CONFLICT(run_key) DO UPDATE SET
                        status = 'completed',
                        selected_model = excluded.selected_model,
                        horizon_days = excluded.horizon_days,
                        finished_at = excluded.finished_at,
                        error_message = NULL
                    """,
                    (
                        run_key,
                        selected_model,
                        horizon_days,
                        generated_at_text,
                        generated_at_text,
                    ),
                )
                run = connection.execute(
                    "SELECT run_id FROM recommendation_runs WHERE run_key = ?",
                    (run_key,),
                ).fetchone()
                if run is None:
                    raise RuntimeError("Could not resolve recommendation run")
                run_id = int(run[0])
                connection.execute(
                    "DELETE FROM stock_recommendations WHERE run_id = ?",
                    (run_id,),
                )

                for item in payload.get("recommendations", []):
                    product_name = str(item["product_name"])
                    product = connection.execute(
                        "SELECT product_id FROM products WHERE product_name = ?",
                        (product_name,),
                    ).fetchone()
                    if product is None:
                        raise ProductNotFoundError(product_name)

                    connection.execute(
                        """
                        INSERT INTO stock_recommendations(
                            run_id, product_id, product_name, forecast_horizon_days,
                            forecasted_demand_raw, forecasted_demand_non_negative,
                            forecasted_demand_units, current_stock, safety_stock,
                            required_stock, recommended_purchase_quantity,
                            supplier_lead_time_days, stock_status, priority_score,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            int(product[0]),
                            product_name,
                            int(item["forecast_horizon_days"]),
                            float(item["forecasted_demand_raw"]),
                            float(item["forecasted_demand_non_negative"]),
                            int(item["forecasted_demand_units"]),
                            int(item["current_stock"]),
                            int(item["safety_stock"]),
                            int(item["required_stock"]),
                            int(item["recommended_purchase_quantity"]),
                            int(item["supplier_lead_time_days"]),
                            str(item["stock_status"]),
                            int(item["priority_score"]),
                            generated_at_text,
                        ),
                    )

                connection.commit()
        except ProductNotFoundError:
            raise
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Could not persist recommendations")
            raise ServiceUnavailableError(
                "Could not persist stock recommendations."
            ) from exc

    def get_latest_recommendations(
        self, horizon_days: int | None = None
    ) -> Mapping[str, Any] | None:
        """Read the most recently completed persisted recommendation run."""

        horizon_filter = "AND horizon_days = ?" if horizon_days is not None else ""
        params: tuple[Any, ...] = (horizon_days,) if horizon_days is not None else ()
        try:
            with connect_database(self.database_path) as connection:
                run = connection.execute(
                    f"""
                    SELECT run_id, selected_model, horizon_days, finished_at
                    FROM recommendation_runs
                    WHERE status = 'completed' {horizon_filter}
                    ORDER BY finished_at DESC, run_id DESC
                    LIMIT 1
                    """,
                    params,
                ).fetchone()
                if run is None:
                    return None

                rows = connection.execute(
                    """
                    SELECT product_name, forecast_horizon_days,
                           forecasted_demand_raw, forecasted_demand_non_negative,
                           forecasted_demand_units, current_stock, safety_stock,
                           required_stock, recommended_purchase_quantity,
                           supplier_lead_time_days, stock_status, priority_score
                    FROM stock_recommendations
                    WHERE run_id = ?
                    ORDER BY recommendation_id
                    """,
                    (int(run["run_id"]),),
                ).fetchall()
        except DatabaseError as exc:
            logger.exception("Could not read persisted recommendations")
            raise ServiceUnavailableError(
                "Persisted recommendations are unavailable."
            ) from exc

        recommendations = [dict(row) for row in rows]
        return {
            "generated_at": run["finished_at"],
            "selected_model": run["selected_model"],
            "horizon_days": int(run["horizon_days"]),
            "count": len(recommendations),
            "recommendations": recommendations,
        }

    def insert_sales(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Insert sales in one transaction, skipping known source keys."""

        if not records:
            return {"inserted": 0, "skipped": 0, "sale_ids": []}

        inserted_ids: list[int] = []
        skipped = 0

        try:
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

                for record in records:
                    product_name = str(record["product_name"])
                    product = connection.execute(
                        "SELECT product_id FROM products WHERE product_name = ?",
                        (product_name,),
                    ).fetchone()
                    if product is None:
                        raise ProductNotFoundError(product_name)

                    sale_date = pd.Timestamp(record["sale_date"]).date().isoformat()
                    quantity = float(record["quantity_sold"])
                    unit_price = float(record["unit_price_brl"])
                    if quantity <= 0 or unit_price < 0:
                        raise ValueError("quantity_sold must be positive and price non-negative")

                    external_id = record.get("external_id")
                    idempotency_key = record.get("idempotency_key")
                    source_value = idempotency_key or external_id or str(uuid4())
                    source_key = f"api:{source_value}"
                    cursor = connection.execute(
                        """
                        INSERT INTO sales(
                            product_id, sale_date, quantity_sold, unit_price_brl,
                            total_revenue_brl, estimated_profit_brl, discount_pct,
                            current_stock_snapshot, supplier_lead_time_days,
                            weather_condition, external_id, idempotency_key,
                            source_key, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_key) DO NOTHING
                        """,
                        (
                            int(product[0]),
                            sale_date,
                            quantity,
                            unit_price,
                            _optional_float(record.get("total_revenue_brl")),
                            _optional_float(record.get("estimated_profit_brl")),
                            _optional_float(record.get("discount_pct")),
                            _optional_float(record.get("current_stock_snapshot")),
                            _optional_int(record.get("supplier_lead_time_days")),
                            record.get("weather_condition"),
                            external_id,
                            idempotency_key,
                            source_key,
                            timestamp,
                            timestamp,
                        ),
                    )
                    if cursor.rowcount == 0:
                        skipped += 1
                        continue

                    sale_id = int(cursor.lastrowid)
                    inserted_ids.append(sale_id)

                    current_stock = _optional_float(record.get("current_stock_snapshot"))
                    if current_stock is not None:
                        lead_time = _optional_int(record.get("supplier_lead_time_days")) or 7
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
                                int(product[0]),
                                current_stock,
                                max(1, lead_time),
                                sale_date,
                                f"{source_key}:inventory",
                                timestamp,
                                timestamp,
                            ),
                        )

                connection.commit()
        except ProductNotFoundError:
            raise
        except ValueError:
            raise
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Could not insert sales")
            raise ServiceUnavailableError("Could not write sales to SQLite.") from exc

        return {
            "inserted": len(inserted_ids),
            "skipped": skipped,
            "sale_ids": inserted_ids,
        }


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


SQLiteRepository = SqliteRepository
