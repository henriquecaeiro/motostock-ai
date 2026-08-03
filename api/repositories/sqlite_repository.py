"""SQLite-backed implementation of the application data repository."""

from __future__ import annotations

import logging
import hashlib
import json
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from api.config import MODEL_PATH
from api.database import (
    DatabaseError,
    connect_database,
    initialize_database,
)
from api.exceptions import (
    ModelVersionNotFoundError,
    ProductNotFoundError,
    ServiceUnavailableError,
)
from src.training.gates import check_promotion_gate

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
                CASE
                    WHEN SUM(s.quantity_sold) > 0
                        THEN SUM(s.quantity_sold * s.unit_price_brl)
                            / NULLIF(SUM(s.quantity_sold), 0)
                    ELSE MAX(NULLIF(s.unit_price_brl, 0))
                END AS unit_price_brl,
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
                SELECT product_id, MAX(observed_at) AS latest_observed_at
                FROM inventory
                GROUP BY product_id
            ) AS latest
                ON latest.product_id = i.product_id
               AND latest.latest_observed_at = i.observed_at
            ORDER BY i.inventory_id DESC
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
                                as_of_date, observed_at, external_id, idempotency_key,
                                source_key, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                int(product[0]),
                                current_stock,
                                max(1, lead_time),
                                sale_date,
                                f"{sale_date}T00:00:00+00:00",
                                external_id,
                                idempotency_key,
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

    def insert_inventory_snapshots(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Persist timestamped stock observations without replacing newer data.

        Every accepted event is retained for auditability.  The recommendation
        engine selects the newest observation by timestamp, so a delayed event
        can never move the current stock backwards.  Stable external or
        idempotency keys make retries safe across process restarts.
        """

        if not records:
            return {
                "inserted": 0,
                "skipped": 0,
                "updated": 0,
                "snapshot_ids": [],
            }

        inserted_ids: list[int] = []
        skipped = 0
        updated = 0

        try:
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

                for record in records:
                    product_name = str(record["product_name"]).strip()
                    product = connection.execute(
                        "SELECT product_id FROM products WHERE product_name = ?",
                        (product_name,),
                    ).fetchone()
                    if product is None:
                        raise ProductNotFoundError(product_name)

                    quantity = float(record["quantity_on_hand"])
                    lead_time = int(record["supplier_lead_time_days"])
                    if quantity < 0:
                        raise ValueError("quantity_on_hand cannot be negative.")
                    if lead_time < 1:
                        raise ValueError("supplier_lead_time_days must be positive.")

                    event_key = str(
                        record.get("idempotency_key")
                        or record.get("external_id")
                        or ""
                    ).strip()
                    if not event_key:
                        raise ValueError("Provide external_id or idempotency_key.")

                    observed_at = _normalize_observed_at(record.get("observed_at"))
                    source_key = f"api:inventory:{event_key}"
                    duplicate = connection.execute(
                        """
                        SELECT inventory_id
                        FROM inventory
                        WHERE source_key = ?
                           OR (
                               ? IS NOT NULL
                               AND external_id = ?
                           )
                           OR (
                               ? IS NOT NULL
                               AND idempotency_key = ?
                           )
                        LIMIT 1
                        """,
                        (
                            source_key,
                            record.get("external_id"),
                            record.get("external_id"),
                            record.get("idempotency_key"),
                            record.get("idempotency_key"),
                        ),
                    ).fetchone()
                    if duplicate is not None:
                        skipped += 1
                        continue

                    latest = connection.execute(
                        """
                        SELECT observed_at
                        FROM inventory
                        WHERE product_id = ?
                        ORDER BY observed_at DESC, inventory_id DESC
                        LIMIT 1
                        """,
                        (int(product[0]),),
                    ).fetchone()

                    cursor = connection.execute(
                        """
                        INSERT INTO inventory(
                            product_id, quantity_on_hand, supplier_lead_time_days,
                            as_of_date, observed_at, external_id, idempotency_key,
                            source_key, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(product[0]),
                            quantity,
                            lead_time,
                            observed_at[:10],
                            observed_at,
                            record.get("external_id"),
                            record.get("idempotency_key"),
                            source_key,
                            timestamp,
                            timestamp,
                        ),
                    )
                    snapshot_id = int(cursor.lastrowid)
                    inserted_ids.append(snapshot_id)
                    if latest is None or observed_at >= str(latest["observed_at"]):
                        updated += 1

                connection.commit()
        except ProductNotFoundError:
            raise
        except ValueError:
            raise
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Could not insert inventory snapshots")
            raise ServiceUnavailableError(
                "Could not write inventory snapshots to SQLite."
            ) from exc

        return {
            "inserted": len(inserted_ids),
            "skipped": skipped,
            "updated": updated,
            "snapshot_ids": inserted_ids,
        }

    def rebuild_modeling_data(self) -> Mapping[str, Any]:
        """Rebuild lag and rolling features from the current daily sales table."""

        daily = self.load_daily_sales()
        if daily.empty:
            raise ServiceUnavailableError("No sales are available for feature refresh.")

        records: list[dict[str, Any]] = []
        for product_name, group in daily.groupby("product_name"):
            history = group.copy()
            history["sale_date"] = pd.to_datetime(history["sale_date"])
            history = history.sort_values("sale_date").set_index("sale_date")
            full_dates = pd.date_range(history.index.min(), history.index.max(), freq="D")

            quantity = history["quantity_sold"].reindex(full_dates).fillna(0.0)
            price = history["unit_price_brl"].reindex(full_dates)
            previous_quantity = quantity.shift(1)
            previous_price = price.shift(1)
            feature_frame = pd.DataFrame(
                {
                    "quantity_sold": quantity,
                    "unit_price_1d": previous_price,
                    "quantity_1d": previous_quantity,
                    "quantity_7d": quantity.shift(7),
                    "quantity_14d": quantity.shift(14),
                    "rolling_1d": previous_quantity.rolling(1).mean(),
                    "rolling_7d": previous_quantity.rolling(7).mean(),
                    "rolling_14d": previous_quantity.rolling(14).mean(),
                },
                index=full_dates,
            )
            feature_frame = feature_frame.dropna()

            product = self._fetch_one(
                "SELECT product_id FROM products WHERE product_name = ?",
                (str(product_name),),
            )
            if product is None:
                raise ProductNotFoundError(str(product_name))

            for sale_date, row in feature_frame.iterrows():
                records.append(
                    {
                        "product_id": int(product["product_id"]),
                        "sale_date": pd.Timestamp(sale_date).date().isoformat(),
                        "quantity_sold": float(row["quantity_sold"]),
                        "unit_price_1d": float(row["unit_price_1d"]),
                        "day_of_week": int(sale_date.dayofweek),
                        "day_of_month": int(sale_date.day),
                        "month": int(sale_date.month),
                        "week_of_year": int(sale_date.isocalendar().week),
                        "is_weekend": int(sale_date.dayofweek in (5, 6)),
                        "quantity_1d": float(row["quantity_1d"]),
                        "quantity_7d": float(row["quantity_7d"]),
                        "quantity_14d": float(row["quantity_14d"]),
                        "rolling_1d": float(row["rolling_1d"]),
                        "rolling_7d": float(row["rolling_7d"]),
                        "rolling_14d": float(row["rolling_14d"]),
                    }
                )

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM modeling_data")
                connection.executemany(
                    """
                    INSERT INTO modeling_data(
                        product_id, sale_date, quantity_sold, unit_price_1d,
                        day_of_week, day_of_month, month, week_of_year, is_weekend,
                        quantity_1d, quantity_7d, quantity_14d, rolling_1d,
                        rolling_7d, rolling_14d, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            record["product_id"],
                            record["sale_date"],
                            record["quantity_sold"],
                            record["unit_price_1d"],
                            record["day_of_week"],
                            record["day_of_month"],
                            record["month"],
                            record["week_of_year"],
                            record["is_weekend"],
                            record["quantity_1d"],
                            record["quantity_7d"],
                            record["quantity_14d"],
                            record["rolling_1d"],
                            record["rolling_7d"],
                            record["rolling_14d"],
                            timestamp,
                            timestamp,
                        )
                        for record in records
                    ],
                )
                connection.commit()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Could not rebuild modeling data")
            raise ServiceUnavailableError("Could not rebuild modeling features.") from exc

        dates = [record["sale_date"] for record in records]
        return {
            "rows": len(records),
            "products": int(daily["product_name"].nunique()),
            "start_date": min(dates) if dates else None,
            "end_date": max(dates) if dates else None,
        }

    def record_application_run(
        self,
        *,
        run_key: str,
        job_name: str,
        status: str,
        started_at: str,
        finished_at: str | None = None,
        details: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Upsert one idempotent application execution record."""

        details_json = json.dumps(details or {}, ensure_ascii=False, default=str)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO application_runs(
                        run_key, job_name, status, started_at, finished_at,
                        details_json, error_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_key) DO UPDATE SET
                        job_name = excluded.job_name,
                        status = excluded.status,
                        started_at = excluded.started_at,
                        finished_at = excluded.finished_at,
                        details_json = excluded.details_json,
                        error_message = excluded.error_message,
                        updated_at = excluded.updated_at
                    """,
                    (
                        run_key,
                        job_name,
                        status,
                        started_at,
                        finished_at,
                        details_json,
                        error_message,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.commit()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Could not persist application run")
            raise ServiceUnavailableError("Could not persist application run.") from exc

    def get_application_run(self, run_key: str) -> Mapping[str, Any] | None:
        """Read a refresh execution by its deterministic key."""

        row = self._fetch_one(
            "SELECT * FROM application_runs WHERE run_key = ?",
            (run_key,),
        )
        if row is None:
            return None
        result = dict(row)
        result["details"] = json.loads(result.pop("details_json") or "{}")
        return result

    def register_model_version(self, metadata: Mapping[str, Any]) -> None:
        """Persist candidate metadata without changing the active artifact."""

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        status = str(metadata.get("status", "candidate"))
        if status not in {"candidate", "production", "active", "rejected", "archived"}:
            raise ValueError(f"Unsupported model status: {status}")

        try:
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO model_versions(
                        version, model_name, artifact_path, artifact_sha256,
                        backup_artifact_path, training_started_at, training_finished_at,
                        data_start_date, data_end_date, feature_columns_json,
                        parameters_json, metrics_json, seed, parent_version, status,
                        promoted_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(version) DO UPDATE SET
                        model_name = excluded.model_name,
                        artifact_path = excluded.artifact_path,
                        artifact_sha256 = excluded.artifact_sha256,
                        training_started_at = excluded.training_started_at,
                        training_finished_at = excluded.training_finished_at,
                        data_start_date = excluded.data_start_date,
                        data_end_date = excluded.data_end_date,
                        feature_columns_json = excluded.feature_columns_json,
                        parameters_json = excluded.parameters_json,
                        metrics_json = excluded.metrics_json,
                        seed = excluded.seed,
                        status = excluded.status,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(metadata["version"]),
                        str(metadata["model_name"]),
                        str(metadata["artifact_path"]),
                        str(metadata.get("artifact_sha256", "")),
                        metadata.get("backup_artifact_path"),
                        metadata.get("training_started_at"),
                        metadata.get("trained_at") or metadata.get("training_finished_at"),
                        metadata.get("data_start_date"),
                        metadata.get("data_end_date"),
                        json.dumps(metadata.get("feature_columns", []), ensure_ascii=False),
                        json.dumps(metadata.get("parameters", {}), ensure_ascii=False, default=str),
                        json.dumps(metadata.get("metrics", {}), ensure_ascii=False, default=str),
                        int(metadata.get("seed", 42)),
                        metadata.get("parent_version"),
                        status,
                        metadata.get("promoted_at"),
                        now,
                        now,
                    ),
                )
                connection.commit()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Could not register model version")
            raise ServiceUnavailableError("Could not persist model metadata.") from exc

    def list_model_versions(self) -> list[Mapping[str, Any]]:
        """List model metadata newest first."""

        try:
            with connect_database(self.database_path) as connection:
                rows = connection.execute(
                    "SELECT * FROM model_versions ORDER BY created_at DESC, model_version_id DESC"
                ).fetchall()
        except DatabaseError as exc:
            logger.exception("Could not list model versions")
            raise ServiceUnavailableError("Model registry is unavailable.") from exc

        return [_model_row_to_dict(row) for row in rows]

    def get_model_version(self, version: str) -> Mapping[str, Any] | None:
        """Read one model version by its exact version string."""

        row = self._fetch_one(
            "SELECT * FROM model_versions WHERE version = ?",
            (version,),
        )
        return None if row is None else _model_row_to_dict(row)

    def promote_model_version(
        self,
        version: str,
        *,
        production_path: str | Path = MODEL_PATH,
    ) -> Mapping[str, Any]:
        """Explicitly copy a candidate to production and update its registry state."""

        candidate = self.get_model_version(version)
        if candidate is None:
            raise ModelVersionNotFoundError(version)
        if candidate["status"] not in {"candidate", "rejected"}:
            raise ValueError("Only candidate or rejected versions can be promoted.")

        gate = check_promotion_gate(dict(candidate))
        if not gate["passed"]:
            raise ValueError(f"Candidate failed promotion gate: {gate}")

        candidate_path = Path(str(candidate["artifact_path"]))
        if not candidate_path.exists():
            raise ServiceUnavailableError("Candidate artifact is unavailable.")
        checksum = _sha256_path(candidate_path)
        if candidate.get("artifact_sha256") and checksum != candidate["artifact_sha256"]:
            raise ServiceUnavailableError("Candidate artifact checksum does not match metadata.")

        try:
            import joblib

            loaded = joblib.load(candidate_path)
            if not hasattr(loaded, "predict"):
                raise ValueError("Candidate artifact does not expose predict().")
        except Exception as exc:
            raise ServiceUnavailableError("Candidate artifact failed smoke loading.") from exc

        production = Path(production_path)
        production.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = production.parent / "backups" / f"{timestamp}_{production.name}"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if production.exists():
            shutil.copy2(production, backup_path)

        current = self._fetch_one(
            """
            SELECT version FROM model_versions
            WHERE status IN ('production', 'active')
            ORDER BY promoted_at DESC, model_version_id DESC
            LIMIT 1
            """
        )
        parent_version = str(current["version"]) if current is not None else None

        if parent_version is None and backup_path.exists():
            legacy_version = f"legacy-production-{timestamp}"
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                with connect_database(self.database_path) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        """
                        INSERT INTO model_versions(
                            version, model_name, artifact_path, artifact_sha256,
                            feature_columns_json, parameters_json, metrics_json,
                            seed, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, '[]', '{}', '{}', 42, 'archived', ?, ?)
                        """,
                        (
                            legacy_version,
                            str(candidate["model_name"]),
                            str(backup_path.resolve()),
                            _sha256_path(backup_path),
                            now,
                            now,
                        ),
                    )
                    connection.commit()
                parent_version = legacy_version
            except (DatabaseError, sqlite3.Error) as exc:
                logger.exception("Could not register legacy production model")
                raise ServiceUnavailableError("Could not prepare model rollback metadata.") from exc

        try:
            shutil.copy2(candidate_path, production)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE model_versions SET status = 'archived', updated_at = ? WHERE status IN ('production', 'active')",
                    (now,),
                )
                connection.execute(
                    """
                    UPDATE model_versions
                    SET status = 'production', parent_version = ?,
                        backup_artifact_path = ?, promoted_at = ?, updated_at = ?
                    WHERE version = ?
                    """,
                    (
                        parent_version,
                        str(backup_path.resolve()) if backup_path.exists() else None,
                        now,
                        now,
                        version,
                    ),
                )
                connection.commit()
        except (DatabaseError, sqlite3.Error, OSError) as exc:
            if backup_path.exists():
                shutil.copy2(backup_path, production)
            logger.exception("Could not promote model version")
            raise ServiceUnavailableError("Could not promote model version.") from exc

        return {
            "version": version,
            "status": "production",
            "production_path": str(production.resolve()),
            "backup_path": str(backup_path.resolve()) if backup_path.exists() else None,
            "parent_version": parent_version,
            "gate": gate,
        }

    def rollback_model_version(
        self,
        version: str,
        *,
        production_path: str | Path = MODEL_PATH,
    ) -> Mapping[str, Any]:
        """Restore the parent artifact of one explicitly selected production version."""

        current = self.get_model_version(version)
        if current is None:
            raise ModelVersionNotFoundError(version)
        if current["status"] not in {"production", "active"}:
            raise ValueError("Only the production version can be rolled back.")

        target = None
        parent_version = current.get("parent_version")
        if parent_version:
            parent = self.get_model_version(str(parent_version))
            if parent and Path(str(parent["artifact_path"])).exists():
                target = Path(str(parent["artifact_path"]))
        if target is None and current.get("backup_artifact_path"):
            backup = Path(str(current["backup_artifact_path"]))
            if backup.exists():
                target = backup
        if target is None:
            raise ServiceUnavailableError("No rollback artifact is available.")

        production = Path(production_path)
        shutil.copy2(target, production)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with connect_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE model_versions SET status = 'archived', updated_at = ? WHERE version = ?",
                    (now, version),
                )
                if parent_version:
                    connection.execute(
                        "UPDATE model_versions SET status = 'production', promoted_at = ?, updated_at = ? WHERE version = ?",
                        (now, now, parent_version),
                    )
                connection.commit()
        except (DatabaseError, sqlite3.Error) as exc:
            logger.exception("Could not persist model rollback")
            raise ServiceUnavailableError("Could not persist model rollback.") from exc

        return {
            "version": version,
            "status": "rolled_back",
            "restored_version": parent_version,
            "production_path": str(production.resolve()),
        }


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _normalize_observed_at(value: Any) -> str:
    """Normalize a timezone-aware event timestamp for lexicographic SQLite order."""

    if isinstance(value, datetime):
        observed = value
    elif value is not None:
        try:
            observed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("observed_at must be a valid ISO-8601 timestamp.") from exc
    else:
        raise ValueError("observed_at is required.")

    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("observed_at must include a timezone offset.")

    return observed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for column in ("feature_columns_json", "parameters_json", "metrics_json"):
        key = column.removesuffix("_json")
        try:
            result[key] = json.loads(result.pop(column) or ("[]" if key == "feature_columns" else "{}"))
        except json.JSONDecodeError:
            result[key] = [] if key == "feature_columns" else {}
    return result


SQLiteRepository = SqliteRepository
