"""Small, versioned SQLite database foundation for MotoStock AI."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 3


class DatabaseError(RuntimeError):
    """Raised when the SQLite database cannot be opened or migrated."""


def utc_now_iso() -> str:
    """Return a stable UTC timestamp suitable for SQLite text columns."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _schema_v1(connection: sqlite3.Connection) -> None:
    """Create the initial normalized-but-small application schema."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL UNIQUE,
            product_category TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id INTEGER PRIMARY KEY AUTOINCREMENT,
            supplier_name TEXT NOT NULL UNIQUE,
            lead_time_days INTEGER CHECK (lead_time_days IS NULL OR lead_time_days >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sales (
            sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE RESTRICT,
            sale_date TEXT NOT NULL,
            quantity_sold REAL NOT NULL CHECK (quantity_sold >= 0),
            unit_price_brl REAL NOT NULL CHECK (unit_price_brl >= 0),
            total_revenue_brl REAL,
            estimated_profit_brl REAL,
            discount_pct REAL CHECK (
                discount_pct IS NULL OR (discount_pct >= 0 AND discount_pct <= 100)
            ),
            current_stock_snapshot REAL CHECK (
                current_stock_snapshot IS NULL OR current_stock_snapshot >= 0
            ),
            supplier_lead_time_days INTEGER CHECK (
                supplier_lead_time_days IS NULL OR supplier_lead_time_days >= 1
            ),
            weather_condition TEXT,
            external_id TEXT,
            idempotency_key TEXT,
            source_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS inventory (
            inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE RESTRICT,
            quantity_on_hand REAL NOT NULL CHECK (quantity_on_hand >= 0),
            supplier_lead_time_days INTEGER NOT NULL DEFAULT 7 CHECK (supplier_lead_time_days >= 1),
            as_of_date TEXT NOT NULL,
            source_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (product_id, as_of_date)
        );

        CREATE TABLE IF NOT EXISTS modeling_data (
            modeling_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE RESTRICT,
            sale_date TEXT NOT NULL,
            quantity_sold REAL NOT NULL,
            unit_price_1d REAL,
            day_of_week INTEGER,
            day_of_month INTEGER,
            month INTEGER,
            week_of_year INTEGER,
            is_weekend INTEGER,
            quantity_1d REAL,
            quantity_7d REAL,
            quantity_14d REAL,
            rolling_1d REAL,
            rolling_7d REAL,
            rolling_14d REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (product_id, sale_date)
        );

        CREATE TABLE IF NOT EXISTS recommendation_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
            selected_model TEXT NOT NULL,
            horizon_days INTEGER NOT NULL CHECK (horizon_days BETWEEN 1 AND 30),
            started_at TEXT NOT NULL,
            finished_at TEXT,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS stock_recommendations (
            recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES recommendation_runs(run_id) ON DELETE CASCADE,
            product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE RESTRICT,
            product_name TEXT NOT NULL,
            forecast_horizon_days INTEGER NOT NULL CHECK (forecast_horizon_days BETWEEN 1 AND 30),
            forecasted_demand_raw REAL NOT NULL,
            forecasted_demand_non_negative REAL NOT NULL,
            forecasted_demand_units INTEGER NOT NULL CHECK (forecasted_demand_units >= 0),
            current_stock INTEGER NOT NULL CHECK (current_stock >= 0),
            safety_stock INTEGER NOT NULL CHECK (safety_stock >= 0),
            required_stock INTEGER NOT NULL CHECK (required_stock >= 0),
            recommended_purchase_quantity INTEGER NOT NULL CHECK (recommended_purchase_quantity >= 0),
            supplier_lead_time_days INTEGER NOT NULL CHECK (supplier_lead_time_days >= 1),
            stock_status TEXT NOT NULL,
            priority_score INTEGER NOT NULL CHECK (priority_score >= 0),
            created_at TEXT NOT NULL,
            UNIQUE (run_id, product_id)
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            model_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL UNIQUE,
            model_name TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            training_started_at TEXT,
            training_finished_at TEXT,
            data_start_date TEXT,
            data_end_date TEXT,
            feature_columns_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'rejected', 'archived')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sales_product_date
            ON sales(product_id, sale_date);
        CREATE INDEX IF NOT EXISTS idx_sales_date
            ON sales(sale_date);
        CREATE INDEX IF NOT EXISTS idx_inventory_product_date
            ON inventory(product_id, as_of_date DESC);
        CREATE INDEX IF NOT EXISTS idx_modeling_product_date
            ON modeling_data(product_id, sale_date);
        CREATE INDEX IF NOT EXISTS idx_recommendation_runs_created
            ON recommendation_runs(finished_at DESC);
        CREATE INDEX IF NOT EXISTS idx_stock_recommendations_run
            ON stock_recommendations(run_id);
        CREATE INDEX IF NOT EXISTS idx_model_versions_status
            ON model_versions(status, created_at DESC);
        """
    )


def _schema_v2(connection: sqlite3.Connection) -> None:
    """Relax historical sales to allow explicit zero-demand days.

    Operational insertion still rejects zero quantities; this migration only
    keeps complete daily history imported from the feature dataset.
    """

    connection.executescript(
        """
        CREATE TABLE sales_v2 (
            sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE RESTRICT,
            sale_date TEXT NOT NULL,
            quantity_sold REAL NOT NULL CHECK (quantity_sold >= 0),
            unit_price_brl REAL NOT NULL CHECK (unit_price_brl >= 0),
            total_revenue_brl REAL,
            estimated_profit_brl REAL,
            discount_pct REAL CHECK (
                discount_pct IS NULL OR (discount_pct >= 0 AND discount_pct <= 100)
            ),
            current_stock_snapshot REAL CHECK (
                current_stock_snapshot IS NULL OR current_stock_snapshot >= 0
            ),
            supplier_lead_time_days INTEGER CHECK (
                supplier_lead_time_days IS NULL OR supplier_lead_time_days >= 1
            ),
            weather_condition TEXT,
            external_id TEXT,
            idempotency_key TEXT,
            source_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        INSERT INTO sales_v2(
            sale_id, product_id, sale_date, quantity_sold, unit_price_brl,
            total_revenue_brl, estimated_profit_brl, discount_pct,
            current_stock_snapshot, supplier_lead_time_days, weather_condition,
            external_id, idempotency_key, source_key, created_at, updated_at
        )
        SELECT sale_id, product_id, sale_date, quantity_sold, unit_price_brl,
               total_revenue_brl, estimated_profit_brl, discount_pct,
               current_stock_snapshot, supplier_lead_time_days, weather_condition,
               external_id, idempotency_key, source_key, created_at, updated_at
        FROM sales;

        DROP TABLE sales;
        ALTER TABLE sales_v2 RENAME TO sales;

        CREATE INDEX IF NOT EXISTS idx_sales_product_date
            ON sales(product_id, sale_date);
        CREATE INDEX IF NOT EXISTS idx_sales_date
            ON sales(sale_date);
        """
    )


def _schema_v3(connection: sqlite3.Connection) -> None:
    """Store idempotent refresh/job executions separately from result rows."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS application_runs (
            application_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL UNIQUE,
            job_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
            started_at TEXT NOT NULL,
            finished_at TEXT,
            details_json TEXT NOT NULL DEFAULT '{}',
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_application_runs_job_status
            ON application_runs(job_name, status, updated_at DESC);
        """
    )


@contextmanager
def connect_database(database_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection with foreign keys and a bounded busy timeout."""

    path = str(database_path)

    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        connection = sqlite3.connect(path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not open SQLite database: {path}") from exc

    try:
        yield connection
    except sqlite3.Error as exc:
        raise DatabaseError(f"SQLite operation failed: {path}") from exc
    finally:
        connection.close()


def initialize_database(database_path: str | Path) -> int:
    """Create or migrate the database and return its schema version."""

    try:
        with connect_database(database_path) as connection:
            connection.execute("BEGIN")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )

            applied_versions = {
                int(row[0])
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }

            if 1 not in applied_versions:
                _schema_v1(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, utc_now_iso()),
                )

            if 2 not in applied_versions:
                _schema_v2(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (2, utc_now_iso()),
                )

            if 3 not in applied_versions:
                _schema_v3(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (3, utc_now_iso()),
                )

            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
            return SCHEMA_VERSION
    except DatabaseError:
        raise
    except sqlite3.Error as exc:
        raise DatabaseError("Could not initialize SQLite database.") from exc


def check_database_integrity(database_path: str | Path) -> bool:
    """Run SQLite's integrity check without mutating application data."""

    try:
        with connect_database(database_path) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            return bool(result and result[0] == "ok")
    except DatabaseError:
        return False
