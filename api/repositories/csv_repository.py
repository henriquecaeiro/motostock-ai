"""Read processed datasets from CSV files."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from api.config import DAILY_SALES_PATH, MODELING_DATA_PATH
from api.exceptions import ProductNotFoundError, ServiceUnavailableError

logger = logging.getLogger(__name__)


class CsvRepository:
    """Simple CSV-backed repository for modeling and sales data."""

    def __init__(
        self,
        modeling_data_path=MODELING_DATA_PATH,
        daily_sales_path=DAILY_SALES_PATH,
    ) -> None:
        self.modeling_data_path = modeling_data_path
        self.daily_sales_path = daily_sales_path
        self._modeling_data: pd.DataFrame | None = None
        self._daily_sales: pd.DataFrame | None = None

    def _read_csv(self, path, parse_dates: list[str] | None = None) -> pd.DataFrame:
        if not path.exists():
            logger.error("Required CSV file is missing: %s", path)
            raise ServiceUnavailableError(
                f"Required data file is unavailable: {path.name}"
            )

        try:
            return pd.read_csv(path, parse_dates=parse_dates or [])
        except Exception as exc:
            logger.exception("Failed to read CSV file: %s", path)
            raise ServiceUnavailableError(
                f"Failed to load data file: {path.name}"
            ) from exc

    def load_modeling_data(self) -> pd.DataFrame:
        """Load the modeling dataset."""

        if self._modeling_data is None:
            self._modeling_data = self._read_csv(
                self.modeling_data_path,
                parse_dates=["sale_date"],
            )
        return self._modeling_data.copy()

    def load_daily_sales(self) -> pd.DataFrame:
        """Load the daily product sales dataset."""

        if self._daily_sales is None:
            self._daily_sales = self._read_csv(
                self.daily_sales_path,
                parse_dates=["sale_date"],
            )
        return self._daily_sales.copy()

    def list_products(self) -> list[str]:
        """Return unique product names sorted alphabetically."""

        daily_sales = self.load_daily_sales()
        return sorted(daily_sales["product_name"].dropna().unique().tolist())

    def product_exists(self, product_name: str) -> bool:
        """Check whether a product exists in the dataset."""

        return product_name in set(self.list_products())

    def get_product_history(self, product_name: str) -> pd.DataFrame:
        """Return historical daily sales for one product."""

        if not self.product_exists(product_name):
            raise ProductNotFoundError(product_name)

        daily_sales = self.load_daily_sales()
        product_history = daily_sales[
            daily_sales["product_name"] == product_name
        ].sort_values("sale_date")

        if product_history.empty:
            raise ProductNotFoundError(product_name)

        return product_history.reset_index(drop=True)

    @staticmethod
    def _latest_valid_value(series: pd.Series) -> Any:
        """Return the latest non-null value from a time-ordered series."""

        valid_values = series.dropna()
        if valid_values.empty:
            return None
        return valid_values.iloc[-1]

    def get_latest_business_state(self) -> pd.DataFrame:
        """Return the latest valid stock and lead time per product."""

        daily_sales = self.load_daily_sales()
        rows: list[dict[str, Any]] = []

        for product_name, product_history in daily_sales.groupby("product_name"):
            product_history = product_history.sort_values("sale_date")
            rows.append(
                {
                    "product_name": product_name,
                    "current_stock": self._latest_valid_value(
                        product_history["current_stock_snapshot"]
                    ),
                    "supplier_lead_time_days": self._latest_valid_value(
                        product_history["supplier_lead_time_days"]
                    ),
                }
            )

        return pd.DataFrame(rows)

    def get_latest_prices(self) -> pd.Series:
        """Return the latest valid unit price for each product."""

        daily_sales = self.load_daily_sales()
        latest_prices: dict[str, float] = {}

        for product_name, product_history in daily_sales.groupby("product_name"):
            product_history = product_history.sort_values("sale_date")
            price_series = product_history["unit_price_brl"].dropna()
            if price_series.empty:
                continue
            latest_prices[product_name] = float(price_series.iloc[-1])

        return pd.Series(latest_prices, name="unit_price_brl")

    def get_last_historical_date(self) -> pd.Timestamp:
        """Return the latest sale date in the daily sales dataset."""

        daily_sales = self.load_daily_sales()
        return pd.Timestamp(daily_sales["sale_date"].max())

    def save_recommendations(self, payload: dict[str, Any]) -> None:
        """Keep the CSV backend read-only; SQLite persists operational results."""

        return None

    def get_latest_recommendations(self, horizon_days: int | None = None):
        """CSV has no persisted recommendation history."""

        return None

    def insert_sales(self, records):
        """Prevent accidental writes when the read-only CSV backend is selected."""

        raise ServiceUnavailableError(
            "Sales ingestion requires DATA_BACKEND=sqlite."
        )
