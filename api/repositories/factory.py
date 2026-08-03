"""Repository selection from validated application settings."""

from __future__ import annotations

import logging

from api.config import DAILY_SALES_PATH, MODELING_DATA_PATH, Settings
from api.repositories.csv_repository import CsvRepository
from api.repositories.sqlite_import import import_csv_files
from api.repositories.sqlite_repository import SqliteRepository

logger = logging.getLogger(__name__)


def create_repository(settings: Settings):
    """Create the configured repository and optionally seed an empty SQLite file."""

    if settings.data_backend == "sqlite":
        repository = SqliteRepository(settings.database_path)

        if (
            settings.auto_import_csv
            and not repository.list_products()
            and DAILY_SALES_PATH.exists()
            and MODELING_DATA_PATH.exists()
        ):
            logger.info("SQLite database is empty; importing processed CSV data")
            import_csv_files(settings.database_path)

        return repository

    return CsvRepository(
        modeling_data_path=MODELING_DATA_PATH,
        daily_sales_path=DAILY_SALES_PATH,
    )
