"""Import processed CSV files into the configured SQLite database."""

from __future__ import annotations

import json
import sys

from api.config import load_settings
from api.repositories.sqlite_import import import_csv_files


def main() -> int:
    settings = load_settings()
    summary = import_csv_files(settings.database_path)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
