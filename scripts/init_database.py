"""Initialize the configured MotoStock SQLite database."""

from __future__ import annotations

import json
import sys

from api.config import load_settings
from api.database import initialize_database


def main() -> int:
    settings = load_settings()
    version = initialize_database(settings.database_path)
    print(
        json.dumps(
            {
                "database_path": str(settings.database_path),
                "schema_version": version,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
