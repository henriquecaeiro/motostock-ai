"""Prepare ignored serving CSVs from the tracked raw dataset."""

from __future__ import annotations

import json
import sys

from api.config import DAILY_SALES_PATH, MODELING_DATA_PATH, PROJECT_ROOT
from src.data_preparation import prepare_processed_data


def main() -> int:
    summary = prepare_processed_data(
        PROJECT_ROOT / "data" / "raw" / "motoretail.csv",
        DAILY_SALES_PATH,
        MODELING_DATA_PATH,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
