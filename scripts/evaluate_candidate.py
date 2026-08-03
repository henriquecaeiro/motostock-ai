"""Evaluate a saved model candidate on the deterministic temporal holdout."""

from __future__ import annotations

import argparse
import json
import sys

from api.config import MODEL_PATH, PROJECT_ROOT, load_settings
from api.repositories.factory import create_repository
from src.training.features import build_modeling_dataset
from src.training.pipeline import _json_safe, evaluate_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    settings = load_settings()
    repository = create_repository(settings)
    modeling_data = build_modeling_dataset(repository.load_daily_sales())
    artifact_path = PROJECT_ROOT / args.artifact
    result = evaluate_artifact(
        artifact_path,
        modeling_data,
        production_path=MODEL_PATH,
    )
    if args.output:
        (PROJECT_ROOT / args.output).write_text(
            json.dumps(_json_safe(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(_json_safe(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
