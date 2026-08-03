"""Train and register a model candidate without changing production."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from api.config import MODEL_PATH, PROJECT_ROOT, load_settings
from api.repositories.factory import create_repository
from src.training.features import build_modeling_dataset
from src.training.pipeline import train_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest"])
    parser.add_argument("--version", default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    repository = create_repository(settings)
    modeling_data = build_modeling_dataset(repository.load_daily_sales())
    version = args.version or (
        f"candidate-{args.model}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    artifact_path = (
        PROJECT_ROOT / "artifacts" / "models" / "candidates" / f"{version}.pkl"
        if args.artifact is None
        else PROJECT_ROOT / args.artifact
    )
    metadata = train_candidate(
        modeling_data,
        model_name=args.model,
        version=version,
        artifact_path=artifact_path,
        production_path=MODEL_PATH,
        random_state=args.seed,
        test_fraction=args.test_fraction,
    )
    if hasattr(repository, "register_model_version"):
        repository.register_model_version(metadata)
    print(json.dumps(metadata, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
