"""Explicitly promote one registered candidate model."""

from __future__ import annotations

import argparse
import json
import sys

from api.config import load_settings
from api.repositories.factory import create_repository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    args = parser.parse_args()
    settings = load_settings()
    repository = create_repository(settings)
    if not hasattr(repository, "promote_model_version"):
        raise RuntimeError("Model promotion requires DATA_BACKEND=sqlite.")
    result = repository.promote_model_version(args.version)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
