"""Model training, evaluation and artifact metadata helpers."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.evaluation import error_by_product, evaluate_forecast
from src.training.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    temporal_split,
)


def build_estimator(model_name: str, *, random_state: int = 42) -> Pipeline:
    """Create the complete preprocessing + estimator pipeline."""

    normalized = model_name.lower().replace(" ", "_")
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if normalized in {"random_forest", "randomforest", "rf"}:
        estimator = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        normalized = "random_forest"
    elif normalized in {"xgboost", "xgb"}:
        estimator = XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        normalized = "xgboost"
    else:
        raise ValueError("model_name must be 'xgboost' or 'random_forest'.")

    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )
    pipeline.model_name = normalized
    return pipeline


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of an artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_predictions(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Return global, product-level and underprediction metrics."""

    clipped = np.clip(np.asarray(predictions, dtype=float), 0, None)
    result_frame = test[["sale_date", "product_name", TARGET_COLUMN]].copy()
    result_frame["prediction"] = clipped
    global_metrics = evaluate_forecast(result_frame[TARGET_COLUMN], clipped)

    product_metrics = [
        error_by_product(group, "prediction", model_name)
        for _, group in result_frame.groupby("product_name", sort=True)
    ]
    underprediction_flags = [
        row
        for row in product_metrics
        if row["signed_total_error"]
        < -max(5.0, float(row["total_actual_quantity"]) * 0.10)
    ]

    return {
        "global": global_metrics,
        "by_product": product_metrics,
        "underprediction_flags": underprediction_flags,
    }


def evaluate_artifact(
    artifact_path: str | Path,
    modeling_data: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
    production_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a saved artifact on the deterministic temporal holdout."""

    split = temporal_split(modeling_data, test_fraction=test_fraction)
    model = joblib.load(artifact_path)
    x_test = split.test[FEATURE_COLUMNS]
    y_test = split.test[TARGET_COLUMN].to_numpy()
    candidate_predictions = model.predict(x_test)
    metrics: dict[str, Any] = {
        "candidate": evaluate_predictions(
            split.test,
            candidate_predictions,
            model_name="candidate",
        ),
        "baselines": {
            "last_value": evaluate_forecast(
                y_test,
                np.clip(split.test["quantity_1d"].to_numpy(), 0, None),
            ),
            "moving_average_7d": evaluate_forecast(
                y_test,
                np.clip(split.test["rolling_7d"].to_numpy(), 0, None),
            ),
        },
    }
    if production_path and Path(production_path).exists():
        try:
            production_model = joblib.load(production_path)
            metrics["production"] = evaluate_predictions(
                split.test,
                production_model.predict(x_test),
                model_name="production",
            )
        except Exception as exc:
            metrics["production"] = {"status": "unavailable", "error": str(exc)}
    else:
        metrics["production"] = {
            "status": "unavailable",
            "error": "Production artifact was not found.",
        }
    return {
        "metrics": metrics,
        "split": {
            "train_start": split.train_start,
            "train_end": split.train_end,
            "test_start": split.test_start,
            "test_end": split.test_end,
            "train_rows": len(split.train),
            "test_rows": len(split.test),
        },
    }


def train_candidate(
    modeling_data: pd.DataFrame,
    *,
    model_name: str,
    version: str,
    artifact_path: str | Path,
    production_path: str | Path | None = None,
    random_state: int = 42,
    test_fraction: float = 0.20,
) -> dict[str, Any]:
    """Fit, smoke-test, save and describe a candidate artifact."""

    np.random.seed(random_state)
    split = temporal_split(modeling_data, test_fraction=test_fraction)
    model = build_estimator(model_name, random_state=random_state)
    model.fit(split.train[FEATURE_COLUMNS], split.train[TARGET_COLUMN])

    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact)
    checksum = sha256_file(artifact)

    smoke_prediction = np.asarray(
        model.predict(split.test[FEATURE_COLUMNS].head(1)),
        dtype=float,
    )
    if smoke_prediction.size != 1 or not np.isfinite(smoke_prediction).all():
        raise RuntimeError("Candidate artifact failed its prediction smoke test.")

    evaluation = evaluate_artifact(
        artifact,
        modeling_data,
        test_fraction=test_fraction,
        production_path=production_path,
    )
    full_dates = pd.to_datetime(modeling_data["sale_date"])
    estimator_params = model.named_steps["model"].get_params()
    metadata = {
        "version": version,
        "model_name": getattr(model, "model_name", model_name),
        "status": "candidate",
        "seed": random_state,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_start_date": full_dates.min().date().isoformat(),
        "data_end_date": full_dates.max().date().isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "parameters": _json_safe(estimator_params),
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": checksum,
        "target": TARGET_COLUMN,
        "split": evaluation["split"],
        "metrics": evaluation["metrics"],
    }
    metadata_path = artifact.with_suffix(artifact.suffix + ".json")
    metadata["metadata_path"] = str(metadata_path.resolve())
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def _json_safe(value: Any) -> Any:
    """Convert numpy values and non-finite floats to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
