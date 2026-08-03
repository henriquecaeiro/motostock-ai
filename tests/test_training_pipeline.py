"""Training, evaluation and model registry tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from api.repositories.sqlite_repository import SqliteRepository
from src.training.features import build_modeling_dataset, temporal_split
from src.training.gates import check_promotion_gate
from src.training.pipeline import train_candidate


def test_feature_builder_reproduces_the_current_processed_dataset() -> None:
    daily = pd.read_csv("data/processed/daily_product_sales.csv", parse_dates=["sale_date"])
    expected = pd.read_csv("data/processed/modeling_dataset.csv", parse_dates=["sale_date"])

    rebuilt = build_modeling_dataset(daily)

    assert rebuilt.shape == expected.shape
    assert rebuilt[["sale_date", "product_name"]].equals(
        expected[["sale_date", "product_name"]]
    )
    for column in [
        "quantity_sold",
        "unit_price_1d",
        "quantity_1d",
        "quantity_7d",
        "quantity_14d",
        "rolling_1d",
        "rolling_7d",
        "rolling_14d",
    ]:
        np.testing.assert_allclose(rebuilt[column], expected[column], rtol=0, atol=1e-9)


def test_temporal_split_is_date_disjoint() -> None:
    daily = pd.read_csv("data/processed/daily_product_sales.csv", parse_dates=["sale_date"])
    modeling = build_modeling_dataset(daily)
    split = temporal_split(modeling)

    assert split.train["sale_date"].max() < split.test["sale_date"].min()
    assert len(split.train) + len(split.test) == len(modeling)


def test_train_candidate_saves_complete_artifact_and_metrics(tmp_path: Path) -> None:
    dates = pd.date_range("2026-01-01", periods=45, freq="D")
    daily = pd.DataFrame(
        {
            "product_name": "Test Product",
            "sale_date": dates,
            "quantity_sold": np.arange(45) % 5,
            "unit_price_brl": 10.0,
        }
    )
    modeling = build_modeling_dataset(daily)
    artifact = tmp_path / "candidate.pkl"

    metadata = train_candidate(
        modeling,
        model_name="random_forest",
        version="candidate-test",
        artifact_path=artifact,
    )

    assert artifact.exists()
    assert Path(str(artifact) + ".json").exists()
    assert metadata["status"] == "candidate"
    assert metadata["split"]["train_rows"] > 0
    assert metadata["metrics"]["candidate"]["global"]["MAE"] >= 0
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert metadata["artifact_sha256"] == digest
    assert joblib.load(artifact).predict(modeling.iloc[:1][metadata["feature_columns"]]).size == 1


def test_model_registry_promotion_and_rollback_are_explicit(tmp_path: Path) -> None:
    database_path = tmp_path / "registry.db"
    candidate_path = tmp_path / "candidate.pkl"
    production_path = tmp_path / "production.pkl"
    previous_model = DummyRegressor(strategy="constant", constant=3)
    previous_model.fit(np.array([[0], [1]]), np.array([3, 3]))
    candidate_model = DummyRegressor(strategy="constant", constant=7)
    candidate_model.fit(np.array([[0], [1]]), np.array([7, 7]))
    joblib.dump(candidate_model, candidate_path)
    joblib.dump(previous_model, production_path)

    repository = SqliteRepository(database_path)
    repository.register_model_version(
        {
            "version": "candidate-registry-test",
            "model_name": "dummy",
            "artifact_path": str(candidate_path),
            "artifact_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "feature_columns": ["feature"],
            "parameters": {"constant": 7},
            "metrics": {"candidate": {"global": {"MAE": 0.0}}},
            "status": "candidate",
            "seed": 42,
        }
    )

    promoted = repository.promote_model_version(
        "candidate-registry-test",
        production_path=production_path,
    )
    rolled_back = repository.rollback_model_version(
        "candidate-registry-test",
        production_path=production_path,
    )

    assert promoted["status"] == "production"
    assert rolled_back["restored_version"].startswith("legacy-production-")
    assert repository.get_model_version("candidate-registry-test")["status"] == "archived"
    assert repository.list_model_versions()[0]["status"] == "production"


def test_promotion_gate_rejects_material_mae_regression() -> None:
    result = check_promotion_gate(
        {
            "metrics": {
                "candidate": {"global": {"MAE": 1.20}, "by_product": []},
                "production": {"global": {"MAE": 1.00}, "by_product": []},
            }
        }
    )

    assert result["passed"] is False
    assert result["checks"]["global_mae"]["passed"] is False
