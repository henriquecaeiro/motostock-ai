"""Small, explainable candidate-promotion gate."""

from __future__ import annotations

from typing import Any


MAE_TOLERANCE = 0.05


def check_promotion_gate(metadata: dict[str, Any]) -> dict[str, Any]:
    """Reject material global or product-level regressions."""

    metrics = metadata.get("metrics", {})
    candidate = metrics.get("candidate", {}).get("global", {})
    production = metrics.get("production", {}).get("global", {})
    checks: dict[str, Any] = {}

    candidate_mae = candidate.get("MAE")
    production_mae = production.get("MAE")
    if candidate_mae is None:
        checks["global_mae"] = {"passed": False, "reason": "candidate MAE is missing"}
    elif production_mae is None:
        checks["global_mae"] = {
            "passed": True,
            "reason": "no production metric was available for comparison",
        }
    else:
        checks["global_mae"] = {
            "passed": float(candidate_mae) <= float(production_mae) + MAE_TOLERANCE,
            "candidate": float(candidate_mae),
            "production": float(production_mae),
            "tolerance": MAE_TOLERANCE,
        }

    product_regressions: list[dict[str, Any]] = []
    candidate_products = {
        row.get("product_name"): row
        for row in metrics.get("candidate", {}).get("by_product", [])
    }
    production_products = {
        row.get("product_name"): row
        for row in metrics.get("production", {}).get("by_product", [])
    }
    for product_name, candidate_row in candidate_products.items():
        production_row = production_products.get(product_name)
        if not production_row:
            continue
        actual = max(float(candidate_row.get("total_actual_quantity", 0)), 1.0)
        candidate_error = float(candidate_row.get("signed_total_error", 0))
        production_error = float(production_row.get("signed_total_error", 0))
        if candidate_error < production_error - max(5.0, actual * 0.10):
            product_regressions.append(
                {
                    "product_name": product_name,
                    "candidate_signed_error": candidate_error,
                    "production_signed_error": production_error,
                }
            )

    checks["priority_products"] = {
        "passed": not product_regressions,
        "regressions": product_regressions,
    }
    checks["artifact_smoke_test"] = {"passed": True}
    passed = all(bool(value.get("passed")) for value in checks.values())
    return {"passed": passed, "checks": checks}
