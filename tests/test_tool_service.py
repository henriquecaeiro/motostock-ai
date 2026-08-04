"""Tests for the allowlisted read-only assistant tools."""

from __future__ import annotations

import pytest

from api.exceptions import ProductNotFoundError
from api.services.tool_service import (
    ToolArgumentError,
    ToolService,
    UnknownToolError,
    format_tool_execution,
)


class FakeRepository:
    def __init__(self):
        self.products = ["Bag Delivery 45L", "Capacete LS2"]

    def list_products(self):
        return list(self.products)


class FakeForecastService:
    def predict_product(self, *, product_name, horizon_days):
        if product_name not in {"Bag Delivery 45L", "Capacete LS2"}:
            raise ProductNotFoundError(product_name)
        return {
            "product_name": product_name,
            "horizon_days": horizon_days,
            "forecasted_demand_units": 17,
        }

    def get_forecast_summary(self, *, horizon_days, limit):
        return {
            "selected_model": "fake-model",
            "horizon_days": horizon_days,
            "total_products": 2,
            "total_forecasted_demand_units": 30,
            "top_products": [
                {
                    "product_name": "Bag Delivery 45L",
                    "forecasted_demand_units": 17,
                    "forecasted_demand_raw": 16.5,
                    "forecasted_demand_non_negative": 17.0,
                }
            ][:limit],
        }


class FakeRecommendationService:
    def get_recommendations(self, **kwargs):
        return {
            "horizon_days": kwargs["horizon_days"],
            "stock_status": kwargs.get("stock_status"),
            "product_name": kwargs.get("product_name"),
            "count": 2,
            "recommendations": [
                {
                    "product_name": "Bag Delivery 45L",
                    "recommended_purchase_quantity": 4,
                    "stock_status": "warning",
                },
                {
                    "product_name": "Capacete LS2",
                    "recommended_purchase_quantity": 2,
                    "stock_status": "warning",
                },
            ],
        }

    def get_summary(self, *, horizon_days):
        return {
            "forecast_horizon_days": horizon_days,
            "total_products": 2,
            "critical_products": 1,
            "total_recommended_purchase_units": 4,
        }


def make_service() -> ToolService:
    return ToolService(
        repository=FakeRepository(),
        forecast_service=FakeForecastService(),
        recommendation_service=FakeRecommendationService(),
    )


def test_list_products_returns_current_repository_values() -> None:
    execution = make_service().execute("list_products", {})

    assert execution.tool_name == "list_products"
    assert execution.result == {
        "count": 2,
        "products": ["Bag Delivery 45L", "Capacete LS2"],
    }


def test_forecast_tool_preserves_service_values() -> None:
    execution = make_service().execute(
        "forecast_product",
        {"product_name": "Bag Delivery 45L", "horizon_days": 7},
    )

    assert execution.arguments == {
        "product_name": "Bag Delivery 45L",
        "horizon_days": 7,
    }
    assert execution.result["forecasted_demand_units"] == 17


def test_forecast_summary_tool_preserves_service_values() -> None:
    execution = make_service().execute(
        "get_forecast_summary",
        {"horizon_days": 7, "limit": 1},
    )

    assert execution.arguments == {"horizon_days": 7, "limit": 1}
    assert execution.result["total_forecasted_demand_units"] == 30
    assert len(execution.result["top_products"]) == 1


def test_recommendations_tool_accepts_status_filter() -> None:
    execution = make_service().execute(
        "get_recommendations",
        {"horizon_days": 14, "stock_status": "warning"},
    )

    assert execution.arguments == {
        "horizon_days": 14,
        "stock_status": "warning",
    }
    assert execution.result["recommendations"][0]["recommended_purchase_quantity"] == 4


def test_recommendations_tool_applies_requested_limit() -> None:
    execution = make_service().execute(
        "get_recommendations",
        {"horizon_days": 14, "stock_status": "critical", "limit": 1},
    )

    assert execution.arguments["limit"] == 1
    assert execution.result["count"] == 1
    assert len(execution.result["recommendations"]) == 1


def test_summary_tool_returns_exact_summary() -> None:
    execution = make_service().execute(
        "get_recommendation_summary",
        {"horizon_days": 10},
    )

    assert execution.result["forecast_horizon_days"] == 10
    assert execution.result["total_recommended_purchase_units"] == 4


def test_unknown_product_is_reported_without_inventing_a_result() -> None:
    with pytest.raises(ToolArgumentError, match="product was not found"):
        make_service().execute(
            "forecast_product",
            {"product_name": "Unknown Product", "horizon_days": 14},
        )


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("forecast_product", {"product_name": "Bag Delivery 45L", "horizon_days": 0}),
        ("forecast_product", {"product_name": "Bag Delivery 45L", "horizon_days": 31}),
        ("get_forecast_summary", {"horizon_days": 14, "limit": 0}),
        ("get_forecast_summary", {"horizon_days": 14, "limit": 21}),
        ("get_recommendations", {"stock_status": "invalid"}),
        ("get_recommendations", {"horizon_days": 14, "unexpected": True}),
    ],
)
def test_invalid_tool_arguments_are_rejected(tool_name, arguments) -> None:
    with pytest.raises(ToolArgumentError):
        make_service().execute(tool_name, arguments)


def test_unknown_tool_is_rejected_by_allowlist() -> None:
    with pytest.raises(UnknownToolError):
        make_service().execute("delete_database", {})


def test_query_planner_separates_conceptual_questions_from_current_data() -> None:
    service = make_service()

    assert service.plan_query("How does the recommendation formula work?") is None

    current_plan = service.plan_query("What are the current recommendations?")
    assert current_plan is not None
    assert current_plan.call.name == "get_recommendations"
    assert current_plan.requires_rag is False

    summary_plan = service.plan_query("Show recommendation summary")
    assert summary_plan is not None
    assert summary_plan.call.name == "get_recommendation_summary"
    assert summary_plan.requires_rag is False

    forecast_plan = service.plan_query("Forecast Bag Delivery 45L for 7 days")
    assert forecast_plan is not None
    assert forecast_plan.call.name == "forecast_product"
    assert forecast_plan.call.arguments == {
        "product_name": "Bag Delivery 45L",
        "horizon_days": 7,
    }

    portuguese_plan = service.plan_query(
        "Quais são as recomendações críticas atuais?"
    )
    assert portuguese_plan is not None
    assert portuguese_plan.call.name == "get_recommendations"
    assert portuguese_plan.call.arguments["stock_status"] == "critical"


def test_query_planner_recognizes_portuguese_top_five_critical_request() -> None:
    plan = make_service().plan_query(
        "Me mande a lista dos 5 produtos mais críticos."
    )

    assert plan is not None
    assert plan.call.name == "get_recommendations"
    assert plan.call.arguments == {
        "horizon_days": 14,
        "stock_status": "critical",
        "limit": 5,
    }


@pytest.mark.parametrize(
    "message",
    [
        "Quanto devo comprar de Bag Delivery 45L?",
        "Quantas unidades de Bag Delivery 45L devo comprar?",
        "Preciso repor Bag Delivery 45L?",
        "How many units of Bag Delivery 45L should I buy?",
        "Should I restock Bag Delivery 45L?",
        "What is the recommended purchase quantity for Bag Delivery 45L?",
        "How much Bag Delivery 45L should I reorder?",
    ],
)
def test_query_planner_routes_purchase_and_replenishment_questions(message) -> None:
    plan = make_service().plan_query(message)

    assert plan is not None
    assert plan.call.name == "get_recommendations"
    assert plan.call.arguments == {
        "horizon_days": 14,
        "product_name": "Bag Delivery 45L",
    }


def test_query_planner_routes_general_forecast_without_fake_product() -> None:
    plan = make_service().plan_query(
        "Qual é a previsão para os próximos 14 dias?"
    )

    assert plan is not None
    assert plan.call.name == "get_forecast_summary"
    assert plan.call.arguments == {
        "horizon_days": 14,
        "limit": 5,
    }


def test_query_planner_routes_general_forecast_in_english() -> None:
    plan = make_service().plan_query("What is the forecast for the next 7 days?")

    assert plan is not None
    assert plan.call.name == "get_forecast_summary"
    assert plan.call.arguments["horizon_days"] == 7
    assert "product_name" not in plan.call.arguments


def test_query_planner_routes_forecast_summary_with_summary_word() -> None:
    plan = make_service().plan_query(
        "Mostre um resumo da previsão para 30 dias."
    )

    assert plan is not None
    assert plan.call.name == "get_forecast_summary"
    assert plan.call.arguments == {"horizon_days": 30, "limit": 5}


def test_query_planner_keeps_specific_forecast_product_and_horizon() -> None:
    plan = make_service().plan_query(
        "Qual é a previsão de Bag Delivery 45L para os próximos 14 dias?"
    )

    assert plan is not None
    assert plan.call.name == "forecast_product"
    assert plan.call.arguments == {
        "product_name": "Bag Delivery 45L",
        "horizon_days": 14,
    }


def test_query_planner_does_not_create_temporal_product_name() -> None:
    plan = make_service().plan_query("Qual é a previsão para os próximos 14 dias?")

    assert plan is not None
    assert plan.call.name == "get_forecast_summary"
    assert "os próximos 14 dias" not in plan.call.arguments.values()


def test_query_planner_keeps_conceptual_purchase_question_without_tool() -> None:
    plan = make_service().plan_query("Como a compra recomendada é calculada?")

    assert plan is None


def test_unknown_explicit_product_is_rejected_safely() -> None:
    service = make_service()
    plan = service.plan_query("Previsão do Produto Inexistente para 14 dias")

    assert plan is not None
    assert plan.call.name == "forecast_product"
    assert plan.call.arguments["product_name"] != "os próximos 14 dias"

    with pytest.raises(ToolArgumentError, match="product was not found"):
        service.execute(plan.call.name, plan.call.arguments)


def test_query_planner_extracts_forecast_summary_limit() -> None:
    plan = make_service().plan_query(
        "Mostre os 5 produtos com maior demanda prevista para 14 dias"
    )

    assert plan is not None
    assert plan.call.name == "get_forecast_summary"
    assert plan.call.arguments == {"horizon_days": 14, "limit": 5}


def test_query_planner_keeps_mixed_questions_grounded() -> None:
    plan = make_service().plan_query(
        "Why is the current recommendation for Bag Delivery 45L important?"
    )

    assert plan is not None
    assert plan.call.name == "get_recommendations"
    assert plan.requires_rag is True


def test_tool_result_renderer_keeps_exact_values_without_internal_names() -> None:
    execution = make_service().execute(
        "forecast_product",
        {"product_name": "Bag Delivery 45L", "horizon_days": 14},
    )

    rendered = format_tool_execution(execution)

    assert "17" in rendered
    assert "previsão de demanda" in rendered
    assert "forecast_product" not in rendered
