"""Allowlisted, read-only business tools for the local assistant."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Type

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from typing_extensions import Annotated

from api.exceptions import (
    ForecastingHTTPError,
    ProductNotFoundError,
    RecommendationHTTPError,
    ServiceUnavailableError,
)
from api.schemas.recommendation import StockStatus
from api.services.forecast_service import ForecastService
from api.services.recommendation_service import RecommendationService
from api.repositories.csv_repository import CsvRepository


ToolName = Literal[
    "list_products",
    "forecast_product",
    "get_recommendations",
    "get_recommendation_summary",
]
ProductName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=200,
    ),
]


class ToolServiceError(RuntimeError):
    """Base error raised while planning or executing a business tool."""


class UnknownToolError(ToolServiceError):
    """Raised when a requested tool is not on the explicit allowlist."""


class ToolArgumentError(ToolServiceError):
    """Raised when tool arguments fail strict validation."""


class ToolDataError(ToolServiceError):
    """Raised when an approved tool cannot read its business data."""


class ListProductsArguments(BaseModel):
    """Arguments for listing products."""

    model_config = ConfigDict(extra="forbid")


class ForecastProductArguments(BaseModel):
    """Arguments for one product forecast."""

    model_config = ConfigDict(extra="forbid")

    product_name: ProductName
    horizon_days: int = Field(default=14, ge=1, le=30)


class GetRecommendationsArguments(BaseModel):
    """Arguments for filtered recommendation retrieval."""

    model_config = ConfigDict(extra="forbid")

    horizon_days: int = Field(default=14, ge=1, le=30)
    stock_status: StockStatus | None = None
    product_name: ProductName | None = None


class GetRecommendationSummaryArguments(BaseModel):
    """Arguments for the recommendation summary."""

    model_config = ConfigDict(extra="forbid")

    horizon_days: int = Field(default=14, ge=1, le=30)


class ToolCall(BaseModel):
    """Validated shape that a tool planner is allowed to execute."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecution(BaseModel):
    """Result returned by a successful read-only tool call."""

    tool_name: ToolName
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class ToolPlan:
    """A deterministic tool call plus whether static RAG context is useful."""

    call: ToolCall
    requires_rag: bool


_ARGUMENT_SCHEMAS: dict[str, Type[BaseModel]] = {
    "list_products": ListProductsArguments,
    "forecast_product": ForecastProductArguments,
    "get_recommendations": GetRecommendationsArguments,
    "get_recommendation_summary": GetRecommendationSummaryArguments,
}

_CONCEPTUAL_MARKERS = (
    "why",
    "how",
    "como",
    "por que",
    "porque",
    "porquê",
    "explain",
    "explique",
    "what is",
    "o que é",
    "o que e",
)

_FORECAST_MARKERS = (
    "forecast",
    "forecasts",
    "previs",
    "previsao",
    "previsoes",
    "demand",
    "demands",
    "demanda",
    "demandas",
)

_RECOMMENDATION_MARKERS = (
    "recommendation",
    "recommendations",
    "recomend",
    "recomendacao",
    "recomendacoes",
    "replenishment",
    "replenishments",
    "reposicao",
    "reposicoes",
    "purchase quantity",
    "quantidade de compra",
)

_CURRENT_MARKERS = (
    "current",
    "currently",
    "latest",
    "today",
    "right now",
    "atual",
    "atuais",
    "atualmente",
    "agora",
    "hoje",
    "recente",
    "recentes",
    "neste momento",
)

_LIST_MARKERS = (
    "list products",
    "show products",
    "known products",
    "quais produtos",
    "liste os produtos",
    "listar produtos",
)

_ACTION_MARKERS = (
    "get",
    "show",
    "give",
    "list",
    "which",
    "quais",
    "mostre",
    "obtenha",
)

_SUMMARY_MARKERS = (
    "summary",
    "resumo",
    "how many critical",
    "quantos produtos",
    "total recommended",
    "total de compras",
)


class ToolService:
    """Expose only the assistant's four read-only business operations."""

    ALLOWED_TOOLS: frozenset[str] = frozenset(_ARGUMENT_SCHEMAS)

    def __init__(
        self,
        repository: CsvRepository,
        forecast_service: ForecastService,
        recommendation_service: RecommendationService,
    ) -> None:
        self.repository = repository
        self.forecast_service = forecast_service
        self.recommendation_service = recommendation_service

    def plan_query(self, message: str) -> ToolPlan | None:
        """Infer at most one allowlisted tool call from a user question.

        This intentionally uses small deterministic rules instead of asking the
        local model to emit executable code or arbitrary tool names.
        """

        if not isinstance(message, str) or not message.strip():
            return None

        normalized = _normalize_text(message)
        product_name = self._extract_product_name(message)
        horizon_days = _extract_horizon_days(message)
        stock_status = _extract_stock_status(normalized)
        conceptual = _contains_any_marker(normalized, _CONCEPTUAL_MARKERS)

        if _contains_any_marker(normalized, _LIST_MARKERS):
            return ToolPlan(
                call=ToolCall(name="list_products", arguments={}),
                requires_rag=conceptual,
            )

        if _contains_any_marker(normalized, _SUMMARY_MARKERS):
            return ToolPlan(
                call=ToolCall(
                    name="get_recommendation_summary",
                    arguments={"horizon_days": horizon_days},
                ),
                requires_rag=conceptual,
            )

        if _contains_any_marker(normalized, _FORECAST_MARKERS) and product_name:
            return ToolPlan(
                call=ToolCall(
                    name="forecast_product",
                    arguments={
                        "product_name": product_name,
                        "horizon_days": horizon_days,
                    },
                ),
                requires_rag=conceptual,
            )

        recommendation_requested = _contains_any_marker(
            normalized,
            _RECOMMENDATION_MARKERS,
        ) and (
            not conceptual
            or _contains_any_marker(normalized, _CURRENT_MARKERS)
            or _contains_any_marker(normalized, _ACTION_MARKERS)
        )
        current_stock_question = (
            ("stock" in normalized or "estoque" in normalized)
            and _contains_any_marker(normalized, _CURRENT_MARKERS)
        )
        status_filter_requested = stock_status is not None and (
            _contains_any_marker(normalized, _CURRENT_MARKERS)
            or _contains_any_marker(
                normalized,
                ("which", "quais", "filter", "filtro"),
            )
        )

        if recommendation_requested or current_stock_question or status_filter_requested:
            arguments: dict[str, Any] = {"horizon_days": horizon_days}

            if stock_status is not None:
                arguments["stock_status"] = stock_status

            if product_name:
                arguments["product_name"] = product_name

            return ToolPlan(
                call=ToolCall(
                    name="get_recommendations",
                    arguments=arguments,
                ),
                requires_rag=conceptual,
            )

        return None

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolExecution:
        """Validate and execute one allowlisted read-only operation."""

        if tool_name not in self.ALLOWED_TOOLS:
            raise UnknownToolError("The requested tool is not available.")

        if not isinstance(arguments, Mapping):
            raise ToolArgumentError("Tool arguments must be a JSON object.")

        argument_schema = _ARGUMENT_SCHEMAS[tool_name]

        try:
            parsed_arguments = argument_schema.model_validate(dict(arguments))
        except (ValidationError, TypeError, ValueError) as exc:
            raise ToolArgumentError("Tool arguments are invalid.") from exc

        normalized_arguments = _model_dump(parsed_arguments, exclude_none=True)

        try:
            result = self._execute_validated(tool_name, parsed_arguments)
        except ProductNotFoundError as exc:
            raise ToolArgumentError("The requested product was not found.") from exc
        except (ForecastingHTTPError, RecommendationHTTPError, ServiceUnavailableError) as exc:
            raise ToolDataError("The requested business data is unavailable.") from exc
        except HTTPException as exc:
            raise ToolArgumentError("The requested tool arguments are invalid.") from exc
        except Exception as exc:
            raise ToolDataError("The requested business data could not be read.") from exc

        return ToolExecution(
            tool_name=tool_name,
            arguments=normalized_arguments,
            result=_json_safe(result),
        )

    def _execute_validated(
        self,
        tool_name: str,
        arguments: BaseModel,
    ) -> dict[str, Any]:
        """Dispatch only to known service methods."""

        values = _model_dump(arguments, exclude_none=True)

        if tool_name == "list_products":
            products = self.repository.list_products()
            return {"count": len(products), "products": products}

        if tool_name == "forecast_product":
            return self.forecast_service.predict_product(**values)

        if tool_name == "get_recommendations":
            return self.recommendation_service.get_recommendations(**values)

        if tool_name == "get_recommendation_summary":
            return self.recommendation_service.get_summary(**values)

        raise UnknownToolError("The requested tool is not available.")

    def _extract_product_name(self, message: str) -> str | None:
        """Match known products first and otherwise capture an explicit candidate."""

        normalized_message = _normalize_text(message)
        products = sorted(
            self.repository.list_products(),
            key=lambda product: len(_normalize_text(product)),
            reverse=True,
        )

        for product in products:
            if _normalize_text(product) in normalized_message:
                return product

        candidate_match = re.search(
            r"(?:for|para|produto|product)\s+['\"]?([^,?!.]+)",
            message,
            flags=re.IGNORECASE,
        )

        if candidate_match is None:
            return None

        candidate = candidate_match.group(1).strip().strip("'\"")
        candidate = re.split(
            r"\s+(?:for|next|horizon|por|nos? próximos?)\s+\d+",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        return candidate or None


def _normalize_text(value: str) -> str:
    """Normalize accents and case for deterministic intent matching."""

    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().split())


def _contains_any_marker(value: str, markers: tuple[str, ...]) -> bool:
    """Match intent markers as words so ``how`` does not match ``show``."""

    return any(
        re.search(rf"\b{re.escape(marker)}\b", value) is not None
        for marker in markers
    )


def _extract_horizon_days(message: str) -> int:
    """Extract an explicit day count while avoiding product numbers such as 45L."""

    match = re.search(r"\b(\d{1,3})\s*(?:days?|dias?)\b", message, re.IGNORECASE)

    if match is None:
        return 14

    return int(match.group(1))


def _extract_stock_status(normalized_message: str) -> str | None:
    """Map Portuguese and English status words to the strict API values."""

    aliases = {
        "critical": "critical",
        "critico": "critical",
        "critica": "critical",
        "criticos": "critical",
        "criticas": "critical",
        "warning": "warning",
        "aviso": "warning",
        "alerta": "warning",
        "healthy": "healthy",
        "saudavel": "healthy",
        "overstock": "overstock",
        "excesso": "overstock",
        "sobrestoque": "overstock",
    }

    for alias, status in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized_message):
            return status

    return None


def _model_dump(model: BaseModel, *, exclude_none: bool = False) -> dict[str, Any]:
    """Serialize Pydantic models without exposing validation internals."""

    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=exclude_none)

    return model.dict(exclude_none=exclude_none)


def _json_safe(value: Any) -> dict[str, Any]:
    """Convert service output to JSON-native values before the LLM sees it."""

    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ToolDataError("The tool returned data that cannot be serialized.") from exc


def format_tool_execution(execution: ToolExecution) -> str:
    """Render verified tool data without allowing an LLM to change values."""

    serialized_arguments = json.dumps(
        execution.arguments,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    serialized_result = json.dumps(
        execution.result,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    return (
        f"Verified read-only tool result: {execution.tool_name}\n"
        "Validated arguments:\n"
        "```json\n"
        f"{serialized_arguments}\n"
        "```\n"
        "The JSON below is the exact result returned by the application service.\n"
        "```json\n"
        f"{serialized_result}\n"
        "```"
    )
