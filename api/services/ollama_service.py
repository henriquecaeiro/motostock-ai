from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from api.config import Settings, load_settings


class OllamaServiceError(RuntimeError):
    """Base exception for Ollama service errors."""


class OllamaConnectionError(OllamaServiceError):
    """Raised when the application cannot connect to Ollama."""


class OllamaTimeoutError(OllamaServiceError):
    """Raised when Ollama takes too long to respond."""


class OllamaRequestError(OllamaServiceError):
    """Raised when Ollama returns an unsuccessful HTTP status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class OllamaInvalidResponseError(OllamaServiceError):
    """Raised when Ollama returns an unexpected response."""


class OllamaModelUnavailableError(OllamaServiceError):
    """Raised when the configured Ollama model is not installed."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"The configured Ollama model is unavailable: {model}")


class OllamaService:
    """Service responsible for communicating with the Ollama HTTP API."""

    VALID_MESSAGE_ROLES = {"system", "user", "assistant"}

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        embedding_model: str | None = None,
        timeout_seconds: float | None = None,
        keep_alive: str = "5m",
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.base_url = (
            base_url or self.settings.ollama_base_url
        ).rstrip("/")

        self.default_model = default_model or self.settings.ollama_model

        configured_embedding_model = (
            self.settings.ollama_embedding_model
            if embedding_model is None
            else embedding_model
        )

        self.default_embedding_model = configured_embedding_model.strip()

        if not self.default_embedding_model:
            raise ValueError("The Ollama embedding model cannot be empty.")

        self.keep_alive = keep_alive

        configured_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.settings.ollama_timeout_seconds
        )

        # LLM generation may take longer than a normal API request,
        # especially when the model is running on the CPU.
        timeout = httpx.Timeout(
            configured_timeout,
            connect=5.0,
            write=30.0,
            pool=5.0,
        )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client and release its resources."""

        await self._client.aclose()

    async def __aenter__(self) -> "OllamaService":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.close()

    async def is_available(self) -> bool:
        """
        Check whether the Ollama server is available.

        This method returns False instead of raising an exception because
        it is intended to be used as a simple availability check.
        """

        try:
            await self.list_models()
        except OllamaServiceError:
            return False

        return True

    async def list_models(self) -> list[str]:
        """Return the names of the models installed in Ollama."""

        data = await self._request_json(
            method="GET",
            path="/api/tags",
        )

        models = data.get("models")

        if not isinstance(models, list):
            raise OllamaInvalidResponseError(
                "Ollama response does not contain a valid 'models' list."
            )

        model_names: list[str] = []

        for index, model_data in enumerate(models):
            if not isinstance(model_data, dict):
                raise OllamaInvalidResponseError(
                    f"Invalid model information at position {index}."
                )

            # Current responses normally contain both "model" and "name".
            model_name = model_data.get("model") or model_data.get("name")

            if not isinstance(model_name, str) or not model_name.strip():
                raise OllamaInvalidResponseError(
                    f"Missing model name at position {index}."
                )

            model_names.append(model_name.strip())

        return sorted(set(model_names))

    async def has_model(self, model: str | None = None) -> bool:
        """Check whether a specific model is installed."""

        selected_model = model or self.default_model
        available_models = await self.list_models()

        return selected_model in available_models

    async def ensure_model_available(
        self,
        model: str | None = None,
    ) -> None:
        """Raise when the requested model is not installed in Ollama."""

        selected_model = model or self.default_model
        available_models = await self.list_models()

        if selected_model not in available_models:
            raise OllamaModelUnavailableError(selected_model)

    async def ask(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Send a single user prompt to Ollama.

        This is a convenience method for simple requests without
        conversation history.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The prompt cannot be empty.")

        messages: list[dict[str, str]] = []

        if system_prompt is not None:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise ValueError("The system prompt must be a non-empty string.")

            messages.append(
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": f"{prompt.strip()}\n\n/no_think",
            }
        )

        return await self.chat(
            messages=messages,
            model=model,
            options=options,
        )

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Send a conversation to Ollama and return only the final answer.

        The messages can contain the roles:
        - system
        - user
        - assistant
        """

        selected_model = (model or self.default_model).strip()

        if not selected_model:
            raise ValueError("The model name cannot be empty.")

        normalized_messages = self._normalize_messages(messages)

        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": normalized_messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
        }

        if options:
            payload["options"] = dict(options)

        try:
            data = await self._request_json(
                method="POST",
                path="/api/chat",
                json=payload,
            )
        except OllamaRequestError as exc:
            if exc.status_code == 404:
                raise OllamaModelUnavailableError(selected_model) from exc
            raise

        if data.get("done") is not True:
            raise OllamaInvalidResponseError(
                "Ollama did not finish generating the response."
            )

        message = data.get("message")

        if not isinstance(message, dict):
            raise OllamaInvalidResponseError(
                "Ollama response does not contain a valid 'message' object."
            )

        content = message.get("content")

        if not isinstance(content, str):
            raise OllamaInvalidResponseError("Ollama returned an invalid answer.")

        final_answer = self._extract_final_answer(content)

        if not final_answer:
            raise OllamaInvalidResponseError("Ollama returned an empty answer.")

        return final_answer

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for one or more text inputs."""

        normalized_inputs = self._normalize_embed_inputs(inputs)

        selected_model = (model or self.default_embedding_model).strip()

        if not selected_model:
            raise ValueError("The embedding model name cannot be empty.")

        payload: dict[str, Any] = {
            "model": selected_model,
            "input": normalized_inputs,
            "keep_alive": self.keep_alive,
        }

        try:
            data = await self._request_json(
                method="POST",
                path="/api/embed",
                json=payload,
            )
        except OllamaRequestError as exc:
            if exc.status_code == 404:
                raise OllamaModelUnavailableError(selected_model) from exc
            raise

        return self._parse_embeddings_response(
            data,
            expected_count=len(normalized_inputs),
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an HTTP request and validate the JSON response."""

        try:
            response = await self._client.request(
                method=method,
                url=path,
                **kwargs,
            )

        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama took too long to respond.") from exc

        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}."
            ) from exc

        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                f"Communication error with Ollama: {exc}"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_detail = self._extract_error_detail(response)

            raise OllamaRequestError(
                f"Ollama returned HTTP status {response.status_code}: {error_detail}",
                status_code=response.status_code,
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaInvalidResponseError(
                "Ollama returned a response that is not valid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise OllamaInvalidResponseError(
                "Ollama returned an unexpected JSON structure."
            )

        return data

    @classmethod
    def _normalize_messages(
        cls,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        """Validate and normalize chat messages."""

        if not messages:
            raise ValueError("At least one message is required.")

        normalized_messages: list[dict[str, str]] = []

        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"Message at position {index} must be a mapping.")

            role = message.get("role")
            content = message.get("content")

            if role not in cls.VALID_MESSAGE_ROLES:
                raise ValueError(f"Invalid role at position {index}: {role!r}.")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Message content at position {index} cannot be empty."
                )

            normalized_messages.append(
                {
                    "role": role,
                    "content": content.strip(),
                }
            )

        return normalized_messages

    @staticmethod
    def _normalize_embed_inputs(inputs: str | Sequence[str]) -> list[str]:
        """Validate and normalize embedding inputs."""

        if isinstance(inputs, str):
            stripped_input = inputs.strip()

            if not stripped_input:
                raise ValueError("Embedding input cannot be empty.")

            return [stripped_input]

        if isinstance(inputs, bytes):
            raise TypeError("Embedding inputs must be strings.")

        if not isinstance(inputs, Sequence):
            raise TypeError(
                "Embedding inputs must be a string or a sequence of strings."
            )

        if not inputs:
            raise ValueError("At least one embedding input is required.")

        normalized_inputs: list[str] = []

        for index, item in enumerate(inputs):
            if not isinstance(item, str):
                raise TypeError(
                    f"Embedding input at position {index} must be a string."
                )

            stripped_item = item.strip()

            if not stripped_item:
                raise ValueError(
                    f"Embedding input at position {index} cannot be empty."
                )

            normalized_inputs.append(stripped_item)

        return normalized_inputs

    @staticmethod
    def _parse_embeddings_response(
        data: Mapping[str, Any],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        """Validate and normalize an Ollama embeddings response."""

        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise OllamaInvalidResponseError(
                "Ollama response does not contain valid embeddings."
            )

        if len(embeddings) != expected_count:
            raise OllamaInvalidResponseError(
                "Ollama returned an unexpected number of embeddings."
            )

        parsed_embeddings: list[list[float]] = []
        expected_dimension: int | None = None

        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list):
                raise OllamaInvalidResponseError(
                    f"Embedding at position {index} is not a list."
                )

            if not vector:
                raise OllamaInvalidResponseError(
                    f"Embedding at position {index} is empty."
                )

            parsed_vector: list[float] = []

            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise OllamaInvalidResponseError(
                        f"Embedding at position {index} contains invalid values."
                    )

                float_value = float(value)

                if not math.isfinite(float_value):
                    raise OllamaInvalidResponseError(
                        f"Embedding at position {index} contains non-finite values."
                    )

                parsed_vector.append(float_value)

            if expected_dimension is None:
                expected_dimension = len(parsed_vector)
            elif len(parsed_vector) != expected_dimension:
                raise OllamaInvalidResponseError(
                    "Ollama returned embeddings with inconsistent dimensions."
                )

            parsed_embeddings.append(parsed_vector)

        return parsed_embeddings

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        """Extract a readable error message from an Ollama response."""

        try:
            data = response.json()

            if isinstance(data, dict):
                error = data.get("error")

                if isinstance(error, str) and error.strip():
                    return error.strip()

        except ValueError:
            pass

        response_text = response.text.strip()

        if response_text:
            return response_text[:500]

        return "No error details were provided."

    @staticmethod
    def _extract_final_answer(content: str) -> str:
        """
        Remove thinking content accidentally returned inside
        the assistant message.
        """

        cleaned_content = content.strip()

        # Some Qwen versions may return the reasoning followed
        # by a closing </think> tag inside message.content.
        if "</think>" in cleaned_content:
            cleaned_content = cleaned_content.rsplit(
                "</think>",
                maxsplit=1,
            )[-1].strip()

        return cleaned_content
