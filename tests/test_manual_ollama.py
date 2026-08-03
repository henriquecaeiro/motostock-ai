"""Opt-in smoke test for a real local Ollama installation."""

from __future__ import annotations

import asyncio
import os

import pytest

from api.services.ollama_service import OllamaService


@pytest.mark.manual
@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run the real Ollama check.",
)
def test_real_ollama_chat_model_is_available():
    async def check() -> None:
        async with OllamaService() as ollama:
            assert await ollama.is_available()
            assert await ollama.has_model()
            answer = await ollama.ask(
                prompt="Reply with the single word OK.",
                options={"temperature": 0.0},
            )
            assert answer.strip()

    asyncio.run(check())
