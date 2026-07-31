"""Manual integration check for Ollama and the MotoStock AI system prompt."""

import asyncio
import sys

from dotenv import load_dotenv

from api.prompts import load_system_prompt
from api.services.ollama_service import (
    OllamaService,
    OllamaServiceError,
)


def configure_utf8_output() -> None:
    """Configure standard streams to use UTF-8 when supported."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if callable(reconfigure):
            reconfigure(
                encoding="utf-8",
                errors="replace",
            )


async def main() -> None:
    """Check the real connection with the local Ollama server."""

    load_dotenv()

    try:
        system_prompt = load_system_prompt()
    except RuntimeError:
        print("The assistant system prompt could not be loaded.")
        return

    async with OllamaService() as ollama:
        print("Checking Ollama availability...")

        if not await ollama.is_available():
            print("Ollama is not available.")
            print("Start Ollama and try again.")
            return

        models = await ollama.list_models()

        print(f"Installed models: {models}")
        print(f"Configured model: {ollama.default_model}")

        if not await ollama.has_model():
            print(f"The required model is not installed: {ollama.default_model}")
            print(f"Run: ollama pull {ollama.default_model}")
            return

        questions = [
            (
                "Conceptual question",
                (
                    "What is the difference between demand forecasting "
                    "and stock recommendation?"
                ),
            ),
            (
                "Current-data question",
                "Which products are currently critical?",
            ),
        ]

        for label, question in questions:
            print("\n" + "=" * 60)
            print(f"[{label}]")
            print(f"User: {question}")

            answer = await ollama.ask(
                prompt=question,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,
                },
            )

            print(f"\nAssistant: {answer}")

        print("\n" + "=" * 60)
        print("Manual review required for the current-data question.")
        print(
            "Confirm the assistant did not invent products, quantities, "
            "or live business values."
        )


def run() -> None:
    """Run the manual integration check."""

    configure_utf8_output()

    try:
        asyncio.run(main())
    except OllamaServiceError:
        print("Ollama service error. Check that Ollama is running and reachable.")
    except KeyboardInterrupt:
        print("\nIntegration check interrupted by the user.")


if __name__ == "__main__":
    run()
