"""Manual integration check for Ollama and the MotoStock AI system prompt."""

import asyncio

from dotenv import load_dotenv

from api.prompts import load_system_prompt
from api.services.ollama_service import (
    OllamaService,
    OllamaServiceError,
)


async def main() -> None:
    """Check the real connection with the local Ollama server."""

    # The current project has a .env file, but it is not loaded
    # automatically anywhere yet.
    load_dotenv()

    system_prompt = load_system_prompt()

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
            return

        questions = [
            "What is safety stock?",
            "How many helmets should I purchase today?",
            "Give me a hypothetical stock recommendation example.",
        ]

        for question in questions:
            print("\n" + "=" * 60)
            print(f"User: {question}")

            answer = await ollama.ask(
                prompt=question,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,
                },
            )

            print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except OllamaServiceError as exc:
        print(f"Ollama service error: {exc}")
    except KeyboardInterrupt:
        print("\nIntegration check interrupted by the user.")
