import asyncio

from api.services.ollama_service import (
    OllamaService,
    OllamaServiceError,
)


async def main() -> None:
    async with OllamaService() as ollama:
        available = await ollama.is_available()

        print(f"Ollama available: {available}")

        if not available:
            print("Start Ollama before running this test.")
            return

        models = await ollama.list_models()

        print(f"Installed models: {models}")

        if not await ollama.has_model():
            print(f"Required model is not installed: {ollama.default_model}")
            return

        answer = await ollama.ask(
            prompt=(
                "Explain safety stock in one short sentence. "
                "Return only the explanation."
            ),
            system_prompt=("You are a concise inventory assistant."),
            options={
                "temperature": 0.2,
            },
        )

        print("\nModel answer:")
        print(answer)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except OllamaServiceError as exc:
        print(f"Ollama error: {exc}")
