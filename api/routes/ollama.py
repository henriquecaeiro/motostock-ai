from fastapi import APIRouter, HTTPException, status

from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaRequestError,
    OllamaService,
    OllamaTimeoutError,
)

router = APIRouter(prefix="/ai", tags=["AI"])


@router.post("/ask")
async def ask_model(prompt: str) -> dict[str, str]:

    async with OllamaService() as ollama:
        try:
            answer = await ollama.ask(prompt)

            return {"answer": answer}

        except OllamaTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
            ) from exc

        except OllamaConnectionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

        except OllamaRequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc

        except OllamaInvalidResponseError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc
