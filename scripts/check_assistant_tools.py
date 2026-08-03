"""Run manual assistant checks for current-data tools and mixed retrieval."""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from api.main import app


CHECK_MESSAGES = (
    "List products",
    "Forecast Bag Delivery 45L for 7 days",
    "Show current critical recommendations",
    "Show recommendation summary",
    "Why is the current recommendation for Bag Delivery 45L important?",
)


def main() -> int:
    """Execute representative read-only assistant tool requests."""

    _configure_utf8_output()
    load_dotenv()

    try:
        with TestClient(app) as client:
            for message in CHECK_MESSAGES:
                response = client.post(
                    "/assistant/chat",
                    json={"message": message},
                )
                payload = response.json()
                print(f"\nQuery: {message}")
                print(f"Status: {response.status_code}")
                print(f"Tools used: {payload.get('tools_used', [])}")
                print(f"Sources: {payload.get('sources', [])}")
                answer = payload.get("answer", payload.get("detail", ""))
                print(f"Answer preview: {str(answer)[:800]}")
    except Exception:
        print(
            "Assistant tool check failed: verify the local data and model artifacts.",
            file=sys.stderr,
        )
        return 1

    return 0


def _configure_utf8_output() -> None:
    """Prefer UTF-8 on Windows terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
