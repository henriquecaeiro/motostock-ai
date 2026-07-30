from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"


def load_system_prompt() -> str:
    """Load the MotoStock AI system prompt from disk"""

    try:
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Could not load the system prompt: {SYSTEM_PROMPT_PATH}"
        ) from exc

    if not system_prompt:
        raise RuntimeError("The MotoStock AI system prompt is empty.")

    return system_prompt
