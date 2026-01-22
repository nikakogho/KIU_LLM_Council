import os
from dotenv import load_dotenv

load_dotenv()

def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

OPENAI_API_KEY = env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY")
GEMINI_API_KEY = env("GEMINI_API_KEY")
XAI_API_KEY = env("XAI_API_KEY")

OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5-nano")
ANTHROPIC_MODEL = env("ANTHROPIC_MODEL", "claude-haiku-4-5")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-2.5-flash-lite")
XAI_MODEL = env("XAI_MODEL", "grok-3-mini")
