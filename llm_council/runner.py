import asyncio
from llm_council import settings
from llm_council.clients.openai_compat import OpenAICompatibleResponsesClient
from llm_council.clients.anthropic import AnthropicClient
from llm_council.clients.gemini import GeminiClient

def build_clients():
    clients = []

    if settings.OPENAI_API_KEY:
        clients.append(OpenAICompatibleResponsesClient(
            provider="openai",
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL or "gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        ))

    if settings.XAI_API_KEY:
        # xAI is OpenAI-compatible and uses base https://api.x.ai 
        clients.append(OpenAICompatibleResponsesClient(
            provider="xai",
            api_key=settings.XAI_API_KEY,
            model=settings.XAI_MODEL or "grok-3-mini",
            base_url="https://api.x.ai/v1",
        ))

    if settings.ANTHROPIC_API_KEY:
        clients.append(AnthropicClient(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL or "claude-haiku-4-5",
        ))

    if settings.GEMINI_API_KEY:
        clients.append(GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL or "gemini-2.5-flash-lite",
        ))

    return clients

async def ask_all(user_prompt: str, *, system_prompt: str | None = None):
    clients = build_clients()
    tasks = [c.generate(user_prompt, system_prompt=system_prompt) for c in clients]
    return await asyncio.gather(*tasks)
