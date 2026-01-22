import asyncio
from llm_council import settings
from llm_council.clients.openai_compat import OpenAICompatibleResponsesClient
from llm_council.clients.anthropic import AnthropicClient
from llm_council.clients.gemini import GeminiClient
from llm_council.roles import ModelInfo

def build_clients_with_roster():
    clients = []
    roster = []

    def add(provider: str, model: str, client):
        clients.append(client)
        roster.append(ModelInfo(idx=len(roster) + 1, provider=provider, model=model))

    if settings.OPENAI_API_KEY:
        model = settings.OPENAI_MODEL or "gpt-5-nano"
        add(
            "openai",
            model,
            OpenAICompatibleResponsesClient(
                provider="openai",
                api_key=settings.OPENAI_API_KEY,
                model=model,
                base_url="https://api.openai.com/v1",
            ),
        )

    if settings.XAI_API_KEY:
        model = settings.XAI_MODEL or "grok-3-mini"
        add(
            "xai",
            model,
            OpenAICompatibleResponsesClient(
                provider="xai",
                api_key=settings.XAI_API_KEY,
                model=model,
                base_url="https://api.x.ai/v1",
            ),
        )

    if settings.ANTHROPIC_API_KEY:
        model = settings.ANTHROPIC_MODEL or "claude-haiku-4-5"
        add(
            "anthropic",
            model,
            AnthropicClient(
                api_key=settings.ANTHROPIC_API_KEY,
                model=model,
            ),
        )

    if settings.GEMINI_API_KEY:
        model = settings.GEMINI_MODEL or "gemini-2.5-flash-lite"
        add(
            "gemini",
            model,
            GeminiClient(
                api_key=settings.GEMINI_API_KEY,
                model=model,
            ),
        )

    return clients, roster

async def ask_all(user_prompt: str, *, system_prompt: str | None = None):
    clients, _ = build_clients_with_roster()
    tasks = [c.generate(user_prompt, system_prompt=system_prompt) for c in clients]
    return await asyncio.gather(*tasks)

def build_client_by_provider():
    clients, roster = build_clients_with_roster()
    client_by_provider = {m.provider: c for m, c in zip(roster, clients)}
    return clients, roster, client_by_provider
