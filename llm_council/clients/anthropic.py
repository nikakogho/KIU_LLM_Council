import time
import httpx
from llm_council.types import LLMReply
from llm_council.clients.base import LLMClient

def _extract_anthropic_text(resp_json: dict) -> str:
    parts = resp_json.get("content", [])
    chunks = [p.get("text", "") for p in parts if p.get("type") == "text"]
    return "".join(chunks).strip()

class AnthropicClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        max_tokens: int = 800,
        temperature: float = 0.0,
        anthropic_version: str = "2023-06-01",
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.anthropic_version = anthropic_version

    async def generate(self, user_prompt: str, *, system_prompt: str | None = None) -> LLMReply:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }

        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            text = _extract_anthropic_text(data)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LLMReply(
                provider="anthropic",
                model=self.model,
                text=text,
                latency_ms=latency_ms,
                raw=data,
                usage=data.get("usage"),
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LLMReply(provider="anthropic", model=self.model, text="", latency_ms=latency_ms, raw=None, error=str(e))
