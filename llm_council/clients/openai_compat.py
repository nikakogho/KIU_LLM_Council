import time
import httpx
from llm_council.types import LLMReply
from llm_council.clients.base import LLMClient

def _extract_responses_output_text(resp_json: dict) -> str:
    chunks: list[str] = []
    for item in resp_json.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    chunks.append(c.get("text", ""))
    return "".join(chunks).strip()

class OpenAICompatibleResponsesClient(LLMClient):
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 60.0,
        max_output_tokens: int = 800,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    async def generate(self, user_prompt: str, *, system_prompt: str | None = None) -> LLMReply:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict = {
            "model": self.model,
            "input": user_prompt,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        # OpenAI supports a top-level "instructions" field for system/dev message. :contentReference[oaicite:6]{index=6}
        if system_prompt:
            payload["instructions"] = system_prompt

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(f"{self.base_url}/responses", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            text = _extract_responses_output_text(data)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LLMReply(
                provider=self.provider,
                model=self.model,
                text=text,
                latency_ms=latency_ms,
                raw=data,
                usage=data.get("usage"),
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LLMReply(
                provider=self.provider,
                model=self.model,
                text="",
                latency_ms=latency_ms,
                raw=None,
                error=str(e),
            )
