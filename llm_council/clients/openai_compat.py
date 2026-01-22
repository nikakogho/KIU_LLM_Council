import time
import httpx
from llm_council.types import LLMReply
from llm_council.clients.base import LLMClient

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
        if not api_key:
            raise ValueError(f"{provider} API key is required")
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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature
        }

        if self.model.startswith("gpt-5"):
            del payload["max_tokens"]
            del payload["temperature"]

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
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

            if not hasattr(e, "response") or e.response is None:
                error_msg = str(e)
            else:
                error_msg = f"{e}\n{e.response.text}"

            return LLMReply(
                provider=self.provider,
                model=self.model,
                text="",
                latency_ms=latency_ms,
                raw=None,
                error=error_msg,
            )