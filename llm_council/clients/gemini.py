import time
import httpx
from llm_council.types import LLMReply
from llm_council.clients.base import LLMClient

def _extract_gemini_text(resp_json: dict) -> str:
    cands = resp_json.get("candidates") or []
    if not cands:
        return ""
    parts = (((cands[0] or {}).get("content") or {}).get("parts")) or []
    chunks = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
    return "".join(chunks).strip()

class GeminiClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    async def generate(self, user_prompt: str, *, system_prompt: str | None = None) -> LLMReply:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        # Gemini supports system instruction fields, but for stability:
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(url, headers=headers, params=params, json=payload)
                r.raise_for_status()
                data = r.json()
            text = _extract_gemini_text(data)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LLMReply(
                provider="gemini",
                model=self.model,
                text=text,
                latency_ms=latency_ms,
                raw=data,
                usage=data.get("usageMetadata"),
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LLMReply(provider="gemini", model=self.model, text="", latency_ms=latency_ms, raw=None, error=str(e))
