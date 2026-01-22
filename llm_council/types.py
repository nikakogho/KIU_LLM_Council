from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class LLMReply:
    provider: str
    model: str
    text: str
    latency_ms: int
    raw: Any
    usage: Optional[dict] = None
    error: Optional[str] = None
