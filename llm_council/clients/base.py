from abc import ABC, abstractmethod
from llm_council.types import LLMReply

class LLMClient(ABC):
    @abstractmethod
    async def generate(self, user_prompt: str, *, system_prompt: str | None = None) -> LLMReply:
        ...
