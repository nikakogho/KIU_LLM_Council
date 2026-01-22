import pytest
import respx
from llm_council.clients.anthropic import AnthropicClient

@pytest.mark.asyncio
@respx.mock
async def test_anthropic_parsing():
    mock = {
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    respx.post("https://api.anthropic.com/v1/messages").respond(200, json=mock)

    c = AnthropicClient(api_key="test", model="claude-haiku-4-5")
    r = await c.generate("hi")
    assert r.error is None
    assert r.text == "hello"
