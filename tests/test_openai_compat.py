import pytest
import respx

from llm_council.clients.openai_compat import OpenAICompatibleResponsesClient

@pytest.mark.asyncio
@respx.mock
async def test_openai_responses_parsing():
    mock = {
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "hello"}],
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    respx.post("https://api.openai.com/v1/responses").respond(200, json=mock)

    c = OpenAICompatibleResponsesClient(
        provider="openai",
        api_key="test",
        model="gpt-5-nano",
        base_url="https://api.openai.com/v1",
    )
    r = await c.generate("hi")
    assert r.error is None
    assert r.text == "hello"
    assert r.usage["total_tokens"] == 2
