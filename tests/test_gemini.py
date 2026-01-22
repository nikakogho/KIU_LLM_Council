import pytest
import respx
from llm_council.clients.gemini import GeminiClient

@pytest.mark.asyncio
@respx.mock
async def test_gemini_parsing():
    mock = {
        "candidates": [{
            "content": {"parts": [{"text": "hello"}]}
        }]
    }

    # We match by URL prefix since query param includes ?key=
    respx.post(url__startswith="https://generativelanguage.googleapis.com/v1beta/models/").respond(200, json=mock)

    c = GeminiClient(api_key="test", model="gemini-2.5-flash-lite")
    r = await c.generate("hi")
    assert r.error is None
    assert r.text == "hello"
