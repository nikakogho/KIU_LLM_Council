import pytest
import respx
from unittest.mock import patch
from llm_council.runner import ask_all

@pytest.mark.asyncio
@respx.mock
async def test_runner_does_not_crash_without_keys():
    # Mock the settings to have no API keys
    with patch("llm_council.runner.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = None
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GEMINI_API_KEY = None
        mock_settings.XAI_API_KEY = None

        replies = await ask_all("hi")
        assert replies == []
