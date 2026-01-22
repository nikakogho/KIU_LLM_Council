import pytest

from llm_council.clients.base import LLMClient
from llm_council.types import LLMReply
from llm_council.roles import ModelInfo
from llm_council.role_planner import SuggestedPlan
from llm_council.council_engine import (
    CouncilState,
    generate_drafts,
    generate_peer_reviews,
    revise_solutions,
    judge_solutions,
)


class FakeClient(LLMClient):
    def __init__(self, provider: str, model: str, *, break_first_review: bool = False, break_first_judge: bool = False):
        self.provider = provider
        self.model = model
        self.break_first_review = break_first_review
        self.break_first_judge = break_first_judge
        self._review_calls = 0
        self._judge_calls = 0

    async def generate(self, user_prompt: str, *, system_prompt: str | None = None) -> LLMReply:
        sp = system_prompt or ""

        if "COUNCIL_PHASE: SOLVE" in sp:
            return LLMReply(self.provider, self.model, f"{self.provider} draft", 1, raw={})

        if "COUNCIL_PHASE: REVIEW" in sp or "COUNCIL_PHASE: REVIEW_REPAIR" in sp:
            self._review_calls += 1

            # Force first review call to be non-JSON for this client (only if enabled)
            if self.break_first_review and self._review_calls == 1 and "REVIEW_REPAIR" not in sp:
                return LLMReply(self.provider, self.model, "oops not json", 1, raw={})

            # Extract target
            marker = "target provider='"
            target = user_prompt.split(marker, 1)[1].split("'", 1)[0] if marker in user_prompt else "unknown"
            overall = 9 if target == "openai" else 6
            text = (
                "{"
                f"\"reviewer_provider\":\"{self.provider}\","
                f"\"target_provider\":\"{target}\","
                "\"correctness\":7,\"completeness\":7,\"clarity\":7,\"feasibility\":7,"
                f"\"overall\":{overall},"
                "\"key_flaws\":[\"x\"],\"suggested_fixes\":[\"y\"],\"summary\":\"ok\""
                "}"
            )
            return LLMReply(self.provider, self.model, text, 1, raw={})

        if "COUNCIL_PHASE: REFINE" in sp:
            return LLMReply(self.provider, self.model, f"{self.provider} revised", 1, raw={})

        if "COUNCIL_PHASE: JUDGE" in sp or "COUNCIL_PHASE: JUDGE_REPAIR" in sp:
            self._judge_calls += 1
            if self.break_first_judge and self._judge_calls == 1 and "JUDGE_REPAIR" not in sp:
                return LLMReply(self.provider, self.model, "NOT JSON", 1, raw={})

            text = (
                "{"
                "\"winner_provider\":\"openai\","
                "\"ranking\":["
                "{\"provider\":\"openai\",\"score\":9,\"reason\":\"best\"},"
                "{\"provider\":\"anthropic\",\"score\":7,\"reason\":\"ok\"},"
                "{\"provider\":\"gemini\",\"score\":6,\"reason\":\"ok\"}"
                "],"
                "\"rationale\":\"openai is best\""
                "}"
            )
            return LLMReply(self.provider, self.model, text, 1, raw={})

        return LLMReply(self.provider, self.model, "unknown", 1, raw={})


def make_plan():
    judge = ModelInfo(idx=1, provider="xai", model="grok-3-mini")
    solvers = [
        ModelInfo(idx=2, provider="openai", model="gpt-5-nano"),
        ModelInfo(idx=3, provider="anthropic", model="claude-haiku-4-5"),
        ModelInfo(idx=4, provider="gemini", model="gemini-2.5-flash-lite"),
    ]
    return SuggestedPlan(judge=judge, solvers=solvers, opinions={}, score_breakdown={})


@pytest.mark.asyncio
async def test_review_json_retry_recovers():
    plan = make_plan()
    state = CouncilState(problem_statement="test", plan=plan)

    client_by_provider = {
        "xai": FakeClient("xai", "grok-3-mini"),
        "openai": FakeClient("openai", "gpt-5-nano"),
        # Make gemini fail FIRST review attempt (so we exercise retry)
        "anthropic": FakeClient("anthropic", "claude-haiku-4-5"),
        "gemini": FakeClient("gemini", "gemini-2.5-flash-lite", break_first_review=True),
    }

    await generate_drafts(state=state, client_by_provider=client_by_provider)
    await generate_peer_reviews(state=state, client_by_provider=client_by_provider)

    # At least one review should have attempts=2 and still be parsed
    retried = [r for r in state.reviews if r.attempts == 2]
    assert retried, "Expected at least one retried review."
    assert any(r.parsed is not None and r.parse_error is None for r in retried)


@pytest.mark.asyncio
async def test_judge_json_retry_recovers():
    plan = make_plan()
    state = CouncilState(problem_statement="test", plan=plan)

    client_by_provider = {
        "xai": FakeClient("xai", "grok-3-mini", break_first_judge=True),  # judge fails once
        "openai": FakeClient("openai", "gpt-5-nano"),
        "anthropic": FakeClient("anthropic", "claude-haiku-4-5"),
        "gemini": FakeClient("gemini", "gemini-2.5-flash-lite"),
    }

    await generate_drafts(state=state, client_by_provider=client_by_provider)
    await generate_peer_reviews(state=state, client_by_provider=client_by_provider)
    await revise_solutions(state=state, client_by_provider=client_by_provider)
    await judge_solutions(state=state, client_by_provider=client_by_provider)

    assert state.judge is not None
    assert state.judge.attempts == 2
    assert state.judge.parsed is not None
    assert state.winner_provider == "openai"
