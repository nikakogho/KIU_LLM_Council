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
    def __init__(self, provider: str, model: str, judge_bad_json: bool = False):
        self.provider = provider
        self.model = model
        self.judge_bad_json = judge_bad_json

    async def generate(self, user_prompt: str, *, system_prompt: str | None = None) -> LLMReply:
        sp = system_prompt or ""

        if "COUNCIL_PHASE: SOLVE" in sp:
            return LLMReply(self.provider, self.model, f"{self.provider} draft", 1, raw={})

        if "COUNCIL_PHASE: REVIEW" in sp:
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

        if "COUNCIL_PHASE: JUDGE" in sp:
            if self.judge_bad_json:
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
async def test_phases_are_separable_and_callbacks_fire():
    plan = make_plan()
    state = CouncilState(problem_statement="test", plan=plan)

    client_by_provider = {
        "xai": FakeClient("xai", "grok-3-mini"),
        "openai": FakeClient("openai", "gpt-5-nano"),
        "anthropic": FakeClient("anthropic", "claude-haiku-4-5"),
        "gemini": FakeClient("gemini", "gemini-2.5-flash-lite"),
    }

    draft_cb = []
    review_cb = []
    rev_cb = []

    await generate_drafts(state=state, client_by_provider=client_by_provider, on_draft=lambda d: draft_cb.append(d))
    assert len(state.drafts) == 3
    assert len(draft_cb) == 3

    await generate_peer_reviews(state=state, client_by_provider=client_by_provider, on_review=lambda r: review_cb.append(r))
    assert len(state.reviews) == 6
    assert len(review_cb) == 6

    await revise_solutions(state=state, client_by_provider=client_by_provider, on_revision=lambda r: rev_cb.append(r))
    assert len(state.revisions) == 3
    assert len(rev_cb) == 3

    await judge_solutions(state=state, client_by_provider=client_by_provider)
    assert state.winner_provider == "openai"
    assert state.winner_text == "openai revised"
    assert state.judge is not None
    assert state.judge.parsed is not None


@pytest.mark.asyncio
async def test_judge_bad_json_falls_back_to_review_scores():
    plan = make_plan()
    state = CouncilState(problem_statement="test", plan=plan)

    client_by_provider = {
        "xai": FakeClient("xai", "grok-3-mini", judge_bad_json=True),
        "openai": FakeClient("openai", "gpt-5-nano"),
        "anthropic": FakeClient("anthropic", "claude-haiku-4-5"),
        "gemini": FakeClient("gemini", "gemini-2.5-flash-lite"),
    }

    await generate_drafts(state=state, client_by_provider=client_by_provider)
    await generate_peer_reviews(state=state, client_by_provider=client_by_provider)
    await revise_solutions(state=state, client_by_provider=client_by_provider)
    await judge_solutions(state=state, client_by_provider=client_by_provider)

    # reviews in FakeClient rate openai highest -> fallback should pick openai
    assert state.winner_provider == "openai"
