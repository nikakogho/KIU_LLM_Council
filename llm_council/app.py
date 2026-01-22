# llm_council/app.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

from llm_council.clients.base import LLMClient
from llm_council.roles import ModelInfo
from llm_council.role_planner import (
    SuggestedPlan,
    ask_for_roles,
    plan_roles,
    apply_user_override,
)
from llm_council.council_engine import (
    CouncilState,
    SolutionResult,
    ReviewResult,
    RevisionResult,
    generate_drafts,
    generate_peer_reviews,
    revise_solutions,
    judge_solutions,
)


@dataclass(frozen=True)
class RunCallbacks:
    """
    Shared callback bundle used by BOTH CLI and GUI.
    CLI prints to console; GUI appends to widgets.
    """
    on_phase_start: Optional[Callable[[str], None]] = None
    on_phase_end: Optional[Callable[[str], None]] = None

    on_draft: Optional[Callable[[SolutionResult], None]] = None
    on_review: Optional[Callable[[ReviewResult], None]] = None
    on_revision: Optional[Callable[[RevisionResult], None]] = None


async def plan_council(
    *,
    problem_statement: str,
    clients: list[LLMClient],
    roster: list[ModelInfo],
    judge_override: str | None = None,
    judge_prior: dict[str, float] | None = None,
) -> SuggestedPlan:
    """
    Shared role planning used by BOTH CLI and GUI.
    - Ask all models for role opinions
    - Compute deterministic default plan
    - Apply optional user override AFTER default is computed
    """
    opinions = await ask_for_roles(problem_statement=problem_statement, clients=clients, roster=roster)
    plan = plan_roles(roster=roster, opinions=opinions, judge_prior=judge_prior)

    if judge_override:
        plan = apply_user_override(plan, judge_provider=judge_override)

    return plan


async def run_council(
    *,
    problem_statement: str,
    plan: SuggestedPlan,
    client_by_provider: dict[str, LLMClient],
    callbacks: RunCallbacks | None = None,
    do_reviews: bool = True,
    do_revise: bool = True,
) -> CouncilState:
    """
    Shared orchestration used by BOTH CLI and GUI.
    Emits phase start/end signals so UI can show "what step is running now".
    Streams drafts/reviews/revisions via existing per-item callbacks.
    """
    cb = callbacks or RunCallbacks()
    state = CouncilState(problem_statement=problem_statement, plan=plan)

    def phase_start(name: str):
        if cb.on_phase_start:
            cb.on_phase_start(name)

    def phase_end(name: str):
        if cb.on_phase_end:
            cb.on_phase_end(name)

    phase_start("generate_drafts")
    await generate_drafts(state=state, client_by_provider=client_by_provider, on_draft=cb.on_draft)
    phase_end("generate_drafts")

    if do_reviews:
        phase_start("generate_peer_reviews")
        await generate_peer_reviews(state=state, client_by_provider=client_by_provider, on_review=cb.on_review)
        phase_end("generate_peer_reviews")

    if do_revise:
        phase_start("revise_solutions")
        await revise_solutions(state=state, client_by_provider=client_by_provider, on_revision=cb.on_revision)
        phase_end("revise_solutions")

    phase_start("judge_solutions")
    await judge_solutions(state=state, client_by_provider=client_by_provider)
    phase_end("judge_solutions")

    return state
