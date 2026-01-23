from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from llm_council.clients.base import LLMClient
from llm_council.roles import ModelInfo, extract_first_json_object
from llm_council.role_planner import SuggestedPlan
from llm_council.types import LLMReply



# JSON schemas


class PeerReviewJSON(BaseModel):
    reviewer_provider: str
    target_provider: str

    correctness: int = Field(ge=0, le=10)
    completeness: int = Field(ge=0, le=10)
    clarity: int = Field(ge=0, le=10)
    feasibility: int = Field(ge=0, le=10)
    overall: int = Field(ge=0, le=10)

    key_flaws: list[str] = Field(default_factory=list, max_length=8)
    suggested_fixes: list[str] = Field(default_factory=list, max_length=8)
    summary: str = Field(min_length=1, max_length=600)


class JudgeDecisionJSON(BaseModel):
    winner_provider: str
    ranking: list[dict[str, Any]]  # [{"provider": "...", "score": 0-10, "reason":"..."}]
    rationale: str = Field(min_length=1, max_length=1200)


T = TypeVar("T", bound=BaseModel)


def parse_json_with_schema(raw_text: str, schema: Type[T]) -> tuple[T | None, str | None]:
    """
    Parse JSON possibly wrapped in extra text. Never raises.
    Returns (parsed_or_none, error_or_none).
    """
    try:
        data = json.loads(raw_text)
        return schema.model_validate(data), None
    except Exception:
        pass

    try:
        extracted = extract_first_json_object(raw_text)
        if not extracted:
            return None, "No JSON object found."
        data = json.loads(extracted)
        return schema.model_validate(data), None
    except ValidationError as ve:
        return None, f"Schema validation error: {ve}"
    except Exception as e:
        return None, f"JSON parse error: {e}"


def _clip(text: str, n: int = 900) -> str:
    text = text or ""
    return text[:n] + ("..." if len(text) > n else "")



# Result records


@dataclass(frozen=True)
class SolutionResult:
    provider: str
    model: str
    text: str
    raw: Any
    error: Optional[str] = None


@dataclass(frozen=True)
class ReviewResult:
    reviewer_provider: str
    target_provider: str
    raw_text: str
    parsed: PeerReviewJSON | None
    parse_error: str | None
    raw: Any
    attempts: int = 1
    error: str | None = None


@dataclass(frozen=True)
class RevisionResult:
    provider: str
    model: str
    text: str
    raw: Any
    error: Optional[str] = None


@dataclass(frozen=True)
class JudgeResult:
    provider: str
    model: str
    raw_text: str
    parsed: JudgeDecisionJSON | None
    parse_error: str | None
    raw: Any
    attempts: int = 1
    error: str | None = None


@dataclass
class CouncilState:
    """
    Mutable container for step-by-step phases (ideal for CLI/GUI progress updates).
    """
    problem_statement: str
    plan: SuggestedPlan

    drafts: dict[str, SolutionResult] = field(default_factory=dict)
    reviews: list[ReviewResult] = field(default_factory=list)
    revisions: dict[str, RevisionResult] = field(default_factory=dict)
    judge: JudgeResult | None = None

    winner_provider: str | None = None
    winner_text: str | None = None



# Prompt builders


def _roster_text(plan: SuggestedPlan) -> str:
    roster = [plan.judge] + plan.solvers
    lines = ["Council roster:"]
    for m in roster:
        lines.append(f"- provider='{m.provider}', model='{m.model}'")
    return "\n".join(lines)


_STRICT_JSON_RULES = (
    "Return ONLY a single JSON object.\n"
    "Do NOT include any extra text.\n"
    "Do NOT use markdown.\n"
    "Do NOT wrap in backticks or ```.\n"
    "The output must start with '{' and end with '}'.\n"
)


def build_solver_prompts(problem_statement: str, plan: SuggestedPlan) -> tuple[str, str]:
    system_prompt = (
        "COUNCIL_PHASE: SOLVE\n"
        "You are a SOLVER in an LLM Council.\n"
        "Write a reasonably structured practical solution in under 400 words.\n"
        "Do not mention you are an AI or that this is a council.\n"
    )
    user_prompt = (
        f"{_roster_text(plan)}\n\n"
        f"Problem statement:\n{problem_statement}\n\n"
        "Deliverable:\n"
        "- A clear solution with steps.\n"
    )
    return system_prompt, user_prompt


def build_review_prompts(
    problem_statement: str,
    plan: SuggestedPlan,
    reviewer: ModelInfo,
    target: ModelInfo,
    target_solution: str,
) -> tuple[str, str, str]:
    """
    Returns (system_prompt, user_prompt, schema_text)
    """
    schema_text = (
        "{\n"
        '  "reviewer_provider": "<your provider>",\n'
        '  "target_provider": "<target provider>",\n'
        '  "correctness": 0-10,\n'
        '  "completeness": 0-10,\n'
        '  "clarity": 0-10,\n'
        '  "feasibility": 0-10,\n'
        '  "overall": 0-10,\n'
        '  "key_flaws": ["..."],\n'
        '  "suggested_fixes": ["..."],\n'
        '  "summary": "short"\n'
        "}\n"
    )

    system_prompt = (
        "COUNCIL_PHASE: REVIEW\n"
        "You are a SOLVER doing peer review of another solver's draft.\n"
        + _STRICT_JSON_RULES
    )

    user_prompt = (
        f"{_roster_text(plan)}\n\n"
        f"Problem statement:\n{problem_statement}\n\n"
        f"You are reviewer provider='{reviewer.provider}', model='{reviewer.model}'.\n"
        f"You must review target provider='{target.provider}', model='{target.model}'.\n\n"
        "Target draft:\n"
        "-----\n"
        f"{target_solution}\n"
        "-----\n\n"
        "Output JSON with this exact schema:\n"
        f"{schema_text}"
    )

    return system_prompt, user_prompt, schema_text


def build_refine_prompts(
    problem_statement: str,
    plan: SuggestedPlan,
    solver: ModelInfo,
    draft: str,
    reviews_about_solver: list[ReviewResult],
) -> tuple[str, str]:
    system_prompt = (
        "COUNCIL_PHASE: REFINE\n"
        "You are a SOLVER. Improve your draft using the peer feedback.\n"
        "Write the revised solution only (no JSON), and do it in under 400 words.\n"
        "Do not mention the council.\n"
    )

    feedback_lines = []
    for r in reviews_about_solver:
        if r.parsed:
            feedback_lines.append(
                f"- From {r.reviewer_provider}: overall={r.parsed.overall}, flaws={r.parsed.key_flaws}, fixes={r.parsed.suggested_fixes}"
            )
        else:
            feedback_lines.append(f"- From {r.reviewer_provider}: (unparsed) {_clip(r.raw_text, 250)}")

    user_prompt = (
        f"{_roster_text(plan)}\n\n"
        f"Problem statement:\n{problem_statement}\n\n"
        f"Your identity: provider='{solver.provider}', model='{solver.model}'.\n\n"
        "Your draft:\n"
        "-----\n"
        f"{draft}\n"
        "-----\n\n"
        "Peer feedback:\n"
        + ("\n".join(feedback_lines) if feedback_lines else "- (no peer feedback)\n")
        + "\n\n"
        "Now output the improved final solution.\n"
    )

    return system_prompt, user_prompt


def build_judge_prompts(
    problem_statement: str,
    plan: SuggestedPlan,
    final_solutions: dict[str, str],
) -> tuple[str, str, str]:
    """
    Returns (system_prompt, user_prompt, schema_text)
    """
    schema_text = (
        "{\n"
        '  "winner_provider": "<one of the solver providers>",\n'
        '  "ranking": [ {"provider":"...", "score":0-10, "reason":"short"}, ... ],\n'
        '  "rationale": "short explanation of why the winner is best"\n'
        "}\n"
    )

    system_prompt = (
        "COUNCIL_PHASE: JUDGE\n"
        "You are the JUDGE in an LLM Council.\n"
        "Compare final solutions and pick the best.\n"
        + _STRICT_JSON_RULES
    )

    blocks = []
    for p in plan.solvers:
        txt = final_solutions.get(p.provider, "")
        blocks.append(f"=== provider='{p.provider}' model='{p.model}' ===\n{txt}\n")

    user_prompt = (
        f"{_roster_text(plan)}\n\n"
        f"Problem statement:\n{problem_statement}\n\n"
        "Final solutions:\n\n"
        + "\n".join(blocks)
        + "\nOutput JSON with this exact schema:\n"
        + schema_text
    )

    return system_prompt, user_prompt, schema_text



# Utility: gather tasks and emit as they complete


async def _gather_as_completed(tasks: list[asyncio.Task], on_item: Callable[[Any], None] | None):
    results = []
    for fut in asyncio.as_completed(tasks):
        item = await fut
        results.append(item)
        if on_item:
            on_item(item)
    return results



# Phase functions


async def generate_drafts(
    *,
    state: CouncilState,
    client_by_provider: dict[str, LLMClient],
    on_draft: Callable[[SolutionResult], None] | None = None,
) -> CouncilState:
    """
    Phase 1: each solver generates a draft (parallel).
    Calls on_draft(draft) as each completes.
    """
    sys_p, user_p = build_solver_prompts(state.problem_statement, state.plan)

    async def one(solver: ModelInfo) -> SolutionResult:
        rep: LLMReply = await client_by_provider[solver.provider].generate(user_p, system_prompt=sys_p)
        return SolutionResult(
            provider=solver.provider,
            model=solver.model,
            text=rep.text,
            raw=rep.raw,
            error=rep.error,
        )

    tasks = [asyncio.create_task(one(s)) for s in state.plan.solvers]
    drafts_list = await _gather_as_completed(tasks, on_draft)
    state.drafts = {d.provider: d for d in drafts_list}
    return state


async def generate_peer_reviews(
    *,
    state: CouncilState,
    client_by_provider: dict[str, LLMClient],
    on_review: Callable[[ReviewResult], None] | None = None,
) -> CouncilState:
    """
    Phase 2: each solver reviews every other solver (parallel).
    Calls on_review(review) as each completes.
    Requires drafts already present in state.
    """
    if not state.drafts:
        raise ValueError("generate_peer_reviews requires state.drafts. Run generate_drafts first.")

    async def one(reviewer: ModelInfo, target: ModelInfo) -> ReviewResult:
        target_text = state.drafts.get(target.provider).text if state.drafts.get(target.provider) else ""
        sys_p, user_p, schema_text = build_review_prompts(
            state.problem_statement, state.plan, reviewer, target, target_text
        )
        client = client_by_provider[reviewer.provider]

        # Attempt 1
        rep1 = await client.generate(user_p, system_prompt=sys_p)
        parsed1, err1 = parse_json_with_schema(rep1.text, PeerReviewJSON)

        if parsed1 is not None and err1 is None:
            # Do not trust model fields: overwrite with ground truth.
            parsed1 = parsed1.model_copy(update={
                "reviewer_provider": reviewer.provider,
                "target_provider": target.provider,
            })
            return ReviewResult(
                reviewer_provider=reviewer.provider,
                target_provider=target.provider,
                raw_text=rep1.text,
                parsed=parsed1,
                parse_error=None,
                raw=rep1.raw,
                attempts=1,
                error=rep1.error,
            )

        # Attempt 2 (repair)
        repair_sys = (
            "COUNCIL_PHASE: REVIEW_REPAIR\n"
            "Your previous output was invalid.\n"
            + _STRICT_JSON_RULES
        )
        repair_user = (
            f"Previous output (invalid):\n{_clip(rep1.text)}\n\n"
            "Fix it now. Output ONLY JSON with this exact schema:\n"
            f"{schema_text}"
        )

        rep2 = await client.generate(repair_user, system_prompt=repair_sys)
        parsed2, err2 = parse_json_with_schema(rep2.text, PeerReviewJSON)

        if parsed2 is not None and err2 is None:
            # Do not trust model fields: overwrite with ground truth.
            parsed2 = parsed2.model_copy(update={
                "reviewer_provider": reviewer.provider,
                "target_provider": target.provider,
            })
            return ReviewResult(
                reviewer_provider=reviewer.provider,
                target_provider=target.provider,
                raw_text=rep2.text,
                parsed=parsed2,
                parse_error=None,
                raw=rep2.raw,
                attempts=2,
                error=rep2.error,
            )

        # Still failed
        return ReviewResult(
            reviewer_provider=reviewer.provider,
            target_provider=target.provider,
            raw_text=rep2.text,
            parsed=None,
            parse_error=err2 or err1,
            raw=rep2.raw,
            attempts=2,
            error=rep2.error or rep1.error,
        )

    tasks: list[asyncio.Task] = []
    solvers = state.plan.solvers
    for reviewer in solvers:
        for target in solvers:
            if reviewer.provider != target.provider:
                tasks.append(asyncio.create_task(one(reviewer, target)))

    state.reviews = await _gather_as_completed(tasks, on_review)
    return state


async def revise_solutions(
    *,
    state: CouncilState,
    client_by_provider: dict[str, LLMClient],
    on_revision: Callable[[RevisionResult], None] | None = None,
) -> CouncilState:
    """
    Phase 3: each solver revises their draft using reviews (parallel).
    Calls on_revision(revision) as each completes.
    Requires drafts already present. Reviews optional.
    """
    if not state.drafts:
        raise ValueError("revise_solutions requires state.drafts. Run generate_drafts first.")

    async def one(solver: ModelInfo) -> RevisionResult:
        my_reviews = [r for r in state.reviews if r.target_provider == solver.provider]
        sys_p, user_p = build_refine_prompts(
            state.problem_statement,
            state.plan,
            solver,
            state.drafts[solver.provider].text,
            my_reviews,
        )
        rep = await client_by_provider[solver.provider].generate(user_p, system_prompt=sys_p)
        text = rep.text.strip() if (rep.text or "").strip() else state.drafts[solver.provider].text
        return RevisionResult(
            provider=solver.provider,
            model=solver.model,
            text=text,
            raw=rep.raw,
            error=rep.error,
        )

    tasks = [asyncio.create_task(one(s)) for s in state.plan.solvers]
    revision_list = await _gather_as_completed(tasks, on_revision)
    state.revisions = {r.provider: r for r in revision_list}
    return state


async def judge_solutions(
    *,
    state: CouncilState,
    client_by_provider: dict[str, LLMClient],
) -> CouncilState:
    """
    Phase 4: judge picks winner among revised solutions.
    Falls back deterministically if judge JSON is invalid or winner_provider is invalid.
    Retries once if judge JSON is invalid.
    """
    solutions = (
        {p: r.text for p, r in state.revisions.items()}
        if state.revisions
        else {p: d.text for p, d in state.drafts.items()}
    )

    sys_p, user_p, schema_text = build_judge_prompts(state.problem_statement, state.plan, solutions)
    judge_client = client_by_provider[state.plan.judge.provider]

    # Attempt 1
    rep1 = await judge_client.generate(user_p, system_prompt=sys_p)
    parsed1, err1 = parse_json_with_schema(rep1.text, JudgeDecisionJSON)

    parsed = parsed1
    err = err1
    rep = rep1
    attempts = 1

    # Attempt 2 (repair) if invalid
    if parsed is None or err is not None:
        repair_sys = (
            "COUNCIL_PHASE: JUDGE_REPAIR\n"
            "Your previous output was invalid.\n"
            + _STRICT_JSON_RULES
        )
        repair_user = (
            f"Previous output (invalid):\n{_clip(rep1.text)}\n\n"
            "Fix it now. Output ONLY JSON with this exact schema:\n"
            f"{schema_text}"
        )
        rep2 = await judge_client.generate(repair_user, system_prompt=repair_sys)
        parsed2, err2 = parse_json_with_schema(rep2.text, JudgeDecisionJSON)
        parsed = parsed2
        err = err2
        rep = rep2
        attempts = 2

    state.judge = JudgeResult(
        provider=state.plan.judge.provider,
        model=state.plan.judge.model,
        raw_text=rep.text,
        parsed=parsed,
        parse_error=err,
        raw=rep.raw,
        attempts=attempts,
        error=rep.error,
    )

    solver_providers = [s.provider for s in state.plan.solvers]

    # If judge parsed and winner is valid, accept it.
    if state.judge.parsed and state.judge.parsed.winner_provider in solver_providers:
        wp = state.judge.parsed.winner_provider
        if wp in solutions:
            state.winner_provider = wp
            state.winner_text = solutions[wp]
            return state

    # Deterministic fallback:
    # Average parsed review "overall" per target; if none, first solver in plan order.
    avg: dict[str, float] = {p: 0.0 for p in solver_providers}
    cnt: dict[str, int] = {p: 0 for p in solver_providers}

    for r in state.reviews:
        if r.parsed and r.target_provider in avg:
            avg[r.target_provider] += float(r.parsed.overall)
            cnt[r.target_provider] += 1

    scored = [(avg[p] / cnt[p], p) for p in solver_providers if cnt[p] > 0]
    winner = max(scored)[1] if scored else solver_providers[0]

    state.winner_provider = winner
    state.winner_text = solutions[winner]
    return state
