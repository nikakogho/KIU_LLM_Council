from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable
import re

from llm_council.clients.base import LLMClient
from llm_council.roles import (
    ModelInfo,
    Role,
    RoleOpinionResult,
    build_role_opinion_prompts,
    parse_role_opinion,
)


@dataclass(frozen=True)
class SuggestedPlan:
    judge: ModelInfo
    solvers: list[ModelInfo]

    # For UI/debugging
    opinions: dict[str, RoleOpinionResult]          # provider -> result
    score_breakdown: dict[str, dict[str, float]]    # provider -> components

DEFAULT_JUDGE_PRIOR = {
    "openai": 1.00,
    "anthropic": 0.95,
    "gemini": 0.90,
    "xai": 0.85,
}

# How much "nominations" contribute relative to priors + self-signal
NOMINATION_WEIGHT = 0.70


def _stable_key(m: ModelInfo) -> tuple:
    return (m.provider, m.model, m.idx)

def _normalize_provider(raw: str, allowed: set[str]) -> str | None:
    """
    Accept exact provider, or common sloppy outputs like:
      "provider='openai', model='gpt-5-nano'"
    Returns normalized provider or None if it can't be mapped safely.
    """
    if not raw:
        return None
    s = raw.strip()
    if s in allowed:
        return s

    tokens = re.findall(r"[a-zA-Z0-9_]+", s.lower())
    for t in tokens:
        if t in allowed:
            return t
    return None


async def ask_for_roles(
    *,
    problem_statement: str,
    clients: Iterable[LLMClient],
    roster: list[ModelInfo],
) -> dict[str, RoleOpinionResult]:
    """
    Ask each model for:
      - its preferred role
      - who it thinks should be judge among the roster
    Returns provider -> RoleOpinionResult.
    """
    clients = list(clients)
    if len(clients) != len(roster):
        raise ValueError(f"clients ({len(clients)}) must align with roster ({len(roster)}).")

    async def one(client: LLMClient, you: ModelInfo):
        system_prompt, user_prompt = build_role_opinion_prompts(
            problem_statement=problem_statement,
            roster=roster,
            you=you,
        )
        reply = await client.generate(user_prompt, system_prompt=system_prompt)
        parsed, err = parse_role_opinion(reply.text)

        # Post-validate / normalize recommended_judge.provider against roster
        allowed = {m.provider for m in roster}
        if parsed is not None:
            norm = _normalize_provider(parsed.recommended_judge.provider, allowed)
            if norm is None:
                parsed = None
                err = (err or "") + f" | Invalid recommended_judge.provider='{parsed.recommended_judge.provider}'"
            elif norm != parsed.recommended_judge.provider:
                parsed = parsed.model_copy(update={
                    "recommended_judge": parsed.recommended_judge.model_copy(update={"provider": norm})
                })

        return you.provider, RoleOpinionResult(
            provider=you.provider,
            model=you.model,
            raw_text=reply.text,
            parsed=parsed,
            parse_error=err or reply.error,
        )

    pairs = await asyncio.gather(*[one(c, m) for c, m in zip(clients, roster)])
    return {p: r for p, r in pairs}


def plan_roles(
    *,
    roster: list[ModelInfo],
    opinions: dict[str, RoleOpinionResult],
    judge_prior: dict[str, float] | None = None,
) -> SuggestedPlan:
    """
    Deterministic default judge selection.

    Score components for candidate provider P:
      - prior(P)
      - self-signal(P): does P claim it should judge (with confidence)
      - nominations(P): other models nominating P (weighted, with nominator prior)

    Fallbacks:
      - If opinions missing/invalid for all, pick highest prior.
      - Deterministic tie-break using (provider, model, idx).
    """
    prior = judge_prior or DEFAULT_JUDGE_PRIOR
    by_provider = {m.provider: m for m in roster}

    breakdown: dict[str, dict[str, float]] = {
        m.provider: {
            "prior": prior.get(m.provider, 0.80),
            "self": 0.0,
            "nominations": 0.0,
        }
        for m in roster
    }

    # self component
    for m in roster:
        r = opinions.get(m.provider)
        if not r or not r.parsed:
            continue
        pref = r.parsed.self.preferred_role
        conf = r.parsed.self.confidence
        # Self-claim of judge matters, but isn't absolute.
        self_factor = 1.0 if pref == Role.judge else 0.55
        breakdown[m.provider]["self"] += self_factor * conf

    # nomination component
    for nominator in roster:
        r = opinions.get(nominator.provider)
        if not r or not r.parsed:
            continue

        rec = r.parsed.recommended_judge
        target = rec.provider.strip()

        # Only count nominations to known providers in roster
        if target not in breakdown:
            continue

        nominator_weight = prior.get(nominator.provider, 0.80)
        breakdown[target]["nominations"] += NOMINATION_WEIGHT * rec.confidence * nominator_weight

    # final score
    scored: list[tuple[float, tuple, str]] = []
    for m in roster:
        b = breakdown[m.provider]
        # This formula is intentionally boring + stable; tweak later via experiments.
        score = b["prior"] * (0.60 + 0.40 * b["self"]) + b["nominations"]
        scored.append((score, _stable_key(m), m.provider))

    scored.sort(reverse=True)
    judge_provider = scored[0][2]
    judge = by_provider[judge_provider]
    solvers = [m for m in roster if m.provider != judge_provider]

    return SuggestedPlan(
        judge=judge,
        solvers=solvers,
        opinions=opinions,
        score_breakdown=breakdown,
    )


def apply_user_override(plan: SuggestedPlan, *, judge_provider: str) -> SuggestedPlan:
    """
    UI layer (CLI/GUI) uses this AFTER showing the default plan and asking the user.
    """
    if judge_provider == plan.judge.provider:
        return plan

    roster = [plan.judge] + plan.solvers
    by_provider = {m.provider: m for m in roster}
    if judge_provider not in by_provider:
        raise ValueError(f"Unknown judge_provider '{judge_provider}'. Must be one of: {list(by_provider.keys())}")

    new_judge = by_provider[judge_provider]
    new_solvers = [m for m in roster if m.provider != judge_provider]

    return SuggestedPlan(
        judge=new_judge,
        solvers=new_solvers,
        opinions=plan.opinions,
        score_breakdown=plan.score_breakdown,
    )
