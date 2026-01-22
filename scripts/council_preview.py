import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_council.runner import build_clients_with_roster
from llm_council.role_planner import ask_for_roles, plan_roles
from llm_council.council_engine import (
    CouncilState,
    generate_drafts,
    generate_peer_reviews,
    revise_solutions,
    judge_solutions,
)


def short(s: str, n: int = 350) -> str:
    s = s or ""
    return s[:n] + ("..." if len(s) > n else "")


async def main():
    problem = " ".join(sys.argv[1:]).strip() or input("Problem statement: ").strip()

    clients, roster = build_clients_with_roster()
    client_by_provider = {m.provider: c for c, m in zip(clients, roster)}

    # Phase 0: plan
    opinions = await ask_for_roles(problem_statement=problem, clients=clients, roster=roster)
    plan = plan_roles(roster=roster, opinions=opinions)

    print(f"\nDefault judge: {plan.judge.provider} | {plan.judge.model}")
    print("Solvers:", ", ".join([s.provider for s in plan.solvers]))

    state = CouncilState(problem_statement=problem, plan=plan)

    # Phase 1: drafts
    print("\n[Phase 1] Drafting...")
    await generate_drafts(
        state=state,
        client_by_provider=client_by_provider,
        on_draft=lambda d: print(f"\n--- Draft ready: {d.provider} ---\n{short(d.text)}"),
    )

    # Phase 2: peer reviews
    print("\n[Phase 2] Peer reviews...")
    await generate_peer_reviews(
        state=state,
        client_by_provider=client_by_provider,
        on_review=lambda r: print(
            f"\n--- Review ready: {r.reviewer_provider} -> {r.target_provider} --- "
            + (f"overall={r.parsed.overall}" if r.parsed else f"(invalid JSON: {r.parse_error})")
        ),
    )

    # Phase 3: revisions
    print("\n[Phase 3] Revisions...")
    await revise_solutions(
        state=state,
        client_by_provider=client_by_provider,
        on_revision=lambda rr: print(f"\n--- Revision ready: {rr.provider} ---\n{short(rr.text)}"),
    )

    # Phase 4: judge
    print("\n[Phase 4] Judge decision...")
    await judge_solutions(state=state, client_by_provider=client_by_provider)

    print("\n==============================")
    print("WINNER:", state.winner_provider)
    print("==============================\n")
    print(state.winner_text or "")

    if state.judge and state.judge.parsed:
        print("\n--- Judge rationale ---")
        print(state.judge.parsed.rationale)
    else:
        print("\n(Judge JSON parse failed; fallback winner selection used.)")


if __name__ == "__main__":
    asyncio.run(main())
