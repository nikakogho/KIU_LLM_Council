import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_council.persistence import save_state
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


async def main_async(args: argparse.Namespace) -> int:
    problem = (args.problem or "").strip()
    if not problem:
        problem = input("Problem statement: ").strip()

    clients, roster = build_clients_with_roster()
    if not clients or not roster:
        print("No clients available (missing API keys?).")
        return 2

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
        on_draft=lambda d: print(f"\n--- Draft ready: {d.provider} ---\n{short(d.text, args.preview_chars)}"),
    )

    # Phase 2: peer reviews
    print("\n[Phase 2] Peer reviews...")
    def _print_review(r):
        if r.parsed and not r.parse_error:
            # keep it compact but informative
            flaws = ", ".join((r.parsed.key_flaws or [])[:2])
            fixes = ", ".join((r.parsed.suggested_fixes or [])[:2])
            extra = []
            if flaws:
                extra.append(f"flaws=[{flaws}]")
            if fixes:
                extra.append(f"fixes=[{fixes}]")
            if r.attempts == 2:
                extra.append("retried")
            suffix = (" | " + " | ".join(extra)) if extra else ""
            print(f"\n--- Review ready: {r.reviewer_provider} -> {r.target_provider} --- overall={r.parsed.overall}{suffix}")
        else:
            retry_note = " (retried)" if r.attempts == 2 else ""
            print(f"\n--- Review ready: {r.reviewer_provider} -> {r.target_provider} --- invalid JSON: {r.parse_error}{retry_note}")

    await generate_peer_reviews(
        state=state,
        client_by_provider=client_by_provider,
        on_review=_print_review,
    )

    # Phase 3: revisions
    print("\n[Phase 3] Revisions...")
    await revise_solutions(
        state=state,
        client_by_provider=client_by_provider,
        on_revision=lambda rr: print(f"\n--- Revision ready: {rr.provider} ---\n{short(rr.text, args.preview_chars)}"),
    )

    # Phase 4: judge
    print("\n[Phase 4] Judge decision...")
    await judge_solutions(state=state, client_by_provider=client_by_provider)

    print("\n==============================")
    print("WINNER:", state.winner_provider)
    print("==============================\n")
    print(state.winner_text or "")

    if state.judge and state.judge.parsed and not state.judge.parse_error:
        print("\n--- Judge rationale ---")
        # parsed is a pydantic model in-memory; if you later load from JSON it will be dict (handled in replay)
        print(state.judge.parsed.rationale)
    else:
        print("\n(Judge JSON parse failed; fallback winner selection used.)")

    # Save artifact
    if args.out:
        out_path = Path(args.out)
        # If user passes a directory-like path without suffix, append .json
        if out_path.suffix.lower() != ".json":
            out_path = out_path.with_suffix(".json")
        p = save_state(state, out_path)
        print(f"\nSaved run: {p}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the LLM Council end-to-end and optionally save the run artifact.")
    ap.add_argument(
        "problem",
        nargs="?",
        default="",
        help="Problem statement (if omitted, you will be prompted).",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Path to save run JSON (e.g., runs/experiment1.json). If omitted, no save.",
    )
    ap.add_argument(
        "--preview-chars",
        type=int,
        default=350,
        help="How many characters to preview for drafts/revisions in the console.",
    )
    args = ap.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
