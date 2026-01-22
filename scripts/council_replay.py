import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_council.persistence import load_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to saved council run JSON")
    args = ap.parse_args()

    state = load_state(args.path)

    print("=== REPLAY ===")
    print("Problem:", state.problem_statement)
    print()
    print("Judge:", f"{state.plan.judge.provider} | {state.plan.judge.model}")
    print("Solvers:", ", ".join([f"{s.provider}|{s.model}" for s in state.plan.solvers]))
    print()

    print(f"Drafts: {len(state.drafts)}")
    print(f"Reviews: {len(state.reviews)}")
    print(f"Revisions: {len(state.revisions)}")
    print("Judge decision present:", state.judge is not None)
    print()

    print("Winner:", state.winner_provider)
    print()
    if state.winner_text:
        print("--- Winner text (first 600 chars) ---")
        print(state.winner_text[:600] + ("..." if len(state.winner_text) > 600 else ""))

    # quick review stats
    parsed_ok = sum(1 for r in state.reviews if r.parsed and not r.parse_error)
    retries = sum(1 for r in state.reviews if getattr(r, "attempts", 1) == 2)
    print()
    print(f"Reviews parsed ok: {parsed_ok}/{len(state.reviews)}")
    print(f"Reviews retried: {retries}")


if __name__ == "__main__":
    main()
