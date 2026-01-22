# llm_council_cli.py
import argparse
import asyncio
from pathlib import Path

from llm_council.app import plan_council, run_council, RunCallbacks
from llm_council.role_planner import apply_user_override
from llm_council.persistence import save_state, default_run_path
from llm_council.runner import build_client_by_provider


def _print_plan(plan):
    print("\n=== Default plan ===")
    print(f"Judge:  {plan.judge.provider}  ({plan.judge.model})")
    print("Solvers:")
    for s in plan.solvers:
        print(f"  - {s.provider} ({s.model})")


def _print_opinions(plan):
    # plan.opinions: provider -> RoleOpinionResult
    print("\n=== Role opinions ===")
    for prov, res in plan.opinions.items():
        if res.parsed:
            self_role = res.parsed.self.preferred_role
            self_conf = res.parsed.self.confidence
            rec = res.parsed.recommended_judge
            print(
                f"- {prov:10} self={self_role} conf={self_conf:.2f} | "
                f"recommends={rec.provider} conf={rec.confidence:.2f}"
            )
        else:
            print(f"- {prov:10} (unparsed) err={res.parse_error}")

    print("\n=== Judge score breakdown (components) ===")
    for prov, b in plan.score_breakdown.items():
        print(f"- {prov:10} prior={b.get('prior',0):.2f} self={b.get('self',0):.2f} nominations={b.get('nominations',0):.2f}")


def _choose_override_interactive(plan) -> str | None:
    roster = [plan.judge] + plan.solvers
    providers = [m.provider for m in roster]

    print("\nOverride judge? Press Enter to keep default.")
    print("Options:", ", ".join(providers))
    ans = input("judge provider> ").strip()
    if not ans:
        return None
    if ans not in providers:
        print(f"Invalid '{ans}'. Keeping default.")
        return None
    return ans


def _make_callbacks(verbose: bool) -> RunCallbacks:
    def on_phase_start(name: str):
        print(f"\n=== PHASE: {name} ===")

    def on_phase_end(name: str):
        print(f"=== DONE:  {name} ===")

    def on_draft(d):
        if not verbose:
            print(f"[draft] {d.provider}")
            return
        status = "OK" if not d.error else f"ERR: {d.error}"
        print(f"[draft] {d.provider:10} {status}  chars={len(d.text or '')}")

    def on_review(r):
        ok = (r.parsed is not None and r.parse_error is None)
        if not verbose:
            print(f"[review] {r.reviewer_provider}->{r.target_provider} {'OK' if ok else 'BAD_JSON'}")
            return
        status = "OK" if ok else f"BAD_JSON ({r.parse_error})"
        print(f"[review] {r.reviewer_provider:10} -> {r.target_provider:10} {status} attempts={r.attempts}")

    def on_revision(r):
        if not verbose:
            print(f"[revise] {r.provider}")
            return
        status = "OK" if not r.error else f"ERR: {r.error}"
        print(f"[revise] {r.provider:10} {status}  chars={len(r.text or '')}")

    return RunCallbacks(
        on_phase_start=on_phase_start,
        on_phase_end=on_phase_end,
        on_draft=on_draft,
        on_review=on_review,
        on_revision=on_revision,
    )


def _print_judge_details(state):
    j = state.judge
    if not j:
        print("Judge: (missing)")
        return

    parsed_ok = (j.parsed is not None) and (j.parse_error is None)
    print("\n=== Judge details ===")
    print(f"Judge provider: {j.provider} ({j.model})")
    print(f"Attempts: {j.attempts}")
    print(f"Parsed JSON: {'YES' if parsed_ok else 'NO'}")
    if j.parse_error:
        print(f"Parse error: {j.parse_error}")

    # Determine whether we used judge decision or fallback
    used_fallback = True
    if parsed_ok:
        try:
            wp = j.parsed.winner_provider  # pydantic model in-memory
            used_fallback = (wp != state.winner_provider)
        except Exception:
            pass

    print(f"Used fallback: {'YES' if used_fallback else 'NO'}")


async def _run(problem: str, args) -> int:
    clients, roster, client_by_provider = build_client_by_provider()
    if len(roster) < 2:
        print("Need at least 2 configured providers to run a council.")
        return 2

    # Role planning ONCE
    print("\n=== PHASE: role_planning ===")
    plan = await plan_council(problem_statement=problem, clients=clients, roster=roster)
    print("=== DONE:  role_planning ===")

    _print_plan(plan)
    if args.show_opinions:
        _print_opinions(plan)

    # Apply override WITHOUT re-asking models
    override = args.judge
    if override is None and not args.no_interactive:
        override = _choose_override_interactive(plan)

    if override:
        plan = apply_user_override(plan, judge_provider=override)
        print("\n=== Using overridden judge ===")
        print(f"Judge:  {plan.judge.provider}  ({plan.judge.model})")

    # Run council with streaming callbacks
    callbacks = _make_callbacks(verbose=args.verbose)
    state = await run_council(
        problem_statement=problem,
        plan=plan,
        client_by_provider=client_by_provider,
        callbacks=callbacks,
        do_reviews=not args.no_reviews,
        do_revise=not args.no_revise,
    )

    # Save
    out_path = Path(args.out) if args.out else default_run_path()
    save_state(state, out_path)

    # Print winner
    print("\n=== WINNER ===")
    print(f"Winner provider: {state.winner_provider}")
    print(f"Saved run: {out_path}")

    if args.show_judge:
        _print_judge_details(state)

    if args.print_winner:
        print("\n--- Winner text ---\n")
        print(state.winner_text or "")

    return 0


def main():
    ap = argparse.ArgumentParser(description="LLM Council CLI")
    ap.add_argument("problem", help="Problem statement text")
    ap.add_argument("--judge", default=None, help="Override judge provider (openai/anthropic/gemini/xai)")
    ap.add_argument("--out", default=None, help="Path to save run JSON (default: runs/council_run_<ts>.json)")

    ap.add_argument("--no-reviews", action="store_true", help="Skip peer review phase")
    ap.add_argument("--no-revise", action="store_true", help="Skip revision phase")
    ap.add_argument("--no-interactive", action="store_true", help="Do not prompt for judge override interactively")

    ap.add_argument("--verbose", action="store_true", help="Print more per-item details")
    ap.add_argument("--print-winner", action="store_true", help="Print winner text at the end")
    ap.add_argument("--show-opinions", action="store_true", help="Print role opinions + judge scoring breakdown components")
    ap.add_argument("--show-judge", action="store_true", help="Print judge parse/attempts + whether fallback was used")

    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args.problem, args)))


if __name__ == "__main__":
    main()
