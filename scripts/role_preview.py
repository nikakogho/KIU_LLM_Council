import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_council.runner import build_clients_with_roster
from llm_council.role_planner import ask_for_roles, plan_roles


def print_plan(plan):
    print("\n=== DEFAULT PLAN ===")
    print(f"Judge:  {plan.judge.idx}) {plan.judge.provider} | {plan.judge.model}")
    print("Solvers:")
    for m in plan.solvers:
        print(f"  {m.idx}) {m.provider} | {m.model}")

    print("\n=== OPINIONS ===")
    for provider, res in plan.opinions.items():
        print(f"\n[{provider}] model={res.model}")
        if res.parsed:
            s = res.parsed.self
            rj = res.parsed.recommended_judge
            print(f"  self: {s.preferred_role.value} conf={s.confidence:.2f}")
            print(f"  self_reason: {s.reason}")
            print(f"  recommends_judge: {rj.provider} conf={rj.confidence:.2f}")
            print(f"  judge_reason: {rj.reason}")
        else:
            print(f"  INVALID: {res.parse_error}")
            raw = res.raw_text or ""
            print("  raw:", raw[:250] + ("..." if len(raw) > 250 else ""))

    print("\n=== SCORE BREAKDOWN ===")
    for provider, b in plan.score_breakdown.items():
        print(
            f"{provider}: prior={b['prior']:.2f} "
            f"self={b['self']:.2f} nominations={b['nominations']:.2f}"
        )


async def main():
    problem = " ".join(sys.argv[1:]).strip()
    if not problem:
        problem = input("Problem statement: ").strip()

    clients, roster = build_clients_with_roster()
    if not clients:
        raise RuntimeError("No clients configured. Put API keys in .env and try again.")

    opinions = await ask_for_roles(problem_statement=problem, clients=clients, roster=roster)
    plan = plan_roles(roster=roster, opinions=opinions)

    print_plan(plan)


if __name__ == "__main__":
    asyncio.run(main())
