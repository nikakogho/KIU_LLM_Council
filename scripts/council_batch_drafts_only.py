"""
council_batch_drafts_only.py

Runs the LLM Council over each question in a dataset, but SKIPS:
- peer reviews
- revisions

Flow per question:
  plan_council -> run_council(do_reviews=False, do_revise=False)

Outputs:
- per-question run JSONs under --run-dir
- aggregate summary JSON at --out
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_council.runner import build_client_by_provider
from llm_council.app import plan_council, run_council, RunCallbacks
from llm_council.persistence import save_state


def load_problem_list(dataset_path: Path) -> list[dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # support {"problems":[...]} or similar
        for key in ("problems", "items", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Dataset JSON must be a list of problems or an object containing a list under a known key.")


def pick_question(problem: dict[str, Any]) -> str:
    # support a few common keys
    for k in ("question", "prompt", "problem", "text"):
        if k in problem and isinstance(problem[k], str) and problem[k].strip():
            return problem[k].strip()
    # fallback: stringify
    return str(problem)


def pick_answer(problem: dict[str, Any]) -> str:
    for k in ("answer", "gold", "correct_answer", "solution"):
        if k in problem and isinstance(problem[k], str):
            return problem[k]
    return ""


def make_callbacks(question_idx: int, total: int, *, verbose: bool) -> RunCallbacks:
    if not verbose:
        return RunCallbacks()

    def on_phase_start(name: str):
        print(f"    -> {name}")

    def on_phase_end(name: str):
        print(f"    <- {name}")

    def on_draft(d):
        print(f"       draft done: {d.provider}")

    def on_review(r):
        # should not happen in drafts-only mode
        print(f"       review done: {r.reviewer_provider} -> {r.target_provider}")

    def on_revision(rv):
        # should not happen in drafts-only mode
        print(f"       revision done: {rv.provider}")

    return RunCallbacks(
        on_phase_start=on_phase_start,
        on_phase_end=on_phase_end,
        on_draft=on_draft,
        on_review=on_review,
        on_revision=on_revision,
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "datasets" / "problem_dataset.json"),
        help="Path to dataset JSON (list of {question, answer}).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "results" / "council_drafts_only_results.json"),
        help="Aggregate output JSON path.",
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default=str(REPO_ROOT / "runs" / "batch_drafts_only"),
        help="Directory to save per-question run JSONs.",
    )
    ap.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Optional judge override provider (openai|anthropic|gemini|xai) if present in roster.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print phase-by-phase progress.",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    problems = load_problem_list(dataset_path)
    print(f"Loaded {len(problems)} problems from {dataset_path}")

    clients, roster, client_by_provider = build_client_by_provider()
    if len(roster) < 2:
        print("Error: Need at least 2 configured providers (API keys) to run a council.")
        return 1

    print(f"Using {len(roster)} providers:")
    for m in roster:
        print(f"  - {m.provider} ({m.model})")

    results: list[dict[str, Any]] = []
    total = len(problems)

    for i, prob in enumerate(problems, 1):
        question = pick_question(prob)
        gold = pick_answer(prob)

        print(f"\n[{i}/{total}] Planning + running (drafts-only)...")

        try:
            plan = await plan_council(
                problem_statement=question,
                clients=clients,
                roster=roster,
                judge_override=args.judge,
            )

            callbacks = make_callbacks(i, total, verbose=args.verbose)

            state = await run_council(
                problem_statement=question,
                plan=plan,
                client_by_provider=client_by_provider,
                callbacks=callbacks,
                do_reviews=False,
                do_revise=False,
            )

            run_path = run_dir / f"drafts_only_q{i:03d}.json"
            save_state(state, run_path)

            results.append(
                {
                    "idx": i,
                    "question": question,
                    "correct_answer": gold,
                    "mode": "drafts_only",
                    "plan": {
                        "judge": {"provider": plan.judge.provider, "model": plan.judge.model},
                        "solvers": [{"provider": s.provider, "model": s.model} for s in plan.solvers],
                    },
                    "winner_provider": state.winner_provider,
                    "winner_text": state.winner_text,
                    "run_json": str(run_path.relative_to(REPO_ROOT)),
                    "judge_parse_error": getattr(state.judge, "parse_error", None) if state.judge else None,
                }
            )

            print(f"  Winner: {state.winner_provider} | saved: {run_path}")

        except Exception as e:
            results.append(
                {
                    "idx": i,
                    "question": question,
                    "correct_answer": gold,
                    "mode": "drafts_only",
                    "error": str(e),
                }
            )
            print(f"  ERROR on problem {i}: {e}")

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAggregate results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
