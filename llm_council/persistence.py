from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from llm_council.council_engine import (
    CouncilState,
    SolutionResult,
    ReviewResult,
    RevisionResult,
    JudgeResult,
)
from llm_council.roles import ModelInfo
from llm_council.role_planner import SuggestedPlan

SCHEMA_VERSION = 1



# JSON-safe conversion


def to_jsonable(x: Any) -> Any:
    """
    Convert arbitrary Python objects to JSON-serializable structures.
    - dict/list/tuple/set recursively
    - dataclasses via asdict
    - pydantic models via model_dump()
    - fallback to repr()
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    if is_dataclass(x):
        return to_jsonable(asdict(x))

    if isinstance(x, BaseModel):
        return to_jsonable(x.model_dump())

    # common fallback for requests/httpx responses, etc.
    try:
        return to_jsonable(vars(x))
    except Exception:
        return repr(x)



# Serialization helpers


def _ser_model_info(m: ModelInfo) -> dict:
    return {"idx": m.idx, "provider": m.provider, "model": m.model}


def _des_model_info(d: dict) -> ModelInfo:
    return ModelInfo(idx=int(d["idx"]), provider=str(d["provider"]), model=str(d["model"]))


def _ser_plan(plan: SuggestedPlan) -> dict:
    return {
        "judge": _ser_model_info(plan.judge),
        "solvers": [_ser_model_info(s) for s in plan.solvers],
        # opinions/score_breakdown may contain pydantic objects -> make JSON-safe
        "opinions": to_jsonable(getattr(plan, "opinions", {})),
        "score_breakdown": to_jsonable(getattr(plan, "score_breakdown", {})),
    }


def _des_plan(d: dict) -> SuggestedPlan:
    judge = _des_model_info(d["judge"])
    solvers = [_des_model_info(x) for x in d.get("solvers", [])]
    opinions = d.get("opinions", {})
    score_breakdown = d.get("score_breakdown", {})
    return SuggestedPlan(judge=judge, solvers=solvers, opinions=opinions, score_breakdown=score_breakdown)


def _ser_solution(r: SolutionResult) -> dict:
    return {
        "provider": r.provider,
        "model": r.model,
        "text": r.text,
        "error": r.error,
        "raw": to_jsonable(r.raw),
    }


def _des_solution(d: dict) -> SolutionResult:
    return SolutionResult(
        provider=str(d["provider"]),
        model=str(d["model"]),
        text=str(d.get("text", "")),
        error=d.get("error"),
        raw=d.get("raw"),
    )


def _ser_review(r: ReviewResult) -> dict:
    return {
        "reviewer_provider": r.reviewer_provider,
        "target_provider": r.target_provider,
        "raw_text": r.raw_text,
        "attempts": r.attempts,
        "error": r.error,
        "parse_error": r.parse_error,
        "parsed": to_jsonable(r.parsed),
        "raw": to_jsonable(r.raw),
    }


def _des_review(d: dict) -> ReviewResult:
    # parsed is stored as dict; we keep it as dict-y structure (still usable for GUI)
    parsed = d.get("parsed")
    return ReviewResult(
        reviewer_provider=str(d["reviewer_provider"]),
        target_provider=str(d["target_provider"]),
        raw_text=str(d.get("raw_text", "")),
        parsed=parsed,  # type: ignore[assignment]
        parse_error=d.get("parse_error"),
        raw=d.get("raw"),
        attempts=int(d.get("attempts", 1)),
        error=d.get("error"),
    )


def _ser_revision(r: RevisionResult) -> dict:
    return {
        "provider": r.provider,
        "model": r.model,
        "text": r.text,
        "error": r.error,
        "raw": to_jsonable(r.raw),
    }


def _des_revision(d: dict) -> RevisionResult:
    return RevisionResult(
        provider=str(d["provider"]),
        model=str(d["model"]),
        text=str(d.get("text", "")),
        error=d.get("error"),
        raw=d.get("raw"),
    )


def _ser_judge(r: JudgeResult) -> dict:
    return {
        "provider": r.provider,
        "model": r.model,
        "raw_text": r.raw_text,
        "attempts": r.attempts,
        "error": r.error,
        "parse_error": r.parse_error,
        "parsed": to_jsonable(r.parsed),
        "raw": to_jsonable(r.raw),
    }


def _des_judge(d: dict) -> JudgeResult:
    parsed = d.get("parsed")
    return JudgeResult(
        provider=str(d["provider"]),
        model=str(d["model"]),
        raw_text=str(d.get("raw_text", "")),
        parsed=parsed,  # type: ignore[assignment]
        parse_error=d.get("parse_error"),
        raw=d.get("raw"),
        attempts=int(d.get("attempts", 1)),
        error=d.get("error"),
    )



# Public API


def serialize_state(state: CouncilState) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem_statement": state.problem_statement,
        "plan": _ser_plan(state.plan),
        "drafts": {k: _ser_solution(v) for k, v in state.drafts.items()},
        "reviews": [_ser_review(r) for r in state.reviews],
        "revisions": {k: _ser_revision(v) for k, v in state.revisions.items()},
        "judge": _ser_judge(state.judge) if state.judge else None,
        "winner_provider": state.winner_provider,
        "winner_text": state.winner_text,
    }


def deserialize_state(data: dict) -> CouncilState:
    ver = int(data.get("schema_version", 0))
    if ver != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version={ver}. Expected {SCHEMA_VERSION}.")

    plan = _des_plan(data["plan"])
    state = CouncilState(problem_statement=str(data.get("problem_statement", "")), plan=plan)

    drafts = data.get("drafts", {}) or {}
    state.drafts = {str(k): _des_solution(v) for k, v in drafts.items()}

    state.reviews = [_des_review(r) for r in (data.get("reviews", []) or [])]

    revisions = data.get("revisions", {}) or {}
    state.revisions = {str(k): _des_revision(v) for k, v in revisions.items()}

    if data.get("judge") is not None:
        state.judge = _des_judge(data["judge"])

    state.winner_provider = data.get("winner_provider")
    state.winner_text = data.get("winner_text")
    return state


def save_state(state: CouncilState, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_state(state)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_state(path: str | Path) -> CouncilState:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return deserialize_state(data)


def default_run_path(dir_: str | Path = "runs", prefix: str = "council_run") -> Path:
    dirp = Path(dir_)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return dirp / f"{prefix}_{ts}.json"
