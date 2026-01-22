from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

class Role(str, Enum):
    judge = "judge"
    solver = "solver"


@dataclass(frozen=True)
class ModelInfo:
    """
    Immutable identity for a council member model.
    idx is for stable roster display (1..N).
    provider is the short identifier used throughout ("openai", "anthropic", "gemini", "xai", ...).
    model is the provider-specific model string.
    """
    idx: int
    provider: str
    model: str

class RolePreference(BaseModel):
    preferred_role: Role
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=400)


class JudgeRecommendation(BaseModel):
    provider: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=400)


class RoleOpinion(BaseModel):
    """
    One model's opinion about:
      - what role it should take
      - who among the roster should be judge
    """
    self: RolePreference
    recommended_judge: JudgeRecommendation


@dataclass(frozen=True)
class RoleOpinionResult:
    provider: str
    model: str
    raw_text: str
    parsed: Optional[RoleOpinion]
    parse_error: Optional[str]

def extract_first_json_object(text: str) -> str | None:
    """
    Extract the first top-level JSON object from a string.
    Works even if the model wraps JSON with explanations.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_role_opinion(raw_text: str) -> tuple[RoleOpinion | None, str | None]:
    """
    Returns (parsed, error). Never raises.
    """
    # Try direct JSON first
    try:
        data = json.loads(raw_text)
        return RoleOpinion.model_validate(data), None
    except Exception:
        pass

    # Try extracting JSON from a larger blob
    try:
        extracted = extract_first_json_object(raw_text)
        if not extracted:
            return None, "No JSON object found in model output."
        data = json.loads(extracted)
        return RoleOpinion.model_validate(data), None
    except ValidationError as ve:
        return None, f"JSON schema validation error: {ve}"
    except Exception as e:
        return None, f"JSON parse error: {e}"

def roster_to_text(roster: list[ModelInfo]) -> str:
    lines = ["Council roster (choose among these exact providers):"]
    for m in roster:
        lines.append(f"{m.idx}) provider='{m.provider}', model='{m.model}'")
    return "\n".join(lines)


def build_role_opinion_prompts(
    *,
    problem_statement: str,
    roster: list[ModelInfo],
    you: ModelInfo,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).

    Design goals:
      - The model knows the full roster.
      - Output format is strict JSON-only.
      - recommended_judge.provider must be one of roster providers.
    """
    system_prompt = (
        "You are a member of an LLM Council. Exactly 1 model will be the JUDGE and the rest are SOLVERS.\n"
        "Solvers generate solutions. The Judge evaluates multiple solver answers and picks the best.\n"
        "Output MUST be ONLY a single JSON object. No extra text.\n\n"
        f"Your identity: provider='{you.provider}', model='{you.model}', roster_index={you.idx}\n"
    )

    user_prompt = (
        f"{roster_to_text(roster)}\n\n"
        "Task:\n"
        "1) Decide whether YOU should be 'judge' or 'solver' given the roster.\n"
        "2) Recommend which PROVIDER from the roster should be the judge overall.\n\n"
        "Return ONLY JSON with this exact schema (no markdown):\n"
        "{\n"
        '  "self": {"preferred_role": "judge" | "solver", "confidence": 0.0-1.0, "reason": "short"},\n'
        '  "recommended_judge": {"provider": "<one of the roster providers>", "confidence": 0.0-1.0, "reason": "short"}\n'
        "}\n\n"
        f"Problem statement:\n{problem_statement}\n"
    )

    return system_prompt, user_prompt
