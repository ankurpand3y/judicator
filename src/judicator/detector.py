from __future__ import annotations
import re

from judicator._types import JudgeType
from judicator.exceptions import DetectionError

_BINARY_KEYWORDS = [
    "yes or no",
    "true or false",
    "answer only yes",
    "answer only no",
    "respond with yes",
    "respond with no",
]

_RESPONSE_SLOTS = {"response", "answer", "output", "completion"}


def detect_judge_type(eval_template: str) -> tuple[JudgeType, float]:
    """Return (JudgeType, confidence). Confidence in [0, 1]."""
    placeholders = set(re.findall(r"\{(\w+)\}", eval_template))

    if {"response_a", "response_b"}.issubset(placeholders):
        return JudgeType.PAIRWISE, 0.95

    lower = eval_template.lower()
    if any(kw in lower for kw in _BINARY_KEYWORDS):
        return JudgeType.BINARY, 0.90

    if _RESPONSE_SLOTS & placeholders:
        return JudgeType.POINTWISE, 0.85

    return JudgeType.UNKNOWN, 0.0


def resolve_judge_type(eval_template: str, override: str | None) -> JudgeType:
    """Resolve judge type from override or auto-detection. Raises DetectionError if ambiguous."""
    if override:
        return JudgeType(override)
    judge_type, confidence = detect_judge_type(eval_template)
    if confidence < 0.8:
        raise DetectionError(
            f"Could not detect judge type from eval_template (confidence={confidence:.2f}). "
            "Pass judge_type='pointwise', 'pairwise', or 'binary' to Judge()."
        )
    return judge_type
