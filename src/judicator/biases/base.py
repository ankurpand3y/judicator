from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from judicator._types import JudgeType
from judicator.cost import CallCounter
from judicator.detector import detect_judge_type
from judicator.judge import Judge


def severity_from_score(score: float) -> str:
    if score < 0.50:
        return "CRITICAL"
    if score < 0.65:
        return "SIGNIFICANT"
    if score < 0.80:
        return "MINOR"
    return "NONE"


def resolve_judge_type_from_judge(judge: Judge) -> JudgeType:
    if judge.judge_type:
        return JudgeType(judge.judge_type)
    jt, _ = detect_judge_type(judge.eval_template)
    return jt


def safe_format(template: str, **kwargs: str) -> str | None:
    """Format template with kwargs. Returns None if any placeholder is missing."""
    placeholders = set(re.findall(r"\{(\w+)\}", template))
    if not placeholders.issubset(kwargs.keys()):
        return None
    return template.format(**{k: v for k, v in kwargs.items() if k in placeholders})


def parse_fail_result(name: str) -> "BiasResult":
    return BiasResult(
        test_name=name,
        score=0.0,
        verdict="FAIL",
        n_fixtures=0,
        severity="CRITICAL",
        details={"error": "all items failed to parse or format"},
    )


@dataclass
class BiasResult:
    test_name: str
    score: float
    verdict: str            # "PASS" | "FAIL" | "N/A"
    n_fixtures: int
    details: dict = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    severity: str | None = None
    rank: int = 0
    not_applicable: bool = False
    skip_reason: str = ""


class BiasTest(ABC):
    name: str
    applicable_types: list[JudgeType]
    fixture_path: str

    @abstractmethod
    def run(
        self,
        judge: Judge,
        fixtures: list[dict],
        call_counter: CallCounter,
    ) -> BiasResult: ...

    def is_applicable(self, judge_type: JudgeType) -> bool:
        return judge_type in self.applicable_types

    def not_applicable_result(self, reason: str) -> BiasResult:
        return BiasResult(
            test_name=self.name,
            score=0.0,
            verdict="N/A",
            n_fixtures=0,
            not_applicable=True,
            skip_reason=reason,
        )
