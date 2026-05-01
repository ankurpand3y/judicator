from __future__ import annotations

import numpy as np

from judicator._types import JudgeType
from judicator.biases.base import (
    BiasResult,
    BiasTest,
    parallel_map,
    parse_fail_result,
    severity_from_score,
)
from judicator.cost import CallCounter
from judicator.judge import Judge
from judicator.parsers import parse_pointwise

# A calibrated judge should use ~70% of the scale range between high/low quality items.
_SCALE_RANGE = 9.0          # 10 - 1
_EXPECTED_SPREAD_FRAC = 0.7
_EXPECTED_SPREAD = _SCALE_RANGE * _EXPECTED_SPREAD_FRAC  # 6.3


class ScaleAnchoringTest(BiasTest):
    name = "scale_anchoring"
    applicable_types = [JudgeType.POINTWISE]
    fixture_path = "universal/scale_tiers.jsonl"

    def run(
        self,
        judge: Judge,
        fixtures: list[dict],
        call_counter: CallCounter,
        max_workers: int = 1,
    ) -> BiasResult:
        def score_one(item: dict) -> tuple[float, str] | None:
            prompt = _format(judge.eval_template, item)
            if prompt is None:
                return None
            s = parse_pointwise(judge.llm_fn(prompt))
            call_counter.increment()
            if s is None:
                return None
            return (s, item["tier"])

        results = parallel_map(score_one, fixtures, max_workers)

        high_scores: list[float] = []
        low_scores: list[float] = []
        failures: list[dict] = []
        for r in results:
            if r is None:
                continue
            s, tier = r
            if tier == "high":
                high_scores.append(s)
            else:
                low_scores.append(s)

        if not high_scores or not low_scores:
            return parse_fail_result(self.name)

        high_mean = float(np.mean(high_scores))
        low_mean = float(np.mean(low_scores))
        spread = high_mean - low_mean

        score = round(float(np.clip(spread / _EXPECTED_SPREAD, 0.0, 1.0)), 3)

        if spread < _EXPECTED_SPREAD and len(failures) < 3:
            failures = [
                {"note": f"spread={spread:.2f} below expected {_EXPECTED_SPREAD:.2f}",
                 "high_mean": round(high_mean, 2), "low_mean": round(low_mean, 2)}
            ]

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=len(high_scores) + len(low_scores),
            details={
                "high_tier_mean": round(high_mean, 3),
                "low_tier_mean": round(low_mean, 3),
                "spread": round(spread, 3),
                "expected_spread": _EXPECTED_SPREAD,
            },
            examples=failures,
            severity=severity_from_score(score),
        )


import re as _re


def _format(template: str, item: dict) -> str | None:
    kwargs = {
        "question": item.get("question", ""),
        "response": item.get("response", ""),
        "statement": item.get("response", ""),
    }
    keys = set(_re.findall(r"\{(\w+)\}", template))
    if not keys.issubset(kwargs.keys()):
        return None
    return template.format(**{k: v for k, v in kwargs.items() if k in keys})
