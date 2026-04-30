from __future__ import annotations

import numpy as np

from judicator._types import JudgeType
from judicator.biases.base import (
    BiasResult,
    BiasTest,
    parse_fail_result,
    resolve_judge_type_from_judge,
    severity_from_score,
)
from judicator.cost import CallCounter
from judicator.judge import Judge
from judicator.parsers import parse_binary, parse_pointwise

# Theoretical max variance for scores on a 1-10 scale (50/50 split between 1 and 10)
_MAX_SCORE_VARIANCE = 20.25  # (9 / 2) ** 2

_N_RUNS = 5


class SelfConsistencyTest(BiasTest):
    name = "self_consistency"
    # Pairwise requires pairs; self_consistency fixture is single-response — deferred to v0.2
    applicable_types = [JudgeType.POINTWISE, JudgeType.BINARY]
    fixture_path = "universal/self_consistency.jsonl"

    def run(
        self,
        judge: Judge,
        fixtures: list[dict],
        call_counter: CallCounter,
    ) -> BiasResult:
        jt = resolve_judge_type_from_judge(judge)
        if jt == JudgeType.PAIRWISE:
            return self.not_applicable_result(
                "self_consistency fixture is single-response; "
                "pairwise judges need response pairs (deferred to v0.2)"
            )
        if jt == JudgeType.BINARY:
            return self._run_binary(judge, fixtures, call_counter)
        return self._run_pointwise(judge, fixtures, call_counter)

    # ── pointwise ──────────────────────────────────────────────────────────────

    def _run_pointwise(
        self, judge: Judge, fixtures: list[dict], call_counter: CallCounter
    ) -> BiasResult:
        variances: list[float] = []

        for item in fixtures:
            prompt = _format(judge.eval_template, item)
            if prompt is None:
                continue
            scores = []
            for _ in range(_N_RUNS):
                s = parse_pointwise(judge.llm_fn(prompt))
                call_counter.increment()
                if s is not None:
                    scores.append(s)
            if len(scores) >= 2:
                variances.append(float(np.var(scores)))

        if not variances:
            return parse_fail_result(self.name)

        mean_var = float(np.mean(variances))
        score = round(1.0 - float(np.clip(mean_var / _MAX_SCORE_VARIANCE, 0.0, 1.0)), 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=len(variances),
            details={
                "mean_score_variance": round(mean_var, 4),
                "n_runs_per_item": _N_RUNS,
            },
            severity=severity_from_score(score),
        )

    # ── binary ─────────────────────────────────────────────────────────────────

    def _run_binary(
        self, judge: Judge, fixtures: list[dict], call_counter: CallCounter
    ) -> BiasResult:
        consistencies: list[float] = []

        for item in fixtures:
            prompt = _format(judge.eval_template, item)
            if prompt is None:
                continue
            verdicts: list[str] = []
            for _ in range(_N_RUNS):
                v = parse_binary(judge.llm_fn(prompt))
                call_counter.increment()
                if v is not None:
                    verdicts.append(v)
            if len(verdicts) >= 2:
                majority = max(set(verdicts), key=verdicts.count)
                consistencies.append(verdicts.count(majority) / len(verdicts))

        if not consistencies:
            return parse_fail_result(self.name)

        mean_consistency = float(np.mean(consistencies))
        score = round(mean_consistency, 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=len(consistencies),
            details={
                "mean_consistency": round(mean_consistency, 3),
                "n_runs_per_item": _N_RUNS,
            },
            severity=severity_from_score(score),
        )


# ── helper ─────────────────────────────────────────────────────────────────────

import re as _re


def _format(template: str, item: dict) -> str | None:
    """Format template with item fields; return None if a required placeholder is missing."""
    kwargs = {
        "question": item.get("question", ""),
        "response": item.get("response", ""),
        "statement": item.get("response", ""),
    }
    try:
        keys = set(_re.findall(r"\{(\w+)\}", template))
        return template.format(**{k: v for k, v in kwargs.items() if k in keys})
    except KeyError:
        return None
