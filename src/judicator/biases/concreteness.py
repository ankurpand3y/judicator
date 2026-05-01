from __future__ import annotations

from judicator._types import JudgeType
from judicator.biases.base import (
    BiasResult,
    BiasTest,
    parallel_map,
    parse_fail_result,
    resolve_judge_type_from_judge,
    severity_from_score,
)
from judicator.cost import CallCounter
from judicator.judge import Judge
from judicator.parsers import parse_pairwise, parse_pointwise


class ConcretenessBiasTest(BiasTest):
    """Measures whether the judge inflates scores for responses with fabricated specifics.

    Uses natural pairs from OffsetBias: vague_response (correct/preferred) vs
    concrete_response (specifics-loaded). n=14 — signal is coarse.
    """
    name = "concreteness"
    applicable_types = [JudgeType.POINTWISE, JudgeType.PAIRWISE]
    fixture_path = "universal/concreteness.jsonl"

    def run(
        self,
        judge: Judge,
        fixtures: list[dict],
        call_counter: CallCounter,
        max_workers: int = 1,
    ) -> BiasResult:
        jt = resolve_judge_type_from_judge(judge)
        if jt == JudgeType.PAIRWISE:
            return self._run_pairwise(judge, fixtures, call_counter, max_workers)
        return self._run_pointwise(judge, fixtures, call_counter, max_workers)

    def _run_pointwise(
        self, judge: Judge, fixtures: list[dict], call_counter: CallCounter, max_workers: int
    ) -> BiasResult:
        def score_one(item: dict) -> tuple[float, float, str] | None:
            q = item.get("question", "")
            s_vague = parse_pointwise(judge.llm_fn(
                judge.eval_template.format(question=q, response=item["vague_response"])
            ))
            call_counter.increment()
            s_concrete = parse_pointwise(judge.llm_fn(
                judge.eval_template.format(question=q, response=item["concrete_response"])
            ))
            call_counter.increment()
            if s_vague is None or s_concrete is None:
                return None
            return (s_vague, s_concrete, q)

        results = parallel_map(score_one, fixtures, max_workers)

        calibrated = 0
        n = 0
        failures: list[dict] = []
        for r in results:
            if r is None:
                continue
            s_vague, s_concrete, q = r
            n += 1
            if s_vague >= s_concrete:
                calibrated += 1
            elif len(failures) < 3:
                failures.append({
                    "question": q[:120],
                    "vague_score": s_vague,
                    "concrete_score": s_concrete,
                    "inflation": round(s_concrete - s_vague, 2),
                })

        if n == 0:
            return parse_fail_result(self.name)

        score = round(calibrated / n, 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=n,
            details={
                "fraction_calibrated": score,
                "note": "n=14 fixture; signal is coarse",
            },
            examples=failures,
            severity=severity_from_score(score),
        )

    def _run_pairwise(
        self, judge: Judge, fixtures: list[dict], call_counter: CallCounter, max_workers: int
    ) -> BiasResult:
        def score_one(item: dict) -> tuple[str, str, str] | None:
            q = item.get("question", "")
            vague = item["vague_response"]
            concrete = item["concrete_response"]
            v_AB = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(question=q, response_a=vague, response_b=concrete)
            ))
            call_counter.increment()
            v_BA = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(question=q, response_a=concrete, response_b=vague)
            ))
            call_counter.increment()
            if v_AB is None or v_BA is None:
                return None
            return (v_AB, v_BA, q)

        results = parallel_map(score_one, fixtures, max_workers)

        calibrated = 0
        n = 0
        failures: list[dict] = []
        for r in results:
            if r is None:
                continue
            v_AB, v_BA, q = r
            n += 1
            if v_AB == "A" and v_BA == "B":
                calibrated += 1
            elif len(failures) < 3:
                failures.append({
                    "question": q[:120],
                    "verdict_vague_in_A": v_AB,
                    "verdict_vague_in_B": v_BA,
                })

        if n == 0:
            return parse_fail_result(self.name)

        score = round(calibrated / n, 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=n,
            details={
                "fraction_calibrated": score,
                "note": "n=14 fixture; signal is coarse",
            },
            examples=failures,
            severity=severity_from_score(score),
        )
