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
from judicator.parsers import parse_binary, parse_pairwise, parse_pointwise

_SCALE_RANGE = 9.0  # 10 - 1 for a 1-10 scoring scale


class VerbosityBiasTest(BiasTest):
    name = "verbosity"
    applicable_types = [JudgeType.POINTWISE, JudgeType.PAIRWISE, JudgeType.BINARY]
    fixture_path = "{domain}/verbosity.jsonl"

    def run(
        self,
        judge: Judge,
        fixtures: list[dict],
        call_counter: CallCounter,
    ) -> BiasResult:
        jt = resolve_judge_type_from_judge(judge)
        if jt == JudgeType.PAIRWISE:
            return self._run_pairwise(judge, fixtures, call_counter)
        if jt == JudgeType.BINARY:
            return self._run_binary(judge, fixtures, call_counter)
        return self._run_pointwise(judge, fixtures, call_counter)

    # ── pointwise ──────────────────────────────────────────────────────────────

    def _run_pointwise(
        self, judge: Judge, fixtures: list[dict], call_counter: CallCounter
    ) -> BiasResult:
        inflations: list[float] = []
        failures: list[dict] = []

        for item in fixtures:
            q = item["question"]
            s_orig = parse_pointwise(judge.llm_fn(
                judge.eval_template.format(question=q, response=item["original_response"])
            ))
            call_counter.increment()
            s_manip = parse_pointwise(judge.llm_fn(
                judge.eval_template.format(question=q, response=item["manipulated_response"])
            ))
            call_counter.increment()

            if s_orig is None or s_manip is None:
                continue

            inflations.append(s_manip - s_orig)
            if s_manip > s_orig and len(failures) < 3:
                failures.append({
                    "question": q[:120],
                    "original_score": s_orig,
                    "padded_score": s_manip,
                    "inflation": round(s_manip - s_orig, 2),
                })

        if not inflations:
            return parse_fail_result(self.name)

        mean_inflation = float(np.mean(inflations))
        fraction_inflated = sum(1 for x in inflations if x > 0) / len(inflations)
        score = round(1.0 - float(np.clip(mean_inflation / _SCALE_RANGE * 2, 0.0, 1.0)), 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=len(inflations),
            details={
                "mean_inflation": round(mean_inflation, 3),
                "fraction_inflated": round(fraction_inflated, 3),
            },
            examples=failures,
            severity=severity_from_score(score),
        )

    # ── pairwise ───────────────────────────────────────────────────────────────

    def _run_pairwise(
        self, judge: Judge, fixtures: list[dict], call_counter: CallCounter
    ) -> BiasResult:
        padded_wins = 0
        total_picks = 0
        failures: list[dict] = []

        for item in fixtures:
            q = item["question"]
            orig = item["original_response"]
            manip = item["manipulated_response"]

            # AB: orig=A, manip=B — padded wins if judge picks B
            v_AB = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(question=q, response_a=orig, response_b=manip)
            ))
            call_counter.increment()
            # BA: manip=A, orig=B — padded wins if judge picks A
            v_BA = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(question=q, response_a=manip, response_b=orig)
            ))
            call_counter.increment()

            if v_AB is not None:
                total_picks += 1
                if v_AB == "B":
                    padded_wins += 1
            if v_BA is not None:
                total_picks += 1
                if v_BA == "A":
                    padded_wins += 1

            if v_AB == "B" and v_BA == "A" and len(failures) < 3:
                failures.append({"question": q[:120], "verdict_AB": v_AB, "verdict_BA": v_BA})

        if total_picks == 0:
            return parse_fail_result(self.name)

        bias_rate = padded_wins / total_picks
        inflation = max(0.0, bias_rate - 0.5)
        score = round(1.0 - float(np.clip(inflation * 2, 0.0, 1.0)), 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=total_picks // 2,
            details={
                "padded_win_rate": round(bias_rate, 3),
                "inflation": round(inflation, 3),
            },
            examples=failures,
            severity=severity_from_score(score),
        )

    # ── binary ─────────────────────────────────────────────────────────────────

    def _run_binary(
        self, judge: Judge, fixtures: list[dict], call_counter: CallCounter
    ) -> BiasResult:
        yes_orig = 0
        yes_manip = 0
        n = 0

        for item in fixtures:
            q = item["question"]
            common = {"question": q, "statement": item["original_response"],
                      "response": item["original_response"]}
            r_orig = parse_binary(judge.llm_fn(judge.eval_template.format(**_pick(judge.eval_template, common))))
            call_counter.increment()

            common["statement"] = item["manipulated_response"]
            common["response"] = item["manipulated_response"]
            r_manip = parse_binary(judge.llm_fn(judge.eval_template.format(**_pick(judge.eval_template, common))))
            call_counter.increment()

            if r_orig is None or r_manip is None:
                continue

            n += 1
            if r_orig == "Yes":
                yes_orig += 1
            if r_manip == "Yes":
                yes_manip += 1

        if n == 0:
            return parse_fail_result(self.name)

        rate_orig = yes_orig / n
        rate_manip = yes_manip / n
        inflation = max(0.0, rate_manip - rate_orig)
        score = round(1.0 - float(np.clip(inflation, 0.0, 1.0)), 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=n,
            details={
                "yes_rate_original": round(rate_orig, 3),
                "yes_rate_manipulated": round(rate_manip, 3),
                "inflation": round(inflation, 3),
            },
            severity=severity_from_score(score),
        )


# ── helpers ────────────────────────────────────────────────────────────────────

import re as _re


def _pick(template: str, kwargs: dict) -> dict:
    """Return only the kwargs whose keys appear as placeholders in template."""
    keys = set(_re.findall(r"\{(\w+)\}", template))
    return {k: v for k, v in kwargs.items() if k in keys}
