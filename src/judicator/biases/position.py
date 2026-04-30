from __future__ import annotations

from judicator._types import JudgeType
from judicator.biases.base import (
    BiasResult,
    BiasTest,
    parse_fail_result,
    severity_from_score,
)
from judicator.cost import CallCounter
from judicator.judge import Judge
from judicator.parsers import parse_pairwise


class PositionBiasTest(BiasTest):
    name = "position"
    applicable_types = [JudgeType.PAIRWISE]
    fixture_path = "{domain}/position.jsonl"

    def run(
        self,
        judge: Judge,
        fixtures: list[dict],
        call_counter: CallCounter,
    ) -> BiasResult:
        inconsistent = 0
        slot_a_picks = 0
        usable = 0
        failures: list[dict] = []

        for item in fixtures:
            q = item["question"]
            winner = item["winner_response"]
            loser = item["loser_response"]

            # Run 1 — winner in slot A; calibrated answer is "A"
            v_AB = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(
                    question=q, response_a=winner, response_b=loser
                )
            ))
            call_counter.increment()

            # Run 2 — winner in slot B; calibrated answer is "B"
            v_BA = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(
                    question=q, response_a=loser, response_b=winner
                )
            ))
            call_counter.increment()

            if v_AB is None or v_BA is None:
                continue

            usable += 1
            if v_AB == "A":
                slot_a_picks += 1
            if v_BA == "A":
                slot_a_picks += 1

            # Inconsistency: same slot letter in both orderings = judge ignored content
            if v_AB == v_BA:
                inconsistent += 1
                if len(failures) < 3:
                    failures.append({
                        "question": q[:120],
                        "verdict_winner_in_A": v_AB,
                        "verdict_winner_in_B": v_BA,
                    })

        if usable == 0:
            return parse_fail_result(self.name)

        inconsistency_rate = inconsistent / usable
        score = round(1.0 - inconsistency_rate, 3)

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=usable,
            details={
                "inconsistency_rate": round(inconsistency_rate, 3),
                "slot_a_pick_rate": round(slot_a_picks / (2 * usable), 3),
            },
            examples=failures,
            severity=severity_from_score(score),
        )
