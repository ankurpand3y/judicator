from __future__ import annotations

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
        max_workers: int = 1,
    ) -> BiasResult:
        def score_one(item: dict) -> tuple[str, str, str] | None:
            q = item["question"]
            winner = item["winner_response"]
            loser = item["loser_response"]
            v_AB = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(question=q, response_a=winner, response_b=loser)
            ))
            call_counter.increment()
            v_BA = parse_pairwise(judge.llm_fn(
                judge.eval_template.format(question=q, response_a=loser, response_b=winner)
            ))
            call_counter.increment()
            if v_AB is None or v_BA is None:
                return None
            return (v_AB, v_BA, q)

        results = parallel_map(score_one, fixtures, max_workers)

        inconsistent = 0
        slot_a_picks = 0
        usable = 0
        failures: list[dict] = []
        for r in results:
            if r is None:
                continue
            v_AB, v_BA, q = r
            usable += 1
            if v_AB == "A":
                slot_a_picks += 1
            if v_BA == "A":
                slot_a_picks += 1
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
