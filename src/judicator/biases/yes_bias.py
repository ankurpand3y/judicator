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
from judicator.parsers import parse_binary


class YesBiasTest(BiasTest):
    """Measures whether a binary judge is biased toward answering Yes.

    Uses balanced T/F statements (50 True, 50 False). Score = balanced accuracy.
    A judge that always says Yes or always says No scores 0.5.
    """
    name = "yes_bias"
    applicable_types = [JudgeType.BINARY]
    fixture_path = "universal/yesno_probes.jsonl"

    def run(
        self,
        judge: Judge,
        fixtures: list[dict],
        call_counter: CallCounter,
    ) -> BiasResult:
        correct = 0
        fp = 0   # said Yes to a False statement
        fn = 0   # said No to a True statement
        n_true = 0
        n_false = 0
        n_usable = 0
        failures: list[dict] = []

        for item in fixtures:
            expected = "Yes" if item["ground_truth"] else "No"
            try:
                prompt = judge.eval_template.format(statement=item["statement"])
            except KeyError:
                continue  # template doesn't use {statement}; skip item
            verdict = parse_binary(judge.llm_fn(prompt))
            call_counter.increment()

            if verdict is None:
                continue

            n_usable += 1
            if item["ground_truth"]:
                n_true += 1
                if verdict == "No":
                    fn += 1
                else:
                    correct += 1
            else:
                n_false += 1
                if verdict == "Yes":
                    fp += 1
                else:
                    correct += 1

            if verdict != expected and len(failures) < 3:
                failures.append({
                    "statement": item["statement"][:120],
                    "expected": expected,
                    "got": verdict,
                })

        if n_usable == 0:
            return parse_fail_result(self.name)

        score = round(correct / n_usable, 3)
        fp_rate = round(fp / n_false, 3) if n_false else 0.0
        fn_rate = round(fn / n_true, 3) if n_true else 0.0

        return BiasResult(
            test_name=self.name,
            score=score,
            verdict="PASS" if score >= 0.7 else "FAIL",
            n_fixtures=n_usable,
            details={
                "accuracy": score,
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
            },
            examples=failures,
            severity=severity_from_score(score),
        )
