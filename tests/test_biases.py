"""Tests for bias tests: position, verbosity, self_consistency, scale_anchoring, authority,
concreteness, yes_bias."""
from __future__ import annotations

import pytest

from judicator.biases.authority import AuthorityBiasTest
from judicator.biases.concreteness import ConcretenessBiasTest
from judicator.biases.position import PositionBiasTest
from judicator.biases.scale_anchoring import ScaleAnchoringTest
from judicator.biases.self_consistency import SelfConsistencyTest
from judicator.biases.verbosity import VerbosityBiasTest
from judicator.biases.yes_bias import YesBiasTest
from judicator.cost import CallCounter
from judicator.fixtures.loader import load_fixtures
from judicator.judge import Judge

MAX = 10  # fixture items per test — keeps suite fast


# ── judge factories ────────────────────────────────────────────────────────────

def _pointwise_judge(response_fn=None) -> Judge:
    fn = response_fn or (lambda _: "7")
    return Judge(
        llm_fn=fn,
        system_prompt="You are an evaluator.",
        eval_template="Question: {question}\nResponse: {response}\nScore 1-10.",
        judge_name="pointwise_test",
    )


def _pairwise_judge(response_fn=None) -> Judge:
    fn = response_fn or (lambda _: "A")
    return Judge(
        llm_fn=fn,
        system_prompt="You are an evaluator.",
        eval_template=(
            "Question: {question}\nResponse A: {response_a}\n"
            "Response B: {response_b}\nWhich is better? Answer A or B."
        ),
        judge_name="pairwise_test",
    )


def _binary_judge(response_fn=None) -> Judge:
    fn = response_fn or (lambda _: "Yes")
    return Judge(
        llm_fn=fn,
        system_prompt="You are an evaluator.",
        eval_template="Response: {response}\nIs this a good response? Answer yes or no.",
        judge_name="binary_test",
    )


def _binary_statement_judge(response_fn=None) -> Judge:
    """Binary judge whose template uses {statement} — required for yes_bias test."""
    fn = response_fn or (lambda _: "Yes")
    return Judge(
        llm_fn=fn,
        system_prompt="You are a fact checker.",
        eval_template="Statement: {statement}\nIs this true? Answer yes or no.",
        judge_name="binary_statement_test",
    )


class _Cyclic:
    """Returns values in a repeating cycle — simulates non-deterministic judge."""
    def __init__(self, values: list[str]) -> None:
        self._v = values
        self._i = 0

    def __call__(self, _: str) -> str:
        v = self._v[self._i % len(self._v)]
        self._i += 1
        return v


# ── position bias ──────────────────────────────────────────────────────────────

class TestPositionBias:
    test = PositionBiasTest()
    fixtures = load_fixtures("qa/position.jsonl", max_items=MAX)

    def test_always_a_judge_fails(self) -> None:
        """A judge that always picks slot A is maximally position-biased."""
        result = self.test.run(_pairwise_judge(lambda _: "A"), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"
        assert result.score < 0.3
        assert result.severity == "CRITICAL"

    def test_calibrated_judge_passes(self) -> None:
        """A judge that always picks the winner passes regardless of slot order."""
        call_count: list[int] = [0]

        def winner_aware(prompt: str) -> str:
            # In AB order winner is in A; in BA order winner is in B.
            # Fixtures put winner first in the AB call, so we alternate A, B, A, B...
            call_count[0] += 1
            return "A" if call_count[0] % 2 == 1 else "B"

        result = self.test.run(_pairwise_judge(winner_aware), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score >= 0.9

    def test_not_applicable_for_pointwise(self) -> None:
        from judicator._types import JudgeType
        assert not self.test.is_applicable(JudgeType.POINTWISE)
        assert not self.test.is_applicable(JudgeType.BINARY)
        assert self.test.is_applicable(JudgeType.PAIRWISE)

    def test_details_keys(self) -> None:
        result = self.test.run(_pairwise_judge(), self.fixtures, CallCounter())
        assert "inconsistency_rate" in result.details
        assert "slot_a_pick_rate" in result.details

    def test_n_fixtures_matches(self) -> None:
        result = self.test.run(_pairwise_judge(), self.fixtures, CallCounter())
        assert result.n_fixtures <= MAX

    def test_call_counter_incremented(self) -> None:
        counter = CallCounter()
        result = self.test.run(_pairwise_judge(), self.fixtures, counter)
        assert counter.count == result.n_fixtures * 2

    def test_fixture_path(self) -> None:
        assert "{domain}" in self.test.fixture_path


# ── verbosity bias ─────────────────────────────────────────────────────────────

class TestVerbosityBias:
    test = VerbosityBiasTest()
    fixtures = load_fixtures("qa/verbosity.jsonl", max_items=MAX)

    def test_length_biased_judge_fails(self) -> None:
        """Judge scores proportional to response length → heavy verbosity bias."""
        def length_score(prompt: str) -> str:
            response_len = len(prompt.split("Response:")[-1])
            return str(min(10, max(1, response_len // 30)))

        result = self.test.run(_pointwise_judge(length_score), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"

    def test_calibrated_judge_passes(self) -> None:
        """Fixed-score judge has zero inflation → PASS."""
        result = self.test.run(_pointwise_judge(lambda _: "7"), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score == 1.0

    def test_details_keys_pointwise(self) -> None:
        result = self.test.run(_pointwise_judge(), self.fixtures, CallCounter())
        assert "mean_inflation" in result.details
        assert "fraction_inflated" in result.details

    def test_pairwise_verbosity_biased_fails(self) -> None:
        """Judge always picks the manipulated (padded) response regardless of slot → FAIL.
        AB order: manip is B → returns B; BA order: manip is A → returns A.
        bias_rate = 1.0, inflation = 0.5, score = 0.0."""
        call_count: list[int] = [0]

        def pick_padded(_: str) -> str:
            call_count[0] += 1
            return "B" if call_count[0] % 2 == 1 else "A"

        fixtures_pairwise = load_fixtures("qa/verbosity.jsonl", max_items=MAX)
        result = self.test.run(_pairwise_judge(pick_padded), fixtures_pairwise, CallCounter())
        assert result.verdict == "FAIL"
        assert result.score == 0.0

    def test_pairwise_calibrated_passes(self) -> None:
        """Pairwise judge alternates A/B → 50% padded win rate → no net inflation."""
        call_count: list[int] = [0]

        def alternating(_: str) -> str:
            call_count[0] += 1
            return "A" if call_count[0] % 2 == 1 else "B"

        result = self.test.run(_pairwise_judge(alternating),
                               load_fixtures("qa/verbosity.jsonl", max_items=MAX),
                               CallCounter())
        assert result.verdict == "PASS"

    def test_binary_calibrated_passes(self) -> None:
        """Binary judge always says Yes regardless → inflation = 0 → PASS."""
        result = self.test.run(_binary_judge(lambda _: "Yes"),
                               load_fixtures("qa/verbosity.jsonl", max_items=MAX),
                               CallCounter())
        assert result.verdict == "PASS"
        assert result.score == 1.0

    def test_binary_biased_fails(self) -> None:
        """Binary judge says No for original, Yes for manipulated → maximum inflation."""
        call_count: list[int] = [0]

        def biased(_: str) -> str:
            call_count[0] += 1
            return "No" if call_count[0] % 2 == 1 else "Yes"

        result = self.test.run(_binary_judge(biased),
                               load_fixtures("qa/verbosity.jsonl", max_items=MAX),
                               CallCounter())
        assert result.verdict == "FAIL"

    def test_call_counter_pointwise(self) -> None:
        counter = CallCounter()
        result = self.test.run(_pointwise_judge(), self.fixtures, counter)
        assert counter.count == result.n_fixtures * 2

    def test_call_counter_pairwise(self) -> None:
        counter = CallCounter()
        result = self.test.run(_pairwise_judge(),
                               load_fixtures("qa/verbosity.jsonl", max_items=MAX),
                               counter)
        assert counter.count == result.n_fixtures * 2


# ── self_consistency ───────────────────────────────────────────────────────────

class TestSelfConsistency:
    test = SelfConsistencyTest()
    fixtures = load_fixtures("universal/self_consistency.jsonl", max_items=MAX)

    def test_deterministic_judge_scores_1(self) -> None:
        """Perfectly deterministic judge has zero variance → score = 1.0."""
        result = self.test.run(_pointwise_judge(lambda _: "7"), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score == 1.0

    def test_high_variance_judge_fails(self) -> None:
        """Judge cycling through extreme values has high variance → FAIL."""
        cyclic = _Cyclic(["1", "10", "1", "10", "1"])
        result = self.test.run(_pointwise_judge(cyclic), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"
        assert result.score < 0.5

    def test_binary_deterministic_passes(self) -> None:
        """Binary judge always returns same answer → consistency = 1.0."""
        result = self.test.run(_binary_judge(lambda _: "Yes"), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score >= 0.99

    def test_binary_alternating_fails(self) -> None:
        """Binary judge alternates Yes/No → low consistency → FAIL."""
        cyclic = _Cyclic(["Yes", "No", "Yes", "No", "Yes"])
        result = self.test.run(_binary_judge(cyclic), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"

    def test_pairwise_returns_na(self) -> None:
        result = self.test.run(_pairwise_judge(), self.fixtures, CallCounter())
        assert result.not_applicable is True
        assert result.verdict == "N/A"

    def test_details_keys_pointwise(self) -> None:
        result = self.test.run(_pointwise_judge(), self.fixtures, CallCounter())
        assert "mean_score_variance" in result.details
        assert result.details["n_runs_per_item"] == 5

    def test_call_counter_pointwise(self) -> None:
        counter = CallCounter()
        result = self.test.run(_pointwise_judge(), self.fixtures, counter)
        assert counter.count == result.n_fixtures * 5

    def test_fixture_count(self) -> None:
        all_fixtures = load_fixtures("universal/self_consistency.jsonl")
        result = self.test.run(_pointwise_judge(), all_fixtures, CallCounter())
        assert result.n_fixtures == 40


# ── ALL_TESTS registry ─────────────────────────────────────────────────────────

# ── scale anchoring ────────────────────────────────────────────────────────────

class TestScaleAnchoring:
    test = ScaleAnchoringTest()
    # Load all 200 items — first 100 are "high", last 100 are "low"; must include both.
    fixtures = load_fixtures("universal/scale_tiers.jsonl")

    def test_compressed_judge_fails(self) -> None:
        """Judge always returns 5 → spread = 0 → FAIL."""
        result = self.test.run(_pointwise_judge(lambda _: "5"), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"
        assert result.score == 0.0

    def test_calibrated_judge_passes(self) -> None:
        """Judge returns 9 for high-tier and 2 for low-tier → spread=7 → PASS."""
        tier_order = [item["tier"] for item in self.fixtures]
        call_count: list[int] = [0]

        def tier_aware(_: str) -> str:
            t = tier_order[call_count[0]]
            call_count[0] += 1
            return "9" if t == "high" else "2"

        result = self.test.run(_pointwise_judge(tier_aware), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score >= 0.9

    def test_details_keys(self) -> None:
        result = self.test.run(_pointwise_judge(), self.fixtures, CallCounter())
        for key in ("high_tier_mean", "low_tier_mean", "spread", "expected_spread"):
            assert key in result.details

    def test_not_applicable_for_pairwise(self) -> None:
        from judicator._types import JudgeType
        assert not self.test.is_applicable(JudgeType.PAIRWISE)
        assert not self.test.is_applicable(JudgeType.BINARY)
        assert self.test.is_applicable(JudgeType.POINTWISE)

    def test_call_counter(self) -> None:
        counter = CallCounter()
        result = self.test.run(_pointwise_judge(), self.fixtures, counter)
        assert counter.count == result.n_fixtures


# ── authority bias ─────────────────────────────────────────────────────────────

class TestAuthorityBias:
    test = AuthorityBiasTest()
    fixtures = load_fixtures("qa/authority.jsonl", max_items=MAX)

    def test_authority_biased_judge_fails(self) -> None:
        """Judge inflates by +3 for every manipulated response → mean_inflation=3 → FAIL."""
        call_count: list[int] = [0]

        def inflate_on_even(_: str) -> str:
            call_count[0] += 1
            return "8" if call_count[0] % 2 == 0 else "5"  # odd=original→5, even=manipulated→8

        result = self.test.run(_pointwise_judge(inflate_on_even), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"
        assert result.score < 0.4

    def test_calibrated_judge_passes(self) -> None:
        result = self.test.run(_pointwise_judge(lambda _: "7"), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score == 1.0

    def test_pairwise_biased_fails(self) -> None:
        call_count: list[int] = [0]

        def pick_authority(_: str) -> str:
            call_count[0] += 1
            return "B" if call_count[0] % 2 == 1 else "A"

        result = self.test.run(_pairwise_judge(pick_authority),
                               load_fixtures("qa/authority.jsonl", max_items=MAX),
                               CallCounter())
        assert result.verdict == "FAIL"

    def test_binary_calibrated_passes(self) -> None:
        result = self.test.run(_binary_judge(lambda _: "Yes"),
                               load_fixtures("qa/authority.jsonl", max_items=MAX),
                               CallCounter())
        assert result.verdict == "PASS"

    def test_call_counter_pointwise(self) -> None:
        counter = CallCounter()
        result = self.test.run(_pointwise_judge(), self.fixtures, counter)
        assert counter.count == result.n_fixtures * 2

    def test_details_keys(self) -> None:
        result = self.test.run(_pointwise_judge(), self.fixtures, CallCounter())
        assert "mean_inflation" in result.details
        assert "fraction_inflated" in result.details


# ── concreteness bias ──────────────────────────────────────────────────────────

class TestConcreteness:
    test = ConcretenessBiasTest()
    fixtures = load_fixtures("universal/concreteness.jsonl")  # all 14

    def test_concrete_biased_judge_fails(self) -> None:
        """Judge gives higher score to concrete (specifics-loaded) responses → FAIL."""
        call_count: list[int] = [0]

        def concrete_wins(_: str) -> str:
            call_count[0] += 1
            return "4" if call_count[0] % 2 == 1 else "8"  # vague=4, concrete=8

        result = self.test.run(_pointwise_judge(concrete_wins), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"

    def test_calibrated_judge_passes(self) -> None:
        """Judge gives same score to both → fraction_calibrated = 1.0."""
        result = self.test.run(_pointwise_judge(lambda _: "7"), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score == 1.0

    def test_pairwise_calibrated_passes(self) -> None:
        """Pairwise judge always picks vague (A in AB order, B in BA order) → PASS."""
        call_count: list[int] = [0]

        def pick_vague(_: str) -> str:
            call_count[0] += 1
            return "A" if call_count[0] % 2 == 1 else "B"

        result = self.test.run(_pairwise_judge(pick_vague), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score == 1.0

    def test_pairwise_biased_fails(self) -> None:
        """Pairwise judge always picks concrete (B in AB order, A in BA order) → FAIL."""
        call_count: list[int] = [0]

        def pick_concrete(_: str) -> str:
            call_count[0] += 1
            return "B" if call_count[0] % 2 == 1 else "A"

        result = self.test.run(_pairwise_judge(pick_concrete), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"
        assert result.score == 0.0

    def test_not_applicable_for_binary(self) -> None:
        from judicator._types import JudgeType
        assert not self.test.is_applicable(JudgeType.BINARY)
        assert self.test.is_applicable(JudgeType.POINTWISE)
        assert self.test.is_applicable(JudgeType.PAIRWISE)

    def test_call_counter_pointwise(self) -> None:
        counter = CallCounter()
        result = self.test.run(_pointwise_judge(), self.fixtures, counter)
        assert counter.count == result.n_fixtures * 2


# ── yes bias ───────────────────────────────────────────────────────────────────

class TestYesBias:
    test = YesBiasTest()
    fixtures = load_fixtures("universal/yesno_probes.jsonl", max_items=MAX * 2)

    def _perfect_judge(self) -> Judge:
        """Returns correct Yes/No per ground_truth in fixture order."""
        items = load_fixtures("universal/yesno_probes.jsonl", max_items=MAX * 2)
        answers = ["Yes" if item["ground_truth"] else "No" for item in items]
        return _binary_statement_judge(_Cyclic(answers))

    def test_perfect_judge_passes(self) -> None:
        result = self.test.run(self._perfect_judge(), self.fixtures, CallCounter())
        assert result.verdict == "PASS"
        assert result.score == 1.0

    def test_always_yes_is_biased(self) -> None:
        """Always-Yes judge scores ~0.5 on balanced set — FAIL."""
        result = self.test.run(_binary_statement_judge(lambda _: "Yes"), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"
        assert 0.4 <= result.score <= 0.6

    def test_always_no_is_biased(self) -> None:
        """Always-No judge also scores ~0.5."""
        result = self.test.run(_binary_statement_judge(lambda _: "No"), self.fixtures, CallCounter())
        assert result.verdict == "FAIL"

    def test_details_keys(self) -> None:
        result = self.test.run(_binary_statement_judge(lambda _: "Yes"), self.fixtures, CallCounter())
        for key in ("accuracy", "false_positive_rate", "false_negative_rate"):
            assert key in result.details

    def test_not_applicable_for_pointwise(self) -> None:
        from judicator._types import JudgeType
        assert not self.test.is_applicable(JudgeType.POINTWISE)
        assert not self.test.is_applicable(JudgeType.PAIRWISE)
        assert self.test.is_applicable(JudgeType.BINARY)

    def test_call_counter(self) -> None:
        counter = CallCounter()
        result = self.test.run(_binary_statement_judge(lambda _: "Yes"), self.fixtures, counter)
        assert counter.count == result.n_fixtures


# ── ALL_TESTS registry ─────────────────────────────────────────────────────────

def test_all_tests_registered() -> None:
    from judicator.biases import ALL_TESTS
    names = [t.name for t in ALL_TESTS]
    for expected in ("position", "verbosity", "self_consistency",
                     "scale_anchoring", "authority", "concreteness", "yes_bias"):
        assert expected in names
    assert len(names) == 7


def test_is_applicable() -> None:
    from judicator._types import JudgeType
    pos = PositionBiasTest()
    assert pos.is_applicable(JudgeType.PAIRWISE)
    assert not pos.is_applicable(JudgeType.POINTWISE)
    assert not pos.is_applicable(JudgeType.BINARY)

    verb = VerbosityBiasTest()
    assert verb.is_applicable(JudgeType.POINTWISE)
    assert verb.is_applicable(JudgeType.PAIRWISE)
    assert verb.is_applicable(JudgeType.BINARY)

    sc = SelfConsistencyTest()
    assert sc.is_applicable(JudgeType.POINTWISE)
    assert sc.is_applicable(JudgeType.BINARY)
    assert not sc.is_applicable(JudgeType.PAIRWISE)

    sa = ScaleAnchoringTest()
    assert sa.is_applicable(JudgeType.POINTWISE)
    assert not sa.is_applicable(JudgeType.PAIRWISE)
    assert not sa.is_applicable(JudgeType.BINARY)

    con = ConcretenessBiasTest()
    assert con.is_applicable(JudgeType.POINTWISE)
    assert con.is_applicable(JudgeType.PAIRWISE)
    assert not con.is_applicable(JudgeType.BINARY)

    yb = YesBiasTest()
    assert yb.is_applicable(JudgeType.BINARY)
    assert not yb.is_applicable(JudgeType.POINTWISE)
    assert not yb.is_applicable(JudgeType.PAIRWISE)
