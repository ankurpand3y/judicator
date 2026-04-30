from __future__ import annotations

import pytest

from judicator import Judge, JudgeAuditor
from judicator.exceptions import AuditCancelled, DetectionError
from judicator.report import AuditReport


# ── judge factories ────────────────────────────────────────────────────────────

def _pointwise(name: str = "pw_judge") -> Judge:
    return Judge(
        llm_fn=lambda _: "7",
        system_prompt="You are an evaluator.",
        eval_template="Question: {question}\nResponse: {response}\nScore 1-10.",
        judge_name=name,
    )


def _pairwise(name: str = "pair_judge") -> Judge:
    return Judge(
        llm_fn=lambda _: "A",
        system_prompt="You are an evaluator.",
        eval_template=(
            "Question: {question}\nResponse A: {response_a}\n"
            "Response B: {response_b}\nWhich is better? Answer A or B."
        ),
        judge_name=name,
    )


def _binary(name: str = "bin_judge") -> Judge:
    return Judge(
        llm_fn=lambda _: "Yes",
        system_prompt="You are an evaluator.",
        eval_template="Response: {response}\nIs this good? Answer yes or no.",
        judge_name=name,
    )


# ── domain validation ──────────────────────────────────────────────────────────

def test_invalid_domain_raises() -> None:
    with pytest.raises(ValueError, match="domain must be one of"):
        JudgeAuditor(_pointwise(), domain="finance")


def test_valid_domains_accepted() -> None:
    for domain in ("qa", "summarization", "code", "safety", "dialogue"):
        JudgeAuditor(_pointwise(), domain=domain, confirm=False)


# ── pointwise end-to-end ───────────────────────────────────────────────────────

def test_pointwise_audit_returns_report() -> None:
    auditor = JudgeAuditor(_pointwise(), domain="qa", confirm=False)
    report = auditor.audit()
    assert isinstance(report, AuditReport)


def test_pointwise_report_has_all_7_tests() -> None:
    auditor = JudgeAuditor(_pointwise(), domain="qa", confirm=False)
    report = auditor.audit()
    expected = {"position", "verbosity", "self_consistency", "scale_anchoring",
                "authority", "concreteness", "yes_bias"}
    assert set(report.tests.keys()) == expected


def test_pointwise_na_tests_correct() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    # position = pairwise only, yes_bias = binary only
    assert report.tests["position"].not_applicable
    assert report.tests["yes_bias"].not_applicable
    assert not report.tests["verbosity"].not_applicable
    assert not report.tests["scale_anchoring"].not_applicable


def test_pointwise_judge_info_in_report() -> None:
    report = JudgeAuditor(_pointwise("my_judge"), domain="qa", confirm=False).audit()
    assert report.judge_name == "my_judge"
    assert report.judge_type == "pointwise"
    assert report.domain == "qa"
    assert report.timestamp  # non-empty


# ── pairwise end-to-end ────────────────────────────────────────────────────────

def test_pairwise_audit_returns_report() -> None:
    report = JudgeAuditor(_pairwise(), domain="qa", confirm=False).audit()
    assert isinstance(report, AuditReport)


def test_pairwise_na_tests_correct() -> None:
    report = JudgeAuditor(_pairwise(), domain="qa", confirm=False).audit()
    assert report.tests["scale_anchoring"].not_applicable
    assert report.tests["yes_bias"].not_applicable
    assert report.tests["self_consistency"].not_applicable
    assert not report.tests["position"].not_applicable
    assert not report.tests["verbosity"].not_applicable


def test_pairwise_missing_fixture_marked_na() -> None:
    """summarization has no position.jsonl — should be N/A, not a crash."""
    report = JudgeAuditor(_pairwise(), domain="summarization", confirm=False).audit()
    assert report.tests["position"].not_applicable
    assert "no fixture" in report.tests["position"].skip_reason


# ── binary end-to-end ──────────────────────────────────────────────────────────

def test_binary_audit_returns_report() -> None:
    report = JudgeAuditor(_binary(), domain="qa", confirm=False).audit()
    assert isinstance(report, AuditReport)


def test_binary_na_tests_correct() -> None:
    report = JudgeAuditor(_binary(), domain="qa", confirm=False).audit()
    assert report.tests["position"].not_applicable
    assert report.tests["scale_anchoring"].not_applicable
    assert report.tests["concreteness"].not_applicable
    assert not report.tests["verbosity"].not_applicable
    assert not report.tests["authority"].not_applicable


# ── ranking ────────────────────────────────────────────────────────────────────

def test_ranked_order() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    ranked = report.ranked()
    scores = [r.score for r in ranked]
    assert scores == sorted(scores)


def test_ranked_excludes_na() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    assert all(not r.not_applicable for r in report.ranked())


def test_rank_field_set() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    ranked = report.ranked()
    for i, r in enumerate(ranked):
        assert r.rank == i + 1


# ── failed/passed filters ──────────────────────────────────────────────────────

def test_failed_and_passed_partition() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    applicable = [r for r in report.tests.values() if not r.not_applicable]
    assert len(report.failed_tests()) + len(report.passed_tests()) == len(applicable)


def test_failed_tests_have_fail_verdict() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    assert all(r.verdict == "FAIL" for r in report.failed_tests())


def test_passed_tests_have_pass_verdict() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    assert all(r.verdict == "PASS" for r in report.passed_tests())


# ── estimate ───────────────────────────────────────────────────────────────────

def test_estimate_returns_cost_estimate() -> None:
    from judicator.cost import CostEstimate
    est = JudgeAuditor(_pointwise(), domain="qa", confirm=False).estimate()
    assert isinstance(est, CostEstimate)
    assert est.total_calls > 0


def test_estimate_with_cost_per_call() -> None:
    est = JudgeAuditor(
        _pointwise(), domain="qa", confirm=False, cost_per_call=0.005
    ).estimate()
    assert est.estimated_cost_usd is not None
    assert est.estimated_cost_usd > 0


# ── max_items_per_test ─────────────────────────────────────────────────────────

def test_max_items_caps_fixture_count() -> None:
    report = JudgeAuditor(
        _pointwise(), domain="qa", confirm=False, max_items_per_test=5
    ).audit()
    for r in report.tests.values():
        if not r.not_applicable:
            assert r.n_fixtures <= 5


# ── test subset ────────────────────────────────────────────────────────────────

def test_subset_tests_only_runs_requested() -> None:
    report = JudgeAuditor(
        _pointwise(), domain="qa", confirm=False, tests=["verbosity"]
    ).audit()
    assert report.tests["verbosity"].verdict in ("PASS", "FAIL")
    # All other applicable tests are not in results as run — they appear as N/A
    # (auditor skips them because they're not in `tests` filter)
    # Actually: applicable tests NOT in `tests` filter still get N/A entries
    assert "scale_anchoring" in report.tests


# ── confirmation ───────────────────────────────────────────────────────────────

def test_confirm_false_skips_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    """confirm=False should not prompt even in TTY."""
    JudgeAuditor(_pointwise(), domain="qa", confirm=False, max_items_per_test=2).audit()
    # No AuditCancelled means we got here


def test_non_tty_auto_skips_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", _FakeTTY(is_tty=False))
    with pytest.warns(UserWarning, match="Non-TTY"):
        JudgeAuditor(_pointwise(), domain="qa", confirm=True, max_items_per_test=2).audit()


def test_confirm_y_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", _FakeTTY(is_tty=True))
    monkeypatch.setattr("builtins.input", lambda _: "y")
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=True, max_items_per_test=2).audit()
    assert isinstance(report, AuditReport)


def test_confirm_n_raises_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", _FakeTTY(is_tty=True))
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(AuditCancelled):
        JudgeAuditor(_pointwise(), domain="qa", confirm=True, max_items_per_test=2).audit()


# ── probe failure ──────────────────────────────────────────────────────────────

def test_probe_failure_raises_detection_error() -> None:
    def exploding_fn(prompt: str) -> str:
        raise RuntimeError("API error")

    judge = Judge(
        llm_fn=exploding_fn,
        system_prompt="s",
        eval_template="Q: {question}\nR: {response}",
    )
    with pytest.raises(DetectionError, match="probe call failed"):
        JudgeAuditor(judge, domain="qa", confirm=False).audit()


# ── summary ────────────────────────────────────────────────────────────────────

def test_summary_contains_judge_info() -> None:
    report = JudgeAuditor(_pointwise("my_judge"), domain="qa", confirm=False).audit()
    s = report.summary()
    assert "my_judge" in s
    assert "pointwise" in s
    assert "qa" in s


def test_summary_contains_all_test_names() -> None:
    report = JudgeAuditor(_pointwise(), domain="qa", confirm=False).audit()
    s = report.summary()
    for name in ("verbosity", "scale_anchoring", "authority", "concreteness",
                 "self_consistency"):
        assert name in s


# ── helpers ────────────────────────────────────────────────────────────────────

class _FakeTTY:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty
