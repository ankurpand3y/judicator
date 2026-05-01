from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from judicator import Judge, JudgeAuditor
from judicator.report import AuditReport


# ── shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pointwise_report() -> AuditReport:
    judge = Judge(
        llm_fn=lambda _: "7",
        system_prompt="You are an evaluator.",
        eval_template="Question: {question}\nResponse: {response}\nScore 1-10.",
        judge_name="test_pw_judge",
    )
    return JudgeAuditor(judge, domain="qa", confirm=False).audit()


@pytest.fixture(scope="module")
def pairwise_report() -> AuditReport:
    judge = Judge(
        llm_fn=lambda _: "A",
        system_prompt="You are an evaluator.",
        eval_template=(
            "Question: {question}\nA: {response_a}\nB: {response_b}\n"
            "Which is better? Answer A or B."
        ),
        judge_name="test_pair_judge",
    )
    return JudgeAuditor(judge, domain="qa", confirm=False).audit()


# ── convenience methods ────────────────────────────────────────────────────────

def test_ranked_returns_worst_first(pointwise_report: AuditReport) -> None:
    scores = [r.score for r in pointwise_report.ranked()]
    assert scores == sorted(scores)


def test_ranked_excludes_na(pointwise_report: AuditReport) -> None:
    assert all(not r.not_applicable for r in pointwise_report.ranked())


def test_failed_tests_verdict(pointwise_report: AuditReport) -> None:
    assert all(r.verdict == "FAIL" for r in pointwise_report.failed_tests())


def test_passed_tests_verdict(pointwise_report: AuditReport) -> None:
    assert all(r.verdict == "PASS" for r in pointwise_report.passed_tests())


def test_failed_passed_count_adds_up(pointwise_report: AuditReport) -> None:
    n_applicable = len(pointwise_report.ranked())
    assert len(pointwise_report.failed_tests()) + len(pointwise_report.passed_tests()) == n_applicable


# ── console output ─────────────────────────────────────────────────────────────

def test_summary_contains_judge_name(pointwise_report: AuditReport) -> None:
    assert "test_pw_judge" in pointwise_report.summary()


def test_summary_contains_domain(pointwise_report: AuditReport) -> None:
    assert "qa" in pointwise_report.summary()


def test_summary_contains_judge_type(pointwise_report: AuditReport) -> None:
    assert "pointwise" in pointwise_report.summary()


def test_summary_contains_all_applicable_tests(pointwise_report: AuditReport) -> None:
    s = pointwise_report.summary()
    for res in pointwise_report.ranked():
        assert res.test_name[:10] in s


def test_summary_na_tests_not_in_console(pointwise_report: AuditReport) -> None:
    # N/A tests are in the JSON report; console shows applicable results only
    s = pointwise_report.summary()
    na = [r for r in pointwise_report.tests.values() if r.not_applicable]
    assert len(na) > 0
    for r in na:
        assert r.test_name not in s


def test_summary_contains_verdict_labels(pointwise_report: AuditReport) -> None:
    s = pointwise_report.summary()
    assert "FAIL" in s or "PASS" in s


def test_summary_lines_fit_80_cols(pointwise_report: AuditReport) -> None:
    for line in pointwise_report.summary().splitlines():
        assert len(line) <= 80, f"Line too long ({len(line)}): {line!r}"


def test_summary_attribution_in_json_not_console(pointwise_report: AuditReport) -> None:
    # Attribution is in the JSON export, not the console summary
    s = pointwise_report.summary()
    assert "OffsetBias" not in s
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    assert "fixtures" in data


def test_summary_printed_during_audit(capsys: pytest.CaptureFixture[str]) -> None:
    """audit() should print the cost estimate; summary() is available on report."""
    judge = Judge(
        llm_fn=lambda _: "7",
        system_prompt="s",
        eval_template="Q: {question}\nR: {response}\nScore 1-10.",
        judge_name="capsys_judge",
    )
    JudgeAuditor(judge, domain="dialogue", confirm=False, max_items_per_test=2).audit()
    out = capsys.readouterr().out
    assert "AUDIT PLAN" in out


# ── JSON export ────────────────────────────────────────────────────────────────

def test_save_json_creates_file(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        assert Path(path).exists()


def test_save_json_valid_json(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
        assert isinstance(data, dict)


def test_json_has_required_top_level_keys(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    for key in ("judicator_version", "timestamp", "judge", "summary", "tests", "fixtures"):
        assert key in data, f"missing key: {key}"


def test_json_version_is_string(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    assert data["judicator_version"] == "0.2.2"


def test_json_all_7_tests_present(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    expected = {"position", "verbosity", "self_consistency", "scale_anchoring",
                "authority", "concreteness", "yes_bias"}
    assert set(data["tests"].keys()) == expected


def test_json_na_test_has_skip_reason(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    assert data["tests"]["position"]["not_applicable"] is True
    assert "skip_reason" in data["tests"]["position"]


def test_json_applicable_test_has_score(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    verb = data["tests"]["verbosity"]
    assert "score" in verb
    assert "verdict" in verb
    assert "severity" in verb
    assert "n_fixtures" in verb


def test_json_summary_fields(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    s = data["summary"]
    for key in ("tests_run", "tests_failed", "tests_passed", "tests_na",
                "worst_bias", "worst_bias_score"):
        assert key in s


def test_json_fixture_version_present(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.json")
        pointwise_report.save_json(path)
        data = json.loads(Path(path).read_text())
    assert data["fixtures"]["fixture_version"] == "0.1.0"


# ── HTML export ────────────────────────────────────────────────────────────────

def test_save_html_creates_file(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.html")
        pointwise_report.save_html(path)
        assert Path(path).exists()


def test_html_starts_with_doctype(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.html")
        pointwise_report.save_html(path)
        content = Path(path).read_text(encoding="utf-8")
    assert content.strip().startswith("<!DOCTYPE html>")


def test_html_contains_judge_name(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.html")
        pointwise_report.save_html(path)
        content = Path(path).read_text(encoding="utf-8")
    assert "test_pw_judge" in content


def test_html_contains_all_test_names(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.html")
        pointwise_report.save_html(path)
        content = Path(path).read_text(encoding="utf-8")
    for name in ("verbosity", "scale_anchoring", "authority", "concreteness",
                 "self_consistency"):
        assert name in content


def test_html_has_inline_css(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.html")
        pointwise_report.save_html(path)
        content = Path(path).read_text(encoding="utf-8")
    assert "<style>" in content


def test_html_no_external_resources(pointwise_report: AuditReport) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "audit.html")
        pointwise_report.save_html(path)
        content = Path(path).read_text(encoding="utf-8")
    assert "http" not in content
    assert "cdn" not in content.lower()


def test_pairwise_report_summary(pairwise_report: AuditReport) -> None:
    s = pairwise_report.summary()
    assert "pairwise" in s
    assert "position" in s
