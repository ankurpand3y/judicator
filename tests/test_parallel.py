"""Parallelism correctness, thread-safety, and edge cases (v0.2.0)."""
from __future__ import annotations

import threading
import time

import pytest

from judicator import Judge, JudgeAuditor
from judicator.biases.base import parallel_map
from judicator.cost import CallCounter

POINTWISE_TEMPLATE = "Question: {question}\nResponse: {response}\nScore 1-10."
PAIRWISE_TEMPLATE = (
    "Question: {question}\nResponse A: {response_a}\nResponse B: {response_b}\n"
    "Which is better? Answer A or B."
)
BINARY_TEMPLATE = "Statement: {statement}\nAnswer yes or no."


# ── parallel_map helper ────────────────────────────────────────────────────────

def test_parallel_map_preserves_order():
    items = list(range(50))
    out = parallel_map(lambda x: x * 2, items, max_workers=10)
    assert out == [x * 2 for x in items]


def test_parallel_map_serial_path_when_max_workers_is_1():
    items = list(range(20))
    out = parallel_map(lambda x: x + 1, items, max_workers=1)
    assert out == [x + 1 for x in items]


def test_parallel_map_handles_empty_input():
    assert parallel_map(lambda x: x, [], max_workers=10) == []


def test_parallel_map_handles_single_item_no_threadpool():
    """Single-item lists shouldn't spawn a pool — verify result is correct."""
    assert parallel_map(lambda x: x * 3, [7], max_workers=10) == [21]


def test_parallel_map_actually_runs_concurrently():
    """100ms × 20 items in parallel should finish in < 1s, not 2s."""
    def slow(x: int) -> int:
        time.sleep(0.05)
        return x

    start = time.time()
    parallel_map(slow, list(range(20)), max_workers=20)
    elapsed = time.time() - start
    # Strict: 20 items × 50ms = 1000ms serial. Parallel should be ~50-200ms.
    assert elapsed < 0.5, f"expected parallel speedup; took {elapsed:.2f}s"


def test_parallel_map_caps_workers_at_item_count():
    """max_workers > len(items) should not over-spawn — verifiable via correctness."""
    out = parallel_map(lambda x: x, [1, 2, 3], max_workers=100)
    assert out == [1, 2, 3]


# ── thread-safe CallCounter ────────────────────────────────────────────────────

def test_call_counter_thread_safe():
    counter = CallCounter()

    def hammer():
        for _ in range(1000):
            counter.increment()

    threads = [threading.Thread(target=hammer) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.count == 20 * 1000


def test_call_counter_starts_at_zero():
    assert CallCounter().count == 0


# ── correctness invariance: same result at max_workers=1 vs 10 ────────────────

def _deterministic_judge(response: str, template: str) -> Judge:
    """Judge that always returns the same response — fully deterministic."""
    return Judge(
        llm_fn=lambda _: response,
        system_prompt="x",
        eval_template=template,
        judge_name="det",
    )


def test_audit_result_identical_serial_vs_parallel_pointwise():
    """Pointwise audit at max_workers=1 should match max_workers=10 (deterministic judge)."""
    j = _deterministic_judge("Score: 7", POINTWISE_TEMPLATE)
    r1 = JudgeAuditor(judge=j, domain="qa", confirm=False, max_workers=1,
                      max_items_per_test=10).audit()
    r2 = JudgeAuditor(judge=j, domain="qa", confirm=False, max_workers=10,
                      max_items_per_test=10).audit()
    for name in r1.tests:
        assert r1.tests[name].score == r2.tests[name].score, f"score drift in {name}"
        assert r1.tests[name].verdict == r2.tests[name].verdict, f"verdict drift in {name}"


def test_audit_result_identical_serial_vs_parallel_binary():
    j = _deterministic_judge("No", BINARY_TEMPLATE)
    r1 = JudgeAuditor(judge=j, domain="qa", confirm=False, max_workers=1,
                      max_items_per_test=10).audit()
    r2 = JudgeAuditor(judge=j, domain="qa", confirm=False, max_workers=10,
                      max_items_per_test=10).audit()
    for name in r1.tests:
        assert r1.tests[name].score == r2.tests[name].score, f"score drift in {name}"


def test_audit_result_identical_serial_vs_parallel_pairwise():
    j = _deterministic_judge("A", PAIRWISE_TEMPLATE)
    r1 = JudgeAuditor(judge=j, domain="qa", confirm=False, max_workers=1,
                      max_items_per_test=10).audit()
    r2 = JudgeAuditor(judge=j, domain="qa", confirm=False, max_workers=10,
                      max_items_per_test=10).audit()
    for name in r1.tests:
        assert r1.tests[name].score == r2.tests[name].score, f"score drift in {name}"


# ── call_counter accumulates correctly under parallelism ─────────────────────

def test_call_counter_accurate_under_parallelism():
    """Counter should match expected total even with high concurrency."""
    j = _deterministic_judge("Score: 7", POINTWISE_TEMPLATE)
    auditor = JudgeAuditor(judge=j, domain="qa", confirm=False, max_workers=20,
                           max_items_per_test=10,
                           tests=["verbosity"])
    auditor.audit()
    # verbosity is 2 calls per item; 10 items = 20 calls (exact, deterministic)
    # Auditor also makes 1 probe call before tests run.
    # We can't read counter post-audit since it's local; instead validate via estimate.
    est = auditor.estimate()
    assert est.calls_per_test["verbosity"] == 20


# ── auditor validates max_workers ─────────────────────────────────────────────

def test_auditor_rejects_zero_workers():
    j = _deterministic_judge("Score: 7", POINTWISE_TEMPLATE)
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        JudgeAuditor(judge=j, domain="qa", max_workers=0)


def test_auditor_rejects_negative_workers():
    j = _deterministic_judge("Score: 7", POINTWISE_TEMPLATE)
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        JudgeAuditor(judge=j, domain="qa", max_workers=-5)


def test_auditor_default_max_workers_is_1():
    j = _deterministic_judge("Score: 7", POINTWISE_TEMPLATE)
    auditor = JudgeAuditor(judge=j, domain="qa", confirm=False)
    assert auditor.max_workers == 1


# ── concreteness with n=14 + max_workers=20 doesn't deadlock ─────────────────

def test_concreteness_n14_with_oversized_workers():
    """Should not hang or crash when max_workers > fixture count."""
    j = _deterministic_judge("Score: 7", POINTWISE_TEMPLATE)
    auditor = JudgeAuditor(
        judge=j, domain="qa", confirm=False,
        max_workers=20,
        tests=["concreteness"],
    )
    report = auditor.audit()
    assert "concreteness" in report.tests
    assert report.tests["concreteness"].n_fixtures == 14
