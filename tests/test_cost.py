import pytest
from judicator.cost import CALLS_PER_TEST, CallCounter, CostEstimate, estimate_calls


# ── CALLS_PER_TEST lambdas ─────────────────────────────────────────────────────

@pytest.mark.parametrize("test,n,expected", [
    ("position",         50,  100),
    ("verbosity",        100, 200),
    ("self_consistency", 40,  200),
    ("scale_anchoring",  200, 200),
    ("authority",        50,  100),
    ("concreteness",     14,   28),
    ("yes_bias",         100, 100),
])
def test_calls_per_test(test: str, n: int, expected: int) -> None:
    assert CALLS_PER_TEST[test](n) == expected  # type: ignore[operator]


# ── CallCounter ────────────────────────────────────────────────────────────────

def test_call_counter_starts_zero() -> None:
    c = CallCounter()
    assert c.count == 0


def test_call_counter_increments() -> None:
    c = CallCounter()
    c.increment()
    c.increment()
    assert c.count == 2


# ── estimate_calls ─────────────────────────────────────────────────────────────

def test_estimate_no_cost() -> None:
    est = estimate_calls(["verbosity", "authority"], {"verbosity": 50, "authority": 50})
    assert est.total_calls == 200
    assert est.estimated_cost_usd is None
    assert est.cost_per_call is None


def test_estimate_with_cost() -> None:
    est = estimate_calls(["verbosity"], {"verbosity": 100}, cost_per_call=0.005)
    assert est.total_calls == 200
    assert abs(est.estimated_cost_usd - 1.0) < 1e-9
    assert est.cost_per_call == 0.005


def test_estimate_multiple_tests() -> None:
    est = estimate_calls(
        ["position", "self_consistency"],
        {"position": 50, "self_consistency": 40},
    )
    assert est.calls_per_test["position"] == 100
    assert est.calls_per_test["self_consistency"] == 200
    assert est.total_calls == 300


def test_estimate_zero_fixtures() -> None:
    est = estimate_calls(["verbosity"], {})
    assert est.total_calls == 0


# ── CostEstimate.display ───────────────────────────────────────────────────────

def test_display_no_cost(capsys: pytest.CaptureFixture[str]) -> None:
    est = estimate_calls(["verbosity"], {"verbosity": 50})
    est.display()
    out = capsys.readouterr().out
    assert "AUDIT PLAN" in out
    assert "verbosity" in out
    assert "100" in out
    assert "Total calls" in out
    assert "cost_per_call" in out


def test_display_with_cost(capsys: pytest.CaptureFixture[str]) -> None:
    est = estimate_calls(["verbosity"], {"verbosity": 50}, cost_per_call=0.005)
    est.display()
    out = capsys.readouterr().out
    assert "$0.50 USD" in out
    assert "0.005000/call" in out


def test_display_all_tests(capsys: pytest.CaptureFixture[str]) -> None:
    counts = {t: 50 for t in CALLS_PER_TEST}
    est = estimate_calls(list(CALLS_PER_TEST.keys()), counts)
    est.display()
    out = capsys.readouterr().out
    for test in CALLS_PER_TEST:
        assert test in out
