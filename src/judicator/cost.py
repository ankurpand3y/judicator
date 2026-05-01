from __future__ import annotations
import threading
from dataclasses import dataclass, field

CALLS_PER_TEST: dict[str, object] = {
    "position":         lambda n: n * 2,
    "verbosity":        lambda n: n * 2,
    "self_consistency": lambda n: n * 5,
    "scale_anchoring":  lambda n: n,
    "authority":        lambda n: n * 2,
    "concreteness":     lambda n: n * 2,
    "yes_bias":         lambda n: n,
}


_W = 64
_INNER = _W - 2  # 62
_COL_TEST = 22
_COL_CALLS = _INNER - _COL_TEST - 1  # remaining inner width minus column separator


def _row(label: str, value: str) -> str:
    text = f"{label}{value}"
    return f"║  {text:<{_INNER - 2}}║"


def _trow(test: str, calls: str) -> str:
    return f"║ {test[:_COL_TEST - 1]:<{_COL_TEST - 1}}║ {calls[:_COL_CALLS - 1]:<{_COL_CALLS - 1}}║"


def _hr(left: str = "╠", right: str = "╣", fill: str = "═") -> str:
    return left + fill * _INNER + right


def _trow_sep(left: str, mid: str, right: str) -> str:
    return left + "═" * _COL_TEST + mid + "═" * _COL_CALLS + right


@dataclass
class CostEstimate:
    total_calls: int
    calls_per_test: dict[str, int]
    estimated_cost_usd: float | None
    cost_per_call: float | None

    def display(
        self,
        judge_name: str | None = None,
        judge_type: str | None = None,
        judge_type_auto: bool = False,
        domain: str | None = None,
    ) -> None:
        lines: list[str] = [
            _hr("╔", "╗"),
            _row("", "AUDIT PLAN"),
            _hr(),
        ]
        if judge_name is not None:
            lines.append(_row("Judge:    ", judge_name))
        if judge_type is not None:
            jt = judge_type + (" (auto-detected)" if judge_type_auto else "")
            lines.append(_row("Type:     ", jt))
        if domain is not None:
            lines.append(_row("Domain:   ", domain))

        lines.append(_trow_sep("╠", "╦", "╣"))
        lines.append(_trow(" TEST", " CALLS"))
        lines.append(_trow_sep("╠", "╬", "╣"))
        for test, calls in self.calls_per_test.items():
            lines.append(_trow(f" {test}", f" {calls}"))
        lines.append(_trow_sep("╠", "╩", "╣"))

        lines.append(_row("Total calls:  ", str(self.total_calls)))
        if self.estimated_cost_usd is not None:
            cost_str = (
                f"~${self.estimated_cost_usd:.2f} USD "
                f"(${self.cost_per_call:.6f}/call)"
            )
            lines.append(_row("Estimated cost: ", cost_str))
        else:
            lines.append(_row("", "Pass cost_per_call=<USD/call> for cost estimate"))
        lines.append(_hr("╚", "╝"))
        print("\n" + "\n".join(lines))


@dataclass
class CallCounter:
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def increment(self) -> None:
        with self._lock:
            self.count += 1


def estimate_calls(
    test_names: list[str],
    fixture_counts: dict[str, int],
    cost_per_call: float | None = None,
) -> CostEstimate:
    calls_per_test: dict[str, int] = {}
    for name in test_names:
        fn = CALLS_PER_TEST.get(name)
        n = fixture_counts.get(name, 0)
        calls_per_test[name] = fn(n) if fn else n  # type: ignore[operator]
    total = sum(calls_per_test.values())
    cost = total * cost_per_call if cost_per_call is not None else None
    return CostEstimate(
        total_calls=total,
        calls_per_test=calls_per_test,
        estimated_cost_usd=cost,
        cost_per_call=cost_per_call,
    )
