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


@dataclass
class CostEstimate:
    total_calls: int
    calls_per_test: dict[str, int]
    estimated_cost_usd: float | None
    cost_per_call: float | None

    def display(self) -> None:
        print(f"\nEstimated judge calls: {self.total_calls}")
        print("\nBreakdown by test:")
        for test, calls in self.calls_per_test.items():
            print(f"  {test:<22} {calls} calls")
        if self.estimated_cost_usd is not None:
            print(f"\nEstimated cost: ${self.estimated_cost_usd:.2f} USD")
            print(f"(based on ${self.cost_per_call:.4f} per call — rough estimate;")
            print(f" actual cost depends on prompt + response length)")
        else:
            print("\nTip: pass cost_per_call=<USD per call> to see cost estimate.")
            print("  OpenAI gpt-4o:        ~$0.005 per call")
            print("  OpenAI gpt-4o-mini:   ~$0.0003 per call")
            print("  Anthropic claude-3.5: ~$0.006 per call")


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
