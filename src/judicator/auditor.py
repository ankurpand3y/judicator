from __future__ import annotations

import re
import sys
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from judicator._types import JudgeType
from judicator.biases import ALL_TESTS
from judicator.biases.base import BiasResult
from judicator.cost import CallCounter, CostEstimate, estimate_calls
from judicator.detector import resolve_judge_type
from judicator.exceptions import AuditCancelled, DetectionError, FixtureNotFound
from judicator.fixtures.loader import load_fixtures
from judicator.report import AuditReport

if TYPE_CHECKING:
    from judicator.judge import Judge

VALID_DOMAINS = {"qa", "summarization", "code", "safety", "dialogue"}


class JudgeAuditor:
    def __init__(
        self,
        judge: "Judge",
        domain: str,
        cost_per_call: float | None = None,
        confirm: bool = True,
        tests: list[str] | None = None,
        max_items_per_test: int | None = None,
        max_workers: int = 1,
    ) -> None:
        if domain not in VALID_DOMAINS:
            raise ValueError(
                f"domain must be one of {sorted(VALID_DOMAINS)}, got {domain!r}"
            )
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.judge = judge
        self.domain = domain
        self.cost_per_call = cost_per_call
        self.confirm = confirm
        self.tests = tests
        self.max_items_per_test = max_items_per_test
        self.max_workers = max_workers

    # ── public API ─────────────────────────────────────────────────────────────

    def audit(self) -> AuditReport:
        # 1. Resolve judge type
        judge_type = resolve_judge_type(self.judge.eval_template, self.judge.judge_type)

        # 2. Probe: verify template formats and judge is callable
        self._probe()

        # 3. Select applicable tests
        applicable = [t for t in ALL_TESTS if t.is_applicable(judge_type)]
        if self.tests is not None:
            applicable = [t for t in applicable if t.name in self.tests]

        # 4. Load fixtures; record skips for missing files
        fixture_map: dict[str, list[dict]] = {}
        skip_reasons: dict[str, str] = {}
        for test in applicable:
            path = test.fixture_path.format(domain=self.domain)
            try:
                fixture_map[test.name] = load_fixtures(
                    path, max_items=self.max_items_per_test
                )
            except FixtureNotFound:
                skip_reasons[test.name] = f"no fixture file at {path!r}"

        runnable = [t for t in applicable if t.name in fixture_map]

        # 5. Estimate + confirm
        est = self._build_estimate(runnable, fixture_map)
        self._print_header(judge_type)
        est.display()
        if self.confirm:
            self._ask_confirmation()

        # 6. Run tests
        call_counter = CallCounter()
        results: dict[str, BiasResult] = {}
        for test in runnable:
            results[test.name] = test.run(
                self.judge, fixture_map[test.name], call_counter, self.max_workers
            )

        # 7. N/A entries for non-applicable and missing-fixture tests
        for test in ALL_TESTS:
            if test.name in results:
                continue
            reason = skip_reasons.get(
                test.name,
                f"{test.name} does not apply to {judge_type.value} judges",
            )
            results[test.name] = BiasResult(
                test_name=test.name,
                score=0.0,
                verdict="N/A",
                n_fixtures=0,
                not_applicable=True,
                skip_reason=reason,
            )

        # 8. Rank applicable results worst-first
        ranked = sorted(
            [r for r in results.values() if not r.not_applicable],
            key=lambda r: r.score,
        )
        for i, r in enumerate(ranked):
            r.rank = i + 1

        # 9. Return report
        return AuditReport(
            judge_name=self.judge.judge_name,
            judge_type=judge_type.value,
            domain=self.domain,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            tests=results,
        )

    def estimate(self) -> CostEstimate:
        judge_type = resolve_judge_type(self.judge.eval_template, self.judge.judge_type)
        applicable = [t for t in ALL_TESTS if t.is_applicable(judge_type)]
        if self.tests is not None:
            applicable = [t for t in applicable if t.name in self.tests]

        fixture_map: dict[str, list[dict]] = {}
        for test in applicable:
            path = test.fixture_path.format(domain=self.domain)
            try:
                fixture_map[test.name] = load_fixtures(
                    path, max_items=self.max_items_per_test
                )
            except FixtureNotFound:
                pass

        runnable = [t for t in applicable if t.name in fixture_map]
        return self._build_estimate(runnable, fixture_map)

    # ── internal ───────────────────────────────────────────────────────────────

    def _probe(self) -> None:
        """Call judge once with dummy inputs to verify it's reachable."""
        placeholders = set(re.findall(r"\{(\w+)\}", self.judge.eval_template))
        dummy = {k: f"[{k}]" for k in placeholders}
        try:
            prompt = self.judge.eval_template.format(**dummy)
            self.judge.llm_fn(prompt)
        except Exception as exc:
            raise DetectionError(f"Judge probe call failed: {exc}") from exc

    def _build_estimate(
        self,
        runnable: list,
        fixture_map: dict[str, list[dict]],
    ) -> CostEstimate:
        return estimate_calls(
            test_names=[t.name for t in runnable],
            fixture_counts={t.name: len(fixture_map[t.name]) for t in runnable},
            cost_per_call=self.cost_per_call,
        )

    def _print_header(self, judge_type: JudgeType) -> None:
        jt_label = judge_type.value
        if not self.judge.judge_type:
            jt_label += " (auto-detected)"
        sep = "-" * 41
        print(f"\n{sep}")
        print(f"  Judge:       {self.judge.judge_name}")
        print(f"  Judge type:  {jt_label}")
        print(f"  Domain:      {self.domain}")
        print(f"{sep}\n")

    def _ask_confirmation(self) -> None:
        if not sys.stdin.isatty():
            warnings.warn(
                "Non-TTY stdin detected; skipping confirm prompt.",
                stacklevel=3,
            )
            return
        answer = input("Proceed? [Y/n]: ").strip().lower()
        if answer == "n":
            raise AuditCancelled("user declined to proceed")
