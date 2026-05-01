from dataclasses import dataclass
from typing import Callable

VALID_JUDGE_TYPES: set[str | None] = {"pointwise", "pairwise", "binary", None}


@dataclass
class Judge:
    # Must be stateless — called independently per fixture with no shared context
    llm_fn: Callable[[str], str]
    system_prompt: str
    eval_template: str
    judge_name: str = "unnamed_judge"
    judge_type: str | None = None

    def __post_init__(self) -> None:
        if self.judge_type not in VALID_JUDGE_TYPES:
            raise ValueError(
                f"judge_type must be one of {VALID_JUDGE_TYPES}, got {self.judge_type!r}"
            )
        if "{" not in self.eval_template or "}" not in self.eval_template:
            raise ValueError("eval_template must contain at least one {placeholder}")
