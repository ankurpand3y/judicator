import pytest
import judicator
from judicator._types import JudgeType
from judicator.detector import detect_judge_type, resolve_judge_type
from judicator.exceptions import DetectionError
from judicator.judge import Judge

# ── template sets ──────────────────────────────────────────────────────────────

PAIRWISE = [
    "Question: {question}\nA: {response_a}\nB: {response_b}\nWhich is better?",
    "Compare {response_a} and {response_b} for the query: {question}",
    "Task: {question}\nResponse A: {response_a}\nResponse B: {response_b}\nPick A or B.",
    "Evaluate {response_a} vs {response_b} for {question}.",
    "{question}\n\nOption A: {response_a}\nOption B: {response_b}\nWhich is superior?",
]

BINARY = [
    "Is this a good answer? {response}\nAnswer yes or no.",
    "Statement: {statement}\nRespond with yes or no.",
    "Evaluate: {response}\nAnswer only yes or no.",
    "Is the following true or false? {statement}",
    "Answer only yes or no: is this {response} correct?",
]

POINTWISE = [
    "Question: {question}\nResponse: {response}\nScore 1-10.",
    "Rate this {answer} on a scale of 1-10.",
    "Evaluate the following {completion}. Provide a numeric score.",
    "Grade this {output} from 1 to 10.",
    "Assess this {response} and give a score between 1 and 10.",
]

AMBIGUOUS = [
    "Evaluate this.",
    "Tell me about {topic}.",
    "Is {x} better than {y}?",
    "Summarize the {document}.",
]


# ── detection tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("template", PAIRWISE)
def test_pairwise_detected(template: str) -> None:
    jt, conf = detect_judge_type(template)
    assert jt == JudgeType.PAIRWISE
    assert conf >= 0.90


@pytest.mark.parametrize("template", BINARY)
def test_binary_detected(template: str) -> None:
    jt, conf = detect_judge_type(template)
    assert jt == JudgeType.BINARY
    assert conf >= 0.85


@pytest.mark.parametrize("template", POINTWISE)
def test_pointwise_detected(template: str) -> None:
    jt, conf = detect_judge_type(template)
    assert jt == JudgeType.POINTWISE
    assert conf >= 0.80


@pytest.mark.parametrize("template", AMBIGUOUS)
def test_ambiguous_returns_unknown(template: str) -> None:
    jt, conf = detect_judge_type(template)
    assert jt == JudgeType.UNKNOWN
    assert conf == 0.0


def test_pairwise_takes_precedence_over_binary() -> None:
    t = "Compare {response_a} and {response_b}. Answer yes or no."
    jt, _ = detect_judge_type(t)
    assert jt == JudgeType.PAIRWISE


# ── resolve tests ──────────────────────────────────────────────────────────────

def test_resolve_raises_on_unknown() -> None:
    with pytest.raises(DetectionError):
        resolve_judge_type("Evaluate this.", override=None)


def test_resolve_override_pointwise() -> None:
    jt = resolve_judge_type("Evaluate this.", override="pointwise")
    assert jt == JudgeType.POINTWISE


def test_resolve_override_pairwise() -> None:
    jt = resolve_judge_type("Evaluate this.", override="pairwise")
    assert jt == JudgeType.PAIRWISE


def test_resolve_override_binary() -> None:
    jt = resolve_judge_type("Evaluate this.", override="binary")
    assert jt == JudgeType.BINARY


def test_resolve_invalid_override() -> None:
    with pytest.raises(ValueError):
        resolve_judge_type("Evaluate this.", override="listwise")


# ── Judge validation ───────────────────────────────────────────────────────────

def test_judge_invalid_type_raises() -> None:
    with pytest.raises(ValueError):
        Judge(llm_fn=lambda p: p, system_prompt="s", eval_template="{x}",
              judge_type="invalid")


def test_judge_no_placeholder_raises() -> None:
    with pytest.raises(ValueError):
        Judge(llm_fn=lambda p: p, system_prompt="s", eval_template="no braces here")


def test_judge_valid_constructs() -> None:
    j = Judge(llm_fn=lambda p: p, system_prompt="s", eval_template="{response}")
    assert j.judge_name == "unnamed_judge"
    assert j.judge_type is None


# ── public API surface ─────────────────────────────────────────────────────────

def test_judge_type_not_top_level() -> None:
    assert not hasattr(judicator, "JudgeType")


def test_only_judge_and_auditor_exported() -> None:
    assert set(judicator.__all__) == {"Judge", "JudgeAuditor"}
