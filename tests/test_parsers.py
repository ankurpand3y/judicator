import pytest
from judicator.parsers import parse_binary, parse_pairwise, parse_pointwise


# ── parse_pointwise ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("I'd give this a 7.", 7.0),
    ("Score: 8.5", 8.5),
    ("Rating: 10", 10.0),
    ("This deserves a 1 out of 10.", 1.0),
    ("The response earns a 6.", 6.0),
    ("I rate it 9.5 overall.", 9.5),
])
def test_parse_pointwise_valid(text: str, expected: float) -> None:
    assert parse_pointwise(text) == expected


@pytest.mark.parametrize("text", [
    "This is excellent.",
    "No numbers here.",
    "There are 25 items in the list.",
    "The year 2024 was productive.",
])
def test_parse_pointwise_none(text: str) -> None:
    assert parse_pointwise(text) is None


# ── parse_pairwise ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Response A is better.", "A"),
    ("I prefer B.", "B"),
    ("A is clearly superior here.", "A"),
    ("The winner is B.", "B"),
    ("A wins.", "A"),
])
def test_parse_pairwise_valid(text: str, expected: str) -> None:
    assert parse_pairwise(text) == expected


def test_parse_pairwise_first_token_wins() -> None:
    assert parse_pairwise("A is better than B here.") == "A"


@pytest.mark.parametrize("text", [
    "Neither is good.",
    "They are equal.",
    "Cannot decide.",
])
def test_parse_pairwise_none(text: str) -> None:
    assert parse_pairwise(text) is None


# ── parse_binary ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Yes, this is correct.", "Yes"),
    ("No, this statement is false.", "No"),
    ("YES", "Yes"),
    ("NO", "No"),
    ("yes.", "Yes"),
    ("The answer is no.", "No"),
])
def test_parse_binary_valid(text: str, expected: str) -> None:
    assert parse_binary(text) == expected


def test_parse_binary_yes_takes_precedence() -> None:
    # "yes" appears before "no" — should return "Yes"
    assert parse_binary("yes or no") == "Yes"


@pytest.mark.parametrize("text", [
    "Maybe.",
    "Possibly.",
    "Hard to say.",
])
def test_parse_binary_none(text: str) -> None:
    assert parse_binary(text) is None
