import pytest
from judicator.exceptions import FixtureNotFound
from judicator.fixtures import FIXTURE_VERSION
from judicator.fixtures.loader import load_fixtures

# ── version ────────────────────────────────────────────────────────────────────

def test_fixture_version() -> None:
    assert FIXTURE_VERSION == "0.1.0"


# ── happy path ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel_path,min_count", [
    ("qa/seeds.jsonl",                    100),
    ("qa/verbosity.jsonl",                100),
    ("qa/authority.jsonl",                100),
    ("qa/position.jsonl",                 100),
    ("code/seeds.jsonl",                   83),
    ("code/verbosity.jsonl",               83),
    ("code/authority.jsonl",               83),
    ("code/position.jsonl",                70),
    ("summarization/seeds.jsonl",         100),
    ("summarization/verbosity.jsonl",     100),
    ("summarization/authority.jsonl",     100),
    ("safety/seeds.jsonl",                100),
    ("safety/verbosity.jsonl",            100),
    ("safety/authority.jsonl",            100),
    ("dialogue/seeds.jsonl",               50),
    ("dialogue/verbosity.jsonl",           50),
    ("dialogue/authority.jsonl",           50),
    ("universal/scale_tiers.jsonl",       200),
    ("universal/self_consistency.jsonl",   40),
    ("universal/concreteness.jsonl",       14),
    ("universal/yesno_probes.jsonl",      100),
])
def test_fixture_counts(rel_path: str, min_count: int) -> None:
    items = load_fixtures(rel_path)
    assert len(items) >= min_count, f"{rel_path}: expected >={min_count}, got {len(items)}"


def test_load_returns_dicts() -> None:
    items = load_fixtures("qa/verbosity.jsonl")
    assert all(isinstance(item, dict) for item in items)


def test_max_items_cap() -> None:
    items = load_fixtures("qa/verbosity.jsonl", max_items=5)
    assert len(items) == 5


def test_max_items_larger_than_file() -> None:
    items = load_fixtures("universal/concreteness.jsonl", max_items=9999)
    assert len(items) == 14


def test_max_items_zero() -> None:
    items = load_fixtures("qa/verbosity.jsonl", max_items=0)
    assert items == []


# ── required fields ────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {"id", "license", "source_dataset"}

@pytest.mark.parametrize("rel_path", [
    "qa/seeds.jsonl",
    "qa/verbosity.jsonl",
    "qa/authority.jsonl",
    "qa/position.jsonl",
    "universal/scale_tiers.jsonl",
    "universal/self_consistency.jsonl",
    "universal/concreteness.jsonl",
    "universal/yesno_probes.jsonl",
])
def test_required_fields_present(rel_path: str) -> None:
    for item in load_fixtures(rel_path):
        missing = REQUIRED_FIELDS - item.keys()
        assert not missing, f"{rel_path} item {item.get('id')} missing {missing}"


# ── domain-specific fields ─────────────────────────────────────────────────────

def test_scale_tiers_tier_field() -> None:
    items = load_fixtures("universal/scale_tiers.jsonl")
    tiers = {item["tier"] for item in items}
    assert tiers == {"high", "low"}
    high = sum(1 for i in items if i["tier"] == "high")
    low  = sum(1 for i in items if i["tier"] == "low")
    assert high >= 10 and low >= 10


def test_self_consistency_count() -> None:
    assert len(load_fixtures("universal/self_consistency.jsonl")) == 40


def test_position_has_winner_loser() -> None:
    for item in load_fixtures("qa/position.jsonl"):
        assert "winner_response" in item
        assert "loser_response" in item
        assert item["winner_response"] != item["loser_response"]


def test_verbosity_has_pair_fields() -> None:
    for item in load_fixtures("qa/verbosity.jsonl"):
        assert "original_response" in item
        assert "manipulated_response" in item


def test_concreteness_has_pair_fields() -> None:
    for item in load_fixtures("universal/concreteness.jsonl"):
        assert "vague_response" in item
        assert "concrete_response" in item


def test_yesno_has_ground_truth() -> None:
    items = load_fixtures("universal/yesno_probes.jsonl")
    true_count  = sum(1 for i in items if i["ground_truth"] is True)
    false_count = sum(1 for i in items if i["ground_truth"] is False)
    assert true_count >= 40 and false_count >= 40


# ── error path ─────────────────────────────────────────────────────────────────

def test_missing_fixture_raises() -> None:
    with pytest.raises(FixtureNotFound):
        load_fixtures("nonexistent.jsonl")


def test_missing_domain_fixture_raises() -> None:
    with pytest.raises(FixtureNotFound):
        load_fixtures("summarization/position.jsonl")
