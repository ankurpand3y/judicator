from __future__ import annotations
import json
from pathlib import Path

from judicator.exceptions import FixtureNotFound

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_fixtures(rel_path: str, max_items: int | None = None) -> list[dict]:
    """
    Load fixtures from a path relative to src/judicator/fixtures/data/.

    Examples:
        load_fixtures("qa/verbosity.jsonl")
        load_fixtures("universal/concreteness.jsonl")
        load_fixtures("qa/position.jsonl", max_items=20)

    Raises FixtureNotFound if the file does not exist.
    No domain-fallback magic — each bias test declares its exact fixture path.
    """
    path = DATA_DIR / rel_path
    if not path.exists():
        raise FixtureNotFound(f"No fixtures at {rel_path!r} (looked in {path})")

    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_items is not None:
        items = items[:max_items]
    return items
