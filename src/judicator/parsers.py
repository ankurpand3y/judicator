from __future__ import annotations
import re


def parse_pointwise(text: str) -> float | None:
    """Extract first number in [1, 10] from judge output. Returns None if not found."""
    for m in re.findall(r"\b(\d+(?:\.\d+)?)\b", text):
        val = float(m)
        if 1.0 <= val <= 10.0:
            return val
    return None


def parse_pairwise(text: str) -> str | None:
    """Return 'A' or 'B' (first verdict token found). Returns None if neither."""
    m = re.search(r"\b(A|B)\b", text)
    return m.group(1) if m else None


def parse_binary(text: str) -> str | None:
    """Return 'Yes' or 'No'. Returns None if neither found."""
    lower = text.strip().lower()
    if re.search(r"\byes\b", lower):
        return "Yes"
    if re.search(r"\bno\b", lower):
        return "No"
    return None
