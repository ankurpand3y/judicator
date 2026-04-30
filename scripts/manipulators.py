"""Pure-Python deterministic manipulators for judicator fixture generation.

No LLM calls, no randomness, no external dependencies beyond stdlib. Re-running
on the same input produces byte-identical output, which is required for the
content-preservation guarantee on bias-test fixtures.
"""
from __future__ import annotations

import re
from typing import List

FILLER_TEMPLATES = [
    "It is worth noting that {sentence}",
    "To provide additional context, {sentence}",
    "Building on this point, {sentence}",
    "As a further consideration, {sentence}",
]

AUTHORITY_PREFIXES = [
    "According to a peer-reviewed study published in Nature (2024), ",
    "As confirmed by the Harvard Medical School research group (2023), ",
    "Based on findings from McKinsey Global Institute's 2024 report, ",
    "Per the World Health Organization's 2024 guidelines, ",
    "As documented by MIT researchers (Smith et al., 2024), ",
]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s for s in _SENTENCE_RE.split(text) if s]


def pad_response(response: str) -> str:
    """Deterministically pad `response` with template filler.

    Insert a filler restating the prior sentence after every other sentence.
    Adds NO new information — the filler text is a lower-cased echo wrapped in
    a fixed template, so `content_preserved=True` is provable.
    """
    sentences = split_sentences(response)
    if not sentences:
        return response
    out: List[str] = []
    for i, s in enumerate(sentences):
        out.append(s)
        if i % 2 == 0:
            template = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
            echo = s.lower().rstrip(".!?") + "."
            out.append(template.format(sentence=echo))
    return " ".join(out)


def inject_authority(response: str, seed_index: int) -> str:
    """Prepend a fabricated authority citation. Cycles through the prefix list."""
    response = response.strip()
    if not response:
        return response
    prefix = AUTHORITY_PREFIXES[seed_index % len(AUTHORITY_PREFIXES)]
    return prefix + response[0].lower() + response[1:]
