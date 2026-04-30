#!/usr/bin/env python3
"""Parallel fixture verifier — one process per JSONL file."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "judicator" / "fixtures" / "data"

REQUIRED_FIELDS = {"id", "license", "source_dataset"}


@dataclass
class Result:
    path: str
    ok: bool
    reason: Optional[str] = None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classify(path: Path) -> str:
    name = path.name
    return {
        "seeds.jsonl":            "seeds",
        "verbosity.jsonl":        "verbosity",
        "authority.jsonl":        "authority",
        "position.jsonl":         "position",
        "yesno_probes.jsonl":     "yesno",
        "scale_tiers.jsonl":      "scale_tiers",
        "concreteness.jsonl":     "concreteness",
        "self_consistency.jsonl": "self_consistency",
    }.get(name, "unknown")


def verify_one_file(path_str: str) -> Result:
    path = Path(path_str)
    try:
        items = load_jsonl(path)
        if not items:
            return Result(path_str, False, "empty file")

        for i, it in enumerate(items):
            missing = REQUIRED_FIELDS - set(it.keys())
            if missing:
                return Result(path_str, False, f"item {i} missing fields: {missing}")
            if not it.get("license"):
                return Result(path_str, False, f"item {i} missing license value")

        kind = classify(path)

        if kind == "seeds":
            for i, it in enumerate(items):
                if not it.get("question") or not it.get("response"):
                    return Result(path_str, False,
                                  f"item {i}: missing question or response")
                if "quality_tier" in it:
                    return Result(path_str, False,
                                  f"item {i}: quality_tier should not be in seeds")

        elif kind == "verbosity":
            for i, it in enumerate(items):
                if not it.get("content_preserved"):
                    return Result(path_str, False,
                                  f"item {i}: content_preserved must be True")
                if len(it["manipulated_response"]) <= len(it["original_response"]):
                    return Result(path_str, False,
                                  f"item {i}: padded response not longer than original")

        elif kind == "authority":
            for i, it in enumerate(items):
                if it.get("manipulation") != "authority_injection":
                    return Result(path_str, False,
                                  f"item {i}: manipulation must be 'authority_injection'")

        elif kind == "position":
            for i, it in enumerate(items):
                if not it.get("winner_response") or not it.get("loser_response"):
                    return Result(path_str, False,
                                  f"item {i}: missing winner_response or loser_response")
                if it["winner_response"] == it["loser_response"]:
                    return Result(path_str, False,
                                  f"item {i}: winner and loser are identical")

        elif kind == "concreteness":
            for i, it in enumerate(items):
                if not it.get("vague_response") or not it.get("concrete_response"):
                    return Result(path_str, False,
                                  f"item {i}: missing vague_response or concrete_response")

        elif kind == "scale_tiers":
            tiers = Counter(it.get("tier") for it in items)
            for t in ("high", "low"):
                if tiers[t] < 10:
                    return Result(path_str, False,
                                  f"tier {t!r} has {tiers[t]} items (need >=10)")
            unexpected = set(tiers) - {"high", "low"}
            if unexpected:
                return Result(path_str, False,
                              f"unexpected tier values: {unexpected}")

        elif kind == "self_consistency":
            if len(items) != 40:
                return Result(path_str, False,
                              f"expected exactly 40 items, got {len(items)}")
            for i, it in enumerate(items):
                if not it.get("source_scale_tier_id"):
                    return Result(path_str, False,
                                  f"item {i}: missing source_scale_tier_id")

        elif kind == "yesno":
            t = sum(1 for it in items if it.get("ground_truth") is True)
            f = sum(1 for it in items if it.get("ground_truth") is False)
            if abs(t - f) > 5:
                return Result(path_str, False,
                              f"yes/no imbalance: {t} true, {f} false (max delta 5)")

        return Result(path_str, True)

    except Exception as e:
        return Result(path_str, False, f"{type(e).__name__}: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()

    files = sorted(glob.glob(str(DATA_DIR / "**" / "*.jsonl"), recursive=True))
    if not files:
        sys.exit(f"no fixture files found under {DATA_DIR}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(verify_one_file, files))

    failures = [r for r in results if not r.ok]
    for r in failures:
        print(f"FAIL: {r.path} — {r.reason}", file=sys.stderr)

    if failures:
        sys.exit(1)
    print(f"PASS: all {len(results)} fixture files verified")


if __name__ == "__main__":
    main()
