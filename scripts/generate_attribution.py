#!/usr/bin/env python3
"""Auto-generate ATTRIBUTION.md by aggregating source_dataset metadata
across every fixture JSONL file.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "judicator" / "fixtures" / "data"

SOURCE_META = {
    "JudgeBench": {
        "url":     "github.com/ScalerLab/JudgeBench",
        "paper":   "Tan et al. 2024, arxiv 2410.12784",
        "license": "MIT",
    },
    "MT-Bench": {
        "url":     "github.com/lm-sys/FastChat",
        "paper":   "Zheng et al. 2023, arxiv 2306.05685",
        "license": "Apache 2.0",
    },
    "BeaverTails": {
        "url":     "github.com/PKU-Alignment/beavertails",
        "paper":   "Ji et al. 2023, arxiv 2307.04657",
        "license": "CC-BY-NC-4.0",
    },
    "OffsetBias": {
        "url":     "github.com/ncsoft/offsetbias",
        "paper":   "Park et al. 2024, arxiv 2407.06551",
        "license": "Apache 2.0",
    },
    "judicator-curated-v0.1": {
        "url":     "github.com/ankurpand3y/judicator",
        "paper":   "(none — hand-authored)",
        "license": "Apache-2.0",
    },
    "SummEval": {
        "url":     "huggingface.co/datasets/mteb/summeval",
        "paper":   "Fabbri et al. 2021, arxiv 2007.12626",
        "license": "MIT",
    },
    "DSTC11-Track4": {
        "url":     "huggingface.co/datasets/mario-rc/dstc11.t4",
        "paper":   "Rodriguez-Cantelar et al. 2023, DSTC11 Track 4",
        "license": "Apache 2.0",
    },
}


def main() -> None:
    counts: dict[str, int] = defaultdict(int)
    for path in glob.glob(str(DATA_DIR / "**" / "*.jsonl"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                ds = item.get("source_dataset") or "unknown"
                counts[ds] += 1

    lines = [
        "# Judicator — Data Attribution",
        "",
        "Judicator ships with fixtures derived from the following datasets.",
        "All are used in accordance with their licenses.",
        "",
    ]

    for ds in sorted(counts.keys()):
        meta = SOURCE_META.get(ds, {})
        lines += [
            f"## {ds}",
            f"- Source: {meta.get('url', '(not on file)')}",
            f"- Paper: {meta.get('paper', '(not on file)')}",
            f"- License: {meta.get('license', '(not on file)')}",
            f"- Items used: {counts[ds]}",
            "",
        ]

    out = ROOT / "ATTRIBUTION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({sum(counts.values())} items across {len(counts)} sources)")


if __name__ == "__main__":
    main()
