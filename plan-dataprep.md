# Judicator — Data Curation Planner

> **Who runs this:** You, the library author, once before publishing v0.1.
> **What it produces:** JSONL fixture files committed to the repo under `src/judicator/fixtures/data/`.
> **Users never run this.** It is not part of the installed package.
> **Tooling lives in:** `scripts/` directory, gitignored alongside raw source downloads.

---

## 1. Purpose

Judicator ships with pre-computed fixtures so users need zero external data to run an audit.
This planner documents how those fixtures are produced — reproducibly, with full attribution.

Every fixture file that ships with the library must:
- Have a documented source (dataset name, license)
- Be deterministically reproducible from the source dataset
- Carry attribution in `ATTRIBUTION.md`

---

## 2. Output Structure (what gets committed)

```
src/judicator/fixtures/data/
│
├── universal/
│   ├── scale_tiers.jsonl         # 2-bucket (high/low) — SummEval + DSTC11 only
│   ├── self_consistency.jsonl    # 40 items sampled from scale_tiers
│   ├── concreteness.jsonl        # OffsetBias 'concreteness' subcategory pairs
│   └── yesno_probes.jsonl        # balanced T/F statements (hand-curated)
│
├── qa/
│   ├── seeds.jsonl               # flat (question, response) pairs
│   ├── verbosity.jsonl           # (original, padded) pairs
│   ├── authority.jsonl           # (original, authority-prefixed) pairs
│   └── position.jsonl            # winner/loser pairs from JudgeBench mmlu-pro
│
├── code/
│   ├── seeds.jsonl
│   ├── verbosity.jsonl
│   ├── authority.jsonl
│   └── position.jsonl            # winner/loser pairs from JudgeBench livecodebench
│
├── summarization/
│   ├── seeds.jsonl
│   ├── verbosity.jsonl
│   └── authority.jsonl
│
├── safety/
│   ├── seeds.jsonl
│   ├── verbosity.jsonl
│   └── authority.jsonl
│
└── dialogue/
    ├── seeds.jsonl
    ├── verbosity.jsonl
    └── authority.jsonl
```

**Why no `position.jsonl` in summarization/safety/dialogue (v0.1):** JudgeBench provides
explicit winner/loser pairs only for qa (mmlu-pro) and code (livecodebench). For other
domains we'd need to synthesize pairs — deferred to v0.2.

**Total v0.1 actuals: ~1,825 items, 21 files. Well under the 5 MB package budget.**

---

## 3. Source Datasets

| Dataset | URL | License | Used for |
|---|---|---|---|
| OffsetBias / EvalBiasBench | github.com/ncsoft/offsetbias | Apache 2.0 | `universal/concreteness.jsonl` (concreteness subcategory only) |
| JudgeBench | github.com/ScalerLab/JudgeBench | MIT | `qa/seeds.jsonl`, `code/seeds.jsonl`, `qa/position.jsonl`, `code/position.jsonl` |
| MT-Bench | github.com/lm-sys/FastChat | Apache 2.0 | `qa/seeds.jsonl` (refs), `code/seeds.jsonl` (refs) |
| BeaverTails | github.com/PKU-Alignment/beavertails (HuggingFace `PKU-Alignment/BeaverTails`) | CC-BY-NC-4.0 | `safety/seeds.jsonl` — **NON-COMMERCIAL only** |
| SummEval | huggingface.co/datasets/mteb/summeval | MIT | `summarization/seeds.jsonl`, `universal/scale_tiers.jsonl` |
| DSTC11 Track 4 | huggingface.co/datasets/mario-rc/dstc11.t4 | Apache 2.0 | `dialogue/seeds.jsonl`, `universal/scale_tiers.jsonl` |
| Hand-curated | (repo) | Apache 2.0 | `universal/yesno_probes.jsonl` |

**License notes:**
- BeaverTails is CC-BY-NC-4.0. **Safety fixtures cannot be used commercially.** If
  judicator will be used commercially, exclude `safety/*.jsonl` from the package
  or replace with synthetic safety probes.
- All other sources are commercially permissive.
- License field is preserved per-item in every fixture, so downstream consumers can filter.

---

## 4. Architectural Principles

Three principles drive the data layout:

1. **Each fixture file exists because exactly one bias test reads it, in one specific way.**
   No reusable infrastructure beyond what's used. No fallback magic in the loader.

2. **Every label reflects a real measurement.** `quality_tier` was dropped from seeds because
   it was assigned heuristically (winner/loser, length proxy, fiat) and only one test uses
   tiered data. Scale tiers now live exclusively in `scale_tiers.jsonl`, drawn only from
   SummEval and DSTC11 — the two sources with real, multi-annotator human ratings.

3. **Manipulations are deterministic and pure-Python.** No LLM during fixture generation.
   Verbosity/authority manipulators echo the input through fixed templates so
   `content_preserved=True` is provable, not aspirational.

---

## 5. Fixture-to-Test Mapping

| Bias test | Fixture file | Source signal |
|---|---|---|
| `position` | `{qa,code}/position.jsonl` | JudgeBench winner/loser labels |
| `verbosity` | `{domain}/verbosity.jsonl` | (original, padded) — deterministic template |
| `self_consistency` | `universal/self_consistency.jsonl` | sampled from scale_tiers (40 items) |
| `scale_anchoring` | `universal/scale_tiers.jsonl` | SummEval (3 raters × 4 dims) + DSTC11 (5 raters × OVERALL) |
| `authority` | `{domain}/authority.jsonl` | (original, authority-prefixed) — deterministic template |
| `concreteness` | `universal/concreteness.jsonl` | OffsetBias 'concreteness' natural pairs |
| `yes_bias` | `universal/yesno_probes.jsonl` | hand-curated T/F facts |

---

## 6. Scripts

```
scripts/
├── download_sources.sh         # parallel git clones (4 repos), fail-fast
├── generate_fixtures.py        # main orchestrator (Steps 1-4), DAG-driven, parallel
├── manipulators.py             # pad_response(), inject_authority() — pure-Python templates
├── verify_fixtures.py          # parallel sanity checks, one process per file
├── generate_attribution.py     # auto-generates ATTRIBUTION.md from fixture metadata
└── handcurated/
    └── yesno_probes.json       # 100 hand-authored T/F statements, 50T/50F balance
```

**One-command build:**
```bash
bash scripts/download_sources.sh
python scripts/generate_fixtures.py --workers $(nproc) --all
python scripts/verify_fixtures.py
python scripts/generate_attribution.py
```

---

## 7. Pipeline Stages

```
Step 0   parallel git clones (network-bound, 4 repos)
   │
Step 1   per-domain seed extraction        (5 parallel workers)
   │       qa, summarization, code, safety, dialogue
   │
Step 2   position pairs (qa, code only)    (2 parallel workers)
   │
Step 3   verbosity + authority × 5 + concreteness    (11 parallel workers)
   │
Step 4a  scale_tiers (sequential, cross-source aggregation from SummEval + DSTC11)
   │
Step 4b  self_consistency (sequential, samples from 4a output)
   │
Step 4c  yesno_probes (sequential, hand-curated JSON)
   │
Step 5   verifier (N parallel workers, one per file)
   │
Step 6   ATTRIBUTION.md (sequential)
```

**Implementation primitives:**
- `concurrent.futures.ProcessPoolExecutor` for CPU-bound steps.
- Background shell jobs (`&` + `wait`) for the git clones.
- Each worker writes its own JSONL — no shared mutable state, no locking.
- Stage-level barriers only.
- All randomness removed (deterministic templates + stride sampling).

**Wall-clock target on 8-core laptop:** under 2 minutes for the full build,
excluding the one-time source download. SummEval + DSTC11 HF downloads cache locally.

---

## 8. Step 1 — Per-domain seed extraction

Flat (question, response) pairs per domain. **No `quality_tier` in seeds** — all tier
information lives in `scale_tiers.jsonl`.

| Domain | Source | Approach | Actual count |
|---|---|---|---|
| qa | MT-Bench reasoning/humanities/stem refs + JudgeBench mmlu-pro winners | Concat, dedupe, cap at 100 | 100 |
| summarization | SummEval | Stride-sample 100 across rating distribution for diversity | 100 |
| code | MT-Bench coding refs + JudgeBench livecodebench winners | Concat, dedupe (~83 total) | 83 |
| safety | BeaverTails | First 60 safe + 40 unsafe (carry `is_safe` flag for transparency) | 100 |
| dialogue | DSTC11 fed-turn + fed-dial | Stride-sample 50 across rating distribution | 50 |

**Seed schema (uniform across domains):**
```json
{
  "id": "qa_001",
  "domain": "qa",
  "question": "What causes lightning?",
  "response": "Lightning is caused by electrical discharge...",
  "source_dataset": "MT-Bench",
  "license": "Apache 2.0"
}
```

Domain-specific extra fields are allowed (e.g., `is_safe` for safety, `rated_model` /
`sub_dataset` for dialogue) but the core 6 fields above are mandatory.

**Note on dialogue domain:** DSTC11 was the cleanest publicly-available source pairing
(context, response) turns with explicit human ratings under a permissive license.
DSTC11 dialogues are **open-domain chitchat / chatbot evaluation**, not e-commerce
customer support — hence the domain is named `dialogue`, not `customer_support`.

---

## 9. Step 2 — Position pairs (qa, code)

Position bias is a pairwise-only test. JudgeBench provides explicit winner/loser pairs
for two of our domains:

- `qa/position.jsonl` ← JudgeBench mmlu-pro (excluding math), capped at 100 pairs
- `code/position.jsonl` ← JudgeBench livecodebench, all valid pairs (~72 after dedup)

**Schema:**
```json
{
  "id": "qa_position_001",
  "bias_type": "position",
  "domain": "qa",
  "question": "...",
  "winner_response": "...",
  "loser_response": "...",
  "source_split": "mmlu-pro-physics",
  "license": "MIT",
  "source_dataset": "JudgeBench"
}
```

**Filters applied:**
- Skip items with label="tie" (no winner)
- Skip items where `winner_response == loser_response` (data noise — both models produced
  identical text)
- Dedupe by `pair_id`

Other domains (summarization, safety, dialogue) get no `position.jsonl` in v0.1 — pairwise
judges in those domains will see this test as `not_applicable=True` with a clear
`skip_reason`. Synthesizing pairs for those domains is deferred to v0.2.

---

## 10. Step 3 — Manipulators (verbosity, authority, concreteness)

### 10a. Verbosity (all 5 domains)

```python
FILLER_TEMPLATES = [
    "It is worth noting that {sentence}",
    "To provide additional context, {sentence}",
    "Building on this point, {sentence}",
    "As a further consideration, {sentence}",
]

def pad_response(response: str) -> str:
    sentences = split_sentences(response)
    out = []
    for i, s in enumerate(sentences):
        out.append(s)
        if i % 2 == 0:
            template = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
            out.append(template.format(sentence=s.lower().rstrip(".!?") + "."))
    return " ".join(out)
```

The padded text is a literal echo of the original, wrapped in fixed templates.
Adds no new information → `content_preserved=True` is provable.

### 10b. Authority (all 5 domains, including safety)

```python
AUTHORITY_PREFIXES = [
    "According to a peer-reviewed study published in Nature (2024), ",
    "As confirmed by the Harvard Medical School research group (2023), ",
    "Based on findings from McKinsey Global Institute's 2024 report, ",
    "Per the World Health Organization's 2024 guidelines, ",
    "As documented by MIT researchers (Smith et al., 2024), ",
]

def inject_authority(response: str, seed_index: int) -> str:
    prefix = AUTHORITY_PREFIXES[seed_index % len(AUTHORITY_PREFIXES)]
    return prefix + response[0].lower() + response[1:]
```

The authority sources are **fabricated**. Intentional — the test probes whether the
judge rewards framing regardless of source veracity. Fixed prefix list keeps fabrications
minimal, obvious, and uniform.

**Verbosity/authority output schema (uniform):**
```json
{
  "id": "qa_001_verbosity",
  "domain": "qa",
  "bias_type": "verbosity",
  "question": "...",
  "original_response": "...",
  "manipulated_response": "...",
  "manipulation": "padding",
  "content_preserved": true,
  "source_seed_id": "qa_001",
  "license": "Apache 2.0",
  "source_dataset": "MT-Bench"
}
```

### 10c. Concreteness (universal, OffsetBias)

OffsetBias ships natural pairs across 6 stylistic-bias categories. We use **only the
`concreteness` subcategory** — pairs where `response2` cites fabricated-sounding
specifics (numbers, named studies) and `response1` gives an equivalently-correct vague
answer. A concreteness-biased judge over-rewards the fabricated-specifics version.

**Schema:**
```json
{
  "id": "concreteness_001",
  "bias_type": "concreteness",
  "question": "...",
  "vague_response": "...",       // OffsetBias response1 — preferred / correct
  "concrete_response": "...",    // OffsetBias response2 — specifics-loaded / biased
  "source_dataset": "OffsetBias",
  "license": "Apache 2.0"
}
```

**v0.1 count: 14 items.** Small but pure. Other OffsetBias categories (length bias,
empty reference, etc.) test different biases and are not included here.

---

## 11. Step 4a — Scale tiers (high/low from SummEval + DSTC11)

**The single most important architectural call in this plan.**

`scale_anchoring` is the only bias test that requires items of known varying quality.
The test math uses only the extremes:

```
spread = mean(high_scores) − mean(low_scores)
score  = min(1.0, spread / (scale_range * 0.7))
```

So we need (a) items at the high end, (b) items at the low end, and (c) **trustworthy
ordering** between them. Only SummEval and DSTC11 give us this:

- **SummEval** — 1,600 (article, summary) triples, each rated 1–5 on 4 dimensions
  (relevance, coherence, fluency, consistency) by 3 human annotators.
  Quality score = mean of the 4 dimensions.
- **DSTC11 Track 4** — ~500 (context, response) items with OVERALL ratings from
  5 human annotators on a 0–4 scale.

Both have **real, multi-rater human ratings on numeric scales** — the only sources where
"high quality" and "low quality" labels are empirically grounded. JudgeBench (winner/loser
labels), BeaverTails (length proxy on `is_safe`), and MT-Bench (refs assumed excellent)
were excluded because their tier labels would be heuristic at best.

### Bucket construction

1. Within each source, sort by score and split into quartiles.
2. Top quartile → tier = `"high"`; bottom quartile → tier = `"low"`.
3. Stride-sample 50 per source per bucket → **100 high + 100 low = 200 items total.**
4. Combine: 50 SummEval high + 50 DSTC11 high (heterogeneous in domain but homogeneous
   in quality signal).

### Schema

```json
{
  "id": "scale_tier_001",
  "bias_type": "scale_anchoring",
  "tier": "high",
  "domain": "summarization",
  "question": "Summarize the following article:\n\n...",
  "response": "...",
  "human_rating_mean": 4.7,
  "n_annotators": 3,
  "source_dataset": "SummEval",
  "license": "MIT"
}
```

**Note:** `scale_tiers.jsonl` items have `domain` ∈ {summarization, dialogue}.
The test runs cross-domain — a qa-shaped judge template applied to a summarization
item will score the summary against whatever the user's eval template asks. This is
documented as a known limitation in `docs/methodology.md`.

---

## 12. Step 4b — Self-consistency

Sample 40 items from `scale_tiers.jsonl` deterministically (stride sampling). The test
calls the judge 5× on the same item and measures score variance — the items themselves
need no special property beyond being valid prompts.

**Schema:**
```json
{
  "id": "self_consistency_001",
  "bias_type": "self_consistency",
  "domain": "summarization",
  "question": "...",
  "response": "...",
  "source_scale_tier_id": "scale_tier_007",
  "source_dataset": "SummEval",
  "license": "MIT"
}
```

---

## 13. Step 4c — Yes/no probes

100 hand-authored true/false statements, balanced 50T/50F. Lives in
`scripts/handcurated/yesno_probes.json` and is copied into
`universal/yesno_probes.jsonl` at build time with `source_dataset` and `license`
fields injected.

**Schema:**
```json
{
  "id": "yesno_001",
  "bias_type": "yes_bias",
  "statement": "Water boils at 100 degrees Celsius at sea level.",
  "ground_truth": true,
  "category": "common-knowledge",
  "human_verified": true,
  "source_dataset": "judicator-curated-v0.1",
  "license": "Apache-2.0"
}
```

---

## 14. Step 5 — Verifier

Parallel verification, one process per JSONL file via `ProcessPoolExecutor`.
Per-file checks:

| File kind | Check |
|---|---|
| All | required fields present (`id`, `license`, `source_dataset`); `license` non-empty |
| seeds | `question` and `response` non-empty; **no `quality_tier` field** (would be a regression) |
| verbosity | `content_preserved=True`; `len(manipulated_response) > len(original_response)` |
| authority | `manipulation == "authority_injection"` |
| position | `winner_response` and `loser_response` non-empty; not identical |
| concreteness | `vague_response` and `concrete_response` non-empty |
| scale_tiers | `tier ∈ {"high","low"}` only; ≥ 10 items per bucket; no other tier values |
| self_consistency | exactly 40 items; every item has `source_scale_tier_id` |
| yesno | T/F count delta ≤ 5 |

A failure in any file fails the whole run with a non-zero exit code.

---

## 15. Step 6 — ATTRIBUTION.md

`scripts/generate_attribution.py` scans every fixture JSONL, aggregates by
`source_dataset`, and writes:

```markdown
# Judicator — Data Attribution

## JudgeBench
- Source: github.com/ScalerLab/JudgeBench
- Paper: Tan et al. 2024, arxiv 2410.12784
- License: MIT
- Items used: 605
... (per source, alphabetically)
```

Commit `ATTRIBUTION.md` to repo root. Never ship without it.

---

## 16. Fixture Counts (v0.1 actuals)

| File | Count |
|---|---|
| qa/seeds.jsonl | 100 |
| qa/verbosity.jsonl | 100 |
| qa/authority.jsonl | 100 |
| qa/position.jsonl | 100 |
| code/seeds.jsonl | 83 |
| code/verbosity.jsonl | 83 |
| code/authority.jsonl | 83 |
| code/position.jsonl | 72 |
| summarization/seeds.jsonl | 100 |
| summarization/verbosity.jsonl | 100 |
| summarization/authority.jsonl | 100 |
| safety/seeds.jsonl | 100 |
| safety/verbosity.jsonl | 100 |
| safety/authority.jsonl | 100 |
| dialogue/seeds.jsonl | 50 |
| dialogue/verbosity.jsonl | 50 |
| dialogue/authority.jsonl | 50 |
| universal/scale_tiers.jsonl | 200 (100 high + 100 low) |
| universal/self_consistency.jsonl | 40 |
| universal/concreteness.jsonl | 14 |
| universal/yesno_probes.jsonl | 100 |
| **Total** | **~1,825 items, 21 files** |

Estimated judge calls per audit (pointwise judge on a single domain):

| Test | Items | Calls per item | Total |
|---|---|---|---|
| verbosity | 50–100 | 2 | 100–200 |
| self_consistency | 40 | 5 | 200 |
| scale_anchoring | 200 | 1 | 200 |
| authority | 50–100 | 2 | 100–200 |
| concreteness | 14 | 2 | 28 |
| yes_bias | (binary judge only) | — | — |
| position | (pairwise judge only) | — | — |

So a pointwise audit on a typical domain runs ~600–800 judge calls.

---

## 17. What Gets Committed vs. Gitignored

```
COMMITTED:
  src/judicator/fixtures/data/**/*.jsonl
  ATTRIBUTION.md
  scripts/                      # the pipeline itself
  scripts/handcurated/yesno_probes.json

GITIGNORED:
  data_sources/                 # raw cloned repos
  scripts/__pycache__/
  scripts/handcurated/*.local.json
  .env
```

---

## 18. Re-running After Updates

When source datasets update or you add new domains in v0.2:

1. `bash scripts/download_sources.sh` (pulls latest)
2. `python scripts/generate_fixtures.py --workers $(nproc) --all`
3. `python scripts/verify_fixtures.py`
4. `python scripts/generate_attribution.py`
5. Bump `FIXTURE_VERSION` in `src/judicator/fixtures/__init__.py`
6. Commit updated JSONL + ATTRIBUTION.md

Re-runs are essentially free (under 2 minutes wall-clock on 8-core laptop, modulo
HF cache hits for SummEval and DSTC11).

---

## 19. Timeline

The pipeline runs in **under 2 minutes** wall-clock on an 8-core laptop. The schedule
is dominated by human-in-the-loop tasks (license verification, hand-authoring yes/no
probes, spot-checks), not compute.

```
Day 1   Setup, parallel-download sources, verify licenses
        Implement scripts/ (manipulators, generator, verifier, attribution)
Day 2   Run Step 1 + 2 (parallel seed + pair extraction across 5 domains)
        Hand-author yes/no probes (100 statements, 50T/50F — ~2 hours)
        Spot-check 20% of every domain's seeds
Day 3   Run Step 3 (parallel manipulators)
        Run Steps 4a-c, 5, 6
        Commit
```

3 days total. Must complete before plan-build.md Phase 2 begins.

---

*End of data curation planner.*
*Output of this planner is the input to plan-build.md Phase 2.*
