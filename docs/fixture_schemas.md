# Fixture Schemas

All fixture files are JSONL (one JSON object per line).

Required fields on every item: `id`, `license`, `source_dataset`.

---

## seeds (`{domain}/seeds.jsonl`)

Seed items used as the basis for manipulator-generated fixtures.
Not consumed directly by any bias test at audit time.

| Field | Type | Description |
|---|---|---|
| `id` | str | Unique item ID |
| `domain` | str | qa / summarization / code / safety / dialogue |
| `question` | str | Input prompt or question |
| `response` | str | Reference response |
| `source_dataset` | str | Original dataset name |
| `license` | str | License of the source |

---

## verbosity (`{domain}/verbosity.jsonl`)

| Field | Type | Description |
|---|---|---|
| `id` | str | |
| `domain` | str | |
| `question` | str | |
| `original_response` | str | Unmodified response |
| `manipulated_response` | str | Padded with filler sentences |
| `manipulation` | str | Manipulation type identifier |
| `content_preserved` | bool | True — content is byte-preserved |
| `source_seed_id` | str | ID of the seed item |
| `source_dataset` | str | |
| `license` | str | |

---

## authority (`{domain}/authority.jsonl`)

Same schema as verbosity. `manipulated_response` has a fake credential prefix
prepended (e.g., "As a certified expert with 15 years of experience, …").

---

## position (`{domain}/position.jsonl`)

Only exists for `qa` and `code` domains in v0.1.

| Field | Type | Description |
|---|---|---|
| `id` | str | |
| `domain` | str | |
| `question` | str | |
| `winner_response` | str | Human-preferred response |
| `loser_response` | str | Human-dispreferred response |
| `source_split` | str | Original dataset split |
| `source_dataset` | str | JudgeBench |
| `license` | str | MIT |

---

## universal/scale_tiers.jsonl

| Field | Type | Description |
|---|---|---|
| `id` | str | |
| `tier` | str | `"high"` or `"low"` |
| `domain` | str | |
| `question` | str | |
| `response` | str | |
| `human_rating_mean` | float | Mean human rating across annotators |
| `n_annotators` | int | Number of human raters |
| `source_dataset` | str | SummEval or DSTC11-Track4 |
| `license` | str | |

200 items: 100 `"high"` (top quartile) + 100 `"low"` (bottom quartile).

---

## universal/self_consistency.jsonl

| Field | Type | Description |
|---|---|---|
| `id` | str | |
| `domain` | str | |
| `question` | str | |
| `response` | str | |
| `source_scale_tier_id` | str | ID of originating scale_tiers item |
| `source_dataset` | str | |
| `license` | str | |

40 items sampled from scale_tiers via stride sampling.

---

## universal/concreteness.jsonl

| Field | Type | Description |
|---|---|---|
| `id` | str | |
| `question` | str | |
| `vague_response` | str | Correct / appropriately vague response |
| `concrete_response` | str | Response with fabricated specifics |
| `source_dataset` | str | OffsetBias |
| `license` | str | Apache 2.0 |

14 items from OffsetBias `concreteness` subcategory.

---

## universal/yesno_probes.jsonl

| Field | Type | Description |
|---|---|---|
| `id` | str | |
| `statement` | str | A factual statement |
| `ground_truth` | bool | `true` if statement is correct, `false` otherwise |
| `source` | str | `"common-knowledge"` |
| `human_verified` | bool | All items hand-verified |
| `source_dataset` | str | judicator-curated-v0.1 |
| `license` | str | Apache 2.0 |

100 items: 50 true statements + 50 false statements.
