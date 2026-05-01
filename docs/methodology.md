# Methodology

## Design principles

1. **Deterministic manipulators.** The verbosity and authority manipulators are
   pure-Python functions with no LLM calls. Given the same input, they produce
   byte-identical output. This means fixture generation is fully reproducible.

2. **Pre-shipped fixtures.** Judicator does not call a helper LLM to construct
   test cases at audit time. All test data is bundled with the package.

3. **No composite metric.** Each bias test is independent. There is no weighted
   average or JCI score. This forces users to engage with individual failure modes
   rather than a single number that hides the cause.

4. **Honest scope.** Tests only run when fixtures exist for the domain and judge
   type. Missing coverage is reported as N/A with a clear reason, not silently
   skipped or substituted.

---

## Signal strength per bias test

Some tests rest on stronger empirical foundations than others.
Judicator reports these honestly so you can calibrate confidence in results.

| Test | Signal strength | Why |
|---|---|---|
| verbosity | Strong | Deterministic manipulator; padded text is byte-equivalent to original + filler |
| authority | Strong | Deterministic manipulator; prefix is the only addition |
| yes_bias | Strong | Hand-curated, balanced 50/50 T/F set; ground truth verified |
| position | Medium | Winner/loser labels from JudgeBench (qa, code only in v0.1) |
| self_consistency | Medium | Variance across 5 calls — temperature-dependent |
| concreteness | Medium-low | Natural pairs from OffsetBias 'concreteness' subcategory; n=14 |
| scale_anchoring | Coarse | Cross-source human ratings (SummEval, DSTC11). Detects scale collapse, not fine-grained miscalibration. |

---

## Fixture sources

| Source | Used for | License |
|---|---|---|
| OffsetBias (Park et al. 2024) | verbosity, authority, concreteness seeds | Apache 2.0 |
| JudgeBench (Tan et al. 2024) | position pairs (qa, code) | MIT |
| MT-Bench (Zheng et al. 2023) | qa/code/safety seeds | Apache 2.0 |
| BeaverTails (Ji et al. 2023) | safety seeds | CC-BY-NC-4.0 |
| SummEval (Fabbri et al. 2021) | scale_tiers (high/low) | MIT |
| DSTC11-Track4 (Rodriguez-Cantelar et al. 2023) | scale_tiers (high/low) | Apache 2.0 |
| judicator-curated-v0.1 | yesno_probes (100 T/F facts) | Apache 2.0 |

---

## Scale anchoring math

The scale anchoring test measures how well a pointwise judge uses the full scoring
range. Given human-verified high-quality and low-quality items:

```
spread = mean(high_scores) - mean(low_scores)
expected_spread = scale_range × 0.70 = 9.0 × 0.70 = 6.3
score = clip(spread / expected_spread, 0, 1)
```

A judge that assigns 9.0 to high items and 2.0 to low items achieves
spread = 7.0, score = min(1.0, 7.0/6.3) = 1.0.

A judge that assigns 7.0 to all items achieves spread = 0, score = 0.0.

**Limitation:** assumes a 1-10 scale. If your judge uses a different scale
(e.g., 0-5), results will be misleading. Scale auto-detection is on the roadmap.

---

## Self-consistency variance normalization

```
score = 1 - clip(mean_variance / max_possible_variance, 0, 1)
max_possible_variance = (scale_range / 2)² = (9/2)² = 20.25
```

A perfectly deterministic judge has variance = 0, score = 1.0.
A judge oscillating between 1 and 10 has variance ≈ 20.25, score ≈ 0.0.
