# Bias Tests

Judicator runs 7 bias tests. Each test is independent — there is no composite score.

---

## Position bias
**Applies to:** pairwise judges

A pairwise judge is position-biased when it consistently picks whichever response
appears in slot A (or slot B) regardless of content. Measured by swapping the
winner/loser into both orderings and checking positional consistency.

**Metric:** inconsistency rate — fraction of items where the judge flipped its
verdict when the same content moved between slots.

---

## Verbosity bias
**Applies to:** pointwise, pairwise, binary judges

A judge that inflates scores for longer responses regardless of quality is
verbosity-biased. Measured using deterministic padding: each response is extended
with filler sentences while preserving meaning. A calibrated judge should score
the original and padded versions identically.

**Metric:** mean score inflation (padded − original).

---

## Self-consistency
**Applies to:** pointwise, binary judges (pairwise deferred to v0.2)

A judge that gives different scores to the same input on repeated calls is
unreliable. Measured by calling the judge 5× on each of 40 items and computing
per-item score variance.

**Metric:** mean score variance across 5 runs (0 = perfectly consistent).

---

## Scale anchoring
**Applies to:** pointwise judges

A pointwise judge that compresses all scores into a narrow band (e.g., 6–8 out of
10) fails to distinguish quality levels. Measured using human-verified high-quality
and low-quality pairs from SummEval and DSTC11.

**Metric:** score spread between high-tier and low-tier means vs. expected spread
(70% of scale range).

---

## Authority bias
**Applies to:** pointwise, pairwise, binary judges

A judge that inflates scores for responses that include fake credentials
("As a PhD in Computer Science…") is authority-biased. Measured using
the same deterministic-prefix manipulation as verbosity.

**Metric:** mean score inflation (authority-prefixed − original).

---

## Concreteness bias
**Applies to:** pointwise, pairwise judges

A judge that prefers responses loaded with fabricated statistics and named entities
over accurate but appropriately vague responses is concreteness-biased. Measured
using natural pairs from OffsetBias.

**Metric:** fraction of items where judge correctly preferred the vague (accurate)
response over the concrete (specifics-inflated) one.

**Note:** n=14 fixture; signal is coarse.

---

## Yes-bias
**Applies to:** binary judges only

A binary judge that over-approves — saying "Yes" to false or harmful statements —
is yes-biased. Measured on a balanced set of 100 T/F statements (50 true, 50 false).

**Metric:** balanced accuracy on T/F classification.
A judge that always says "Yes" or always says "No" scores ~0.5.
