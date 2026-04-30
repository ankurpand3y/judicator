# Interpreting Results

## Score

Each bias test returns a score in **[0, 1]**, where **1.0 = fully calibrated**
(no detectable bias) and **0.0 = maximally biased**.

## Verdict

| Verdict | Score threshold |
|---|---|
| PASS | score ≥ 0.70 |
| FAIL | score < 0.70 |
| N/A | test does not apply to this judge type |

The 0.70 threshold is a calibration heuristic, not an empirically derived cutoff.
Use it as a flag for investigation, not a definitive judgment.

## Severity

| Severity | Score range | Verdict |
|---|---|---|
| CRITICAL | < 0.50 | always FAIL |
| SIGNIFICANT | 0.50–0.65 | always FAIL |
| MINOR | 0.65–0.80 | FAIL if < 0.70, PASS if ≥ 0.70 |
| NONE | ≥ 0.80 | always PASS |

The MINOR band straddles the PASS/FAIL threshold intentionally.
A "PASS but MINOR" result means the judge is technically passing but trending
toward bias — monitor for drift.

## Rank

Tests are ranked **worst-first** (rank 1 = highest bias). Use rank to prioritize
which bias to investigate first.

## No composite score

Judicator deliberately does not produce a single overall score. Composite metrics
obscure which specific bias is the problem. Address the worst-ranked test first.

## Interpreting specific tests

### Scale anchoring FAIL (CRITICAL)
The judge is compressing all scores into a narrow band. High-quality and
low-quality responses receive similar scores. This is the most common failure
mode for off-the-shelf judge prompts.

**Action:** Add explicit rubric anchors to your `eval_template` (e.g.,
"A score of 1 means completely irrelevant; a score of 10 means expert-level").

### Position bias FAIL
The judge's verdicts are influenced by which slot (A or B) a response appears in.
For a biased judge, swapping the same two responses produces different verdicts.

**Action:** In production, run each pair in both orderings and average the result.

### Verbosity / Authority bias FAIL
The judge inflates scores for longer responses or responses with fake credentials.

**Action:** Add an explicit instruction to your system prompt:
"Evaluate the *content* of the response, not its length or claimed credentials."

### Yes-bias FAIL
The binary judge over-approves. False statements are classified as true.

**Action:** Add negative examples to your system prompt or few-shot section.
Consider calibrating the threshold (e.g., require "Yes, definitely" rather than "Yes").
