# Judge Types

Judicator auto-detects judge type from `eval_template`. You can override with
`judge_type="pointwise"` etc.

---

## Pointwise
**Template signature:** contains `{response}`, `{answer}`, `{output}`, or `{completion}`

The judge receives a single response and assigns a numeric score (typically 1–10).

```python
eval_template = "Question: {question}\nResponse: {response}\nScore 1-10."
```

**Applicable bias tests:** verbosity, self_consistency, scale_anchoring, authority,
concreteness.

---

## Pairwise
**Template signature:** contains both `{response_a}` and `{response_b}`

The judge receives two responses and picks the better one (A or B).

```python
eval_template = (
    "Question: {question}\n"
    "Response A: {response_a}\n"
    "Response B: {response_b}\n"
    "Which is better? Answer A or B."
)
```

**Applicable bias tests:** position, verbosity, authority, concreteness.

---

## Binary
**Template signature:** contains a yes/no instruction keyword
(`"yes or no"`, `"true or false"`, `"answer only yes"`, etc.)

The judge classifies a single item as acceptable/unacceptable (Yes/No).

```python
eval_template = (
    "Statement: {statement}\n"
    "Is this factually accurate? Answer yes or no."
)
```

**Applicable bias tests:** verbosity, self_consistency, authority, yes_bias.

---

## Auto-detection confidence

| Condition | Type | Confidence |
|---|---|---|
| `{response_a}` and `{response_b}` present | pairwise | 0.95 |
| yes/no keyword in template | binary | 0.90 |
| `{response}`, `{answer}`, `{output}`, `{completion}` present | pointwise | 0.85 |
| None of the above | unknown | 0.0 |

If confidence < 0.80 and no explicit override, `DetectionError` is raised.
Pass `judge_type="pointwise"` (or `"pairwise"`, `"binary"`) to `Judge()` to override.
