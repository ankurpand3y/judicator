# Judicator

**Audit your LLM-as-a-Judge for bias and miscalibration.**

[![PyPI version](https://badge.fury.io/py/judicator.svg)](https://pypi.org/project/judicator/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Install

```bash
pip install judicator
```

---

## Quickstart

```python
import openai
from judicator import Judge, JudgeAuditor

system_prompt = "You are an expert evaluator. Score responses objectively."
eval_template = "Question: {question}\nResponse: {response}\nScore 1-10."

def my_judge_call(prompt: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content

judge = Judge(
    llm_fn=my_judge_call,
    system_prompt=system_prompt,
    eval_template=eval_template,
    judge_name="my_first_judge"
)

# Shows cost estimate and prompts Y/n. Pass confirm=False to skip.
# max_workers=20 runs API calls in parallel — typically 10–15× faster.
report = JudgeAuditor(
    judge=judge,
    domain="qa",
    cost_per_call=0.0003,
    max_workers=20,
).audit()
print(report.summary())
report.save_json("my_audit.json")
```

---

## Speed (`max_workers`)

A full audit makes ~1,000 LLM calls. Sequential runs take 20–25 minutes.
Set `max_workers` to run calls in parallel via a thread pool:

```python
JudgeAuditor(judge=judge, domain="qa", max_workers=20).audit()
```

| `max_workers` | Wall time (~1k calls) | Speedup |
|---|---|---|
| 1 (default) | 20–25 min | 1× |
| 10 | 2.5 min | 8× |
| **20** | **1.5 min** | **13×** |
| 50+ | diminishing returns; rate-limit risk | — |

**Caveats**

- **Rate limits.** Cost is unchanged but request rate is much higher. Lower `max_workers` if you see 429 errors — there is no auto-backoff.
- **Thread-safe `llm_fn` required.** Stateless calls are safe (OpenAI/Anthropic/OpenRouter clients are thread-safe). Don't share conversation state across calls.
- **Parallelism is per-test.** Within a single bias test, fixture items run concurrently; tests still execute one after another.

---

## What it tests

| Bias | What it catches | Applies to |
|---|---|---|
| **position** | Judge picks slot A/B regardless of content | pairwise |
| **verbosity** | Judge inflates scores for longer responses | all types |
| **self_consistency** | Judge gives different scores to the same input | pointwise, binary |
| **scale_anchoring** | Judge compresses all scores into a narrow band | pointwise |
| **authority** | Judge inflates scores for fake credentials | all types |
| **concreteness** | Judge prefers fabricated specifics over accurate vague answers | pointwise, pairwise |
| **yes_bias** | Binary judge over-approves false statements | binary |

---

## Supported judge types

| Type | Template shape | Detected by |
|---|---|---|
| **pointwise** | `{question}` + `{response}` → numeric score | `{response}` placeholder |
| **pairwise** | `{question}` + `{response_a}` + `{response_b}` → A or B | `{response_a}` and `{response_b}` |
| **binary** | `{statement}` → Yes or No | yes/no keyword in template |

Judge type is auto-detected from your `eval_template`. Override with
`judge_type="pointwise"` if detection fails.

---

## Works with any LLM

Judicator never touches your API keys or model configuration.
You wrap your LLM call in a function — Judicator calls that function.

**OpenAI**
```python
import openai

def my_fn(prompt: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content
```

**Anthropic**
```python
import anthropic
client = anthropic.Anthropic()

def my_fn(prompt: str) -> str:
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text
```

**Ollama (local)**
```python
import ollama

def my_fn(prompt: str) -> str:
    return ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )["message"]["content"]
```

Pass any of these as `llm_fn` to `Judge`. Judicator works identically with all three.

---

## Understanding the report

```
╔══════════════════════════════════════════════════════════════╗
║  JUDICATOR — AUDIT REPORT                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Judge:   my_qa_judge   Domain:  qa   Type:  pointwise       ║
╠════════════════════╦═══════╦═══════╦══════════╦══════════════╣
║   BIAS TEST        ║  SCORE║  RANK ║  VERDICT ║  SEVERITY    ║
╠════════════════════╬═══════╬═══════╬══════════╬══════════════╣
║   scale_anchoring  ║  0.312║  1/5  ║  FAIL    ║  CRITICAL    ║
║   verbosity        ║  0.620║  2/5  ║  FAIL    ║  SIGNIFICANT ║
║   concreteness     ║  0.714║  3/5  ║  PASS    ║  MINOR       ║
║   authority        ║  0.810║  4/5  ║  PASS    ║  NONE        ║
║   self_consistency ║  0.950║  5/5  ║  PASS    ║  NONE        ║
╚══════════════════════════════════════════════════════════════╝
```

**Score:** 0–1. Higher = more calibrated. No composite score — each test is independent.

**Rank:** 1 = worst bias. Address rank 1 first.

**Severity bands:**
- `CRITICAL` (< 0.50): strong bias, investigate immediately
- `SIGNIFICANT` (0.50–0.65): meaningful bias, likely affects production quality
- `MINOR` (0.65–0.80): borderline — PASS if ≥ 0.70, FAIL otherwise
- `NONE` (≥ 0.80): no detectable bias

**N/A** results mean the test does not apply to your judge type or domain,
not that the judge passed the test.

---

## What v0.1 does NOT cover

- Composite scoring / single overall grade
- Sycophancy, compassion fade, bandwagon, sentiment, fallacy oversight, and
  other biases beyond the 7 tested
- Listwise, reference-based, CoT, or multi-turn judge types
- Translation, medical, legal, financial, or creative writing domains
- Position pairs for summarization, safety, or dialogue (qa + code only)
- Custom bias tests or BYO-data mode
- Token-aware cost estimation (currently flat-per-call)
- GitHub Actions integration or SaaS dashboard

These are v0.2+ scope. See [Out of Scope in plan-build.md](plan-build.md).

---

## Citation

If you use Judicator in your research, please cite:

```bibtex
@software{judicator2026,
  author = {Pandey, Ankur},
  title  = {Judicator: An LLM-as-a-Judge Bias Auditing Library},
  year   = {2026},
  url    = {https://github.com/ankurpand3y/judicator},
  version = {0.2.0}
}
```

---

## Built on

Judicator ships with fixtures derived from the following datasets.
All are used in accordance with their licenses.

| Dataset | Paper | License |
|---|---|---|
| [OffsetBias](https://github.com/ncsoft/offsetbias) | Park et al. 2024 | Apache 2.0 |
| [JudgeBench](https://github.com/ScalerLab/JudgeBench) | Tan et al. 2024 | MIT |
| [MT-Bench](https://github.com/lm-sys/FastChat) | Zheng et al. 2023 | Apache 2.0 |
| [BeaverTails](https://github.com/PKU-Alignment/beavertails) | Ji et al. 2023 | CC-BY-NC-4.0 |
| [SummEval](https://huggingface.co/datasets/mteb/summeval) | Fabbri et al. 2021 | MIT |
| [DSTC11-Track4](https://huggingface.co/datasets/mario-rc/dstc11.t4) | Rodriguez-Cantelar et al. 2023 | Apache 2.0 |

See [ATTRIBUTION.md](ATTRIBUTION.md) for full item counts.
