"""Pointwise judge on the dialogue domain with cost estimate preview."""
import anthropic
from judicator import Judge, JudgeAuditor

client = anthropic.Anthropic()

system_prompt = (
    "You are evaluating the quality of conversational AI responses. "
    "Be precise and consistent."
)
eval_template = (
    "Conversation turn:\n{question}\n\n"
    "Response:\n{response}\n\n"
    "Rate the response on a scale of 1-10 for helpfulness and coherence."
)


def claude_judge(prompt: str) -> str:
    return (
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        .content[0]
        .text
    )


judge = Judge(
    llm_fn=claude_judge,
    system_prompt=system_prompt,
    eval_template=eval_template,
    judge_name="dialogue_judge_v1",
    model_name="claude-haiku-4-5",
)

auditor = JudgeAuditor(
    judge=judge,
    domain="dialogue",
    cost_per_call=0.0003,
    confirm=True,
)

# Preview cost before running
est = auditor.estimate()
est.display()

# Run full audit
report = auditor.audit()
print(report.summary())

# Inspect worst bias
worst = report.ranked()[0]
print(f"\nWorst bias: {worst.test_name} (score={worst.score:.3f})")
print(f"Details: {worst.details}")
