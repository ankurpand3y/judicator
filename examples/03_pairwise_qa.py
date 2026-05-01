"""Pairwise judge on QA domain — tests position bias.

Note: position bias only runs on domains with position fixtures (qa and code).
Summarization/safety/dialogue position pairs are on the v0.3+ roadmap.
"""
import openai
from judicator import Judge, JudgeAuditor

system_prompt = "You are a careful evaluator comparing two responses."
eval_template = (
    "Question: {question}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Which response better answers the question? Reply with only 'A' or 'B'."
)


def pairwise_judge(prompt: str) -> str:
    return (
        openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
        )
        .choices[0]
        .message.content
    )


judge = Judge(
    llm_fn=pairwise_judge,
    system_prompt=system_prompt,
    eval_template=eval_template,
    judge_name="pairwise_qa_judge",
)

report = JudgeAuditor(
    judge=judge,
    domain="qa",
    cost_per_call=0.005,
    max_workers=20,
).audit()

print(report.summary())

pos = report.tests["position"]
if not pos.not_applicable:
    print(f"\nPosition bias: inconsistency_rate={pos.details['inconsistency_rate']:.3f}")
    print(f"Slot-A pick rate: {pos.details['slot_a_pick_rate']:.3f}  (0.5 = no preference)")
