"""Quickstart — pointwise judge on QA domain."""
import openai
from judicator import Judge, JudgeAuditor

system_prompt = "You are an expert evaluator. Score responses objectively."
eval_template = "Question: {question}\nResponse: {response}\nScore 1-10."


def my_judge_call(prompt: str) -> str:
    return (
        openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        .choices[0]
        .message.content
    )


judge = Judge(
    llm_fn=my_judge_call,
    system_prompt=system_prompt,
    eval_template=eval_template,
    judge_name="my_first_judge",
)

# confirm=True (default) → press Y at the prompt to start.
# max_workers=20 runs API calls in parallel (~10–15× faster than serial).
report = JudgeAuditor(
    judge=judge,
    domain="qa",
    cost_per_call=0.0003,
    max_workers=20,
).audit()
print(report.summary())
report.save_json("my_audit.json")
report.save_html("my_audit.html")
