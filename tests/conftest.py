import pytest
from judicator.judge import Judge

POINTWISE_TEMPLATE = "Question: {question}\nResponse: {response}\nScore 1-10."
PAIRWISE_TEMPLATE = (
    "Question: {question}\nResponse A: {response_a}\nResponse B: {response_b}\n"
    "Which is better? Answer A or B."
)
BINARY_TEMPLATE = "Statement: {statement}\nAnswer yes or no."


def make_judge(
    template: str,
    response: str = "7",
    judge_type: str | None = None,
    name: str = "test_judge",
) -> Judge:
    return Judge(
        llm_fn=lambda _: response,
        system_prompt="You are an evaluator.",
        eval_template=template,
        judge_name=name,
        judge_type=judge_type,
    )


@pytest.fixture
def pointwise_judge() -> Judge:
    return make_judge(POINTWISE_TEMPLATE)


@pytest.fixture
def pairwise_judge() -> Judge:
    return make_judge(PAIRWISE_TEMPLATE, response="A")


@pytest.fixture
def binary_judge() -> Judge:
    return make_judge(BINARY_TEMPLATE, response="Yes")
