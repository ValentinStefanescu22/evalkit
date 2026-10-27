# Cost estimates shown before a paid (Claude) run, so no money is spent without --yes.
# Input tokens are counted exactly by the API; output tokens can't be known in advance,
# so we show an expected figure from a guess and a worst case at max_tokens.

from dataclasses import dataclass

from evalkit.anthropic_provider import DEFAULT_MAX_TOKENS, AnthropicProvider
from evalkit.checks import RubricCheck
from evalkit.judge import judge_messages
from evalkit.models import Message
from evalkit.runner import build_messages, judge_question
from evalkit.suite import DocumentFormat, Suite

# ponytail: guesses, not measurements. Tune them from real runs
# (case_results.output_tokens) if the estimates turn out far off.
EXPECTED_OUTPUT_TOKENS = 1500  # per call: the visible answer plus the model's thinking
EXPECTED_ANSWER_TOKENS = 400  # an answer the judge will read, before that answer exists


@dataclass
class Estimate:
    provider: AnthropicProvider
    what: str  # e.g. "claude-opus-5-5 as judge"
    calls: int
    input_tokens: int
    max_tokens: int = DEFAULT_MAX_TOKENS

    def expected_usd(self) -> float:
        return self.provider.cost(self.input_tokens, self.calls * EXPECTED_OUTPUT_TOKENS)

    def worst_usd(self) -> float:
        return self.provider.cost(self.input_tokens, self.calls * self.max_tokens)

    def lines(self) -> list[str]:
        return [
            f"  {self.what}: {self.calls} calls, {self.input_tokens:,} input tokens (counted),"
            f" ~{self.calls * EXPECTED_OUTPUT_TOKENS:,} output tokens"
            f" (assumed {EXPECTED_OUTPUT_TOKENS:,}/call)",
            f"    expected ~${self.expected_usd():.2f}, worst case ${self.worst_usd():.2f}"
            f" (every call using max_tokens={self.max_tokens:,})",
        ]


# The model answering every case `repeats` times. Each distinct case is counted once.
def estimate_answers(
    provider: AnthropicProvider, suite: Suite, document_format: DocumentFormat, repeats: int
) -> Estimate:
    per_pass = sum(
        provider.count_tokens(build_messages(suite, case, document_format), case.tools or None)
        for case in suite.cases
    )
    return Estimate(
        provider,
        f"{provider.model} answering",
        calls=len(suite.cases) * repeats,
        input_tokens=per_pass * repeats,
        max_tokens=int(suite.defaults.params.get("max_tokens", DEFAULT_MAX_TOKENS)),
    )


# The judge grading `conversations` (one per rubric check). Before a run the answers don't
# exist yet, so `answer_tokens` per call is added for them.
def estimate_judging(
    judge: AnthropicProvider,
    conversations: list[list[Message]],
    repeats: int = 1,
    answer_tokens: int = 0,
) -> Estimate:
    per_pass = sum(judge.count_tokens(messages) + answer_tokens for messages in conversations)
    return Estimate(
        judge,
        f"{judge.model} as judge",
        calls=len(conversations) * repeats,
        input_tokens=per_pass * repeats,
    )


# Judge conversations for a suite before it runs: the answers are left empty.
def judge_conversations_before_run(suite: Suite) -> list[list[Message]]:
    return [
        judge_messages(check, judge_question(case), "")
        for case in suite.cases
        for check in case.checks
        if isinstance(check, RubricCheck)
    ]
