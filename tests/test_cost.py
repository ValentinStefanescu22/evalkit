import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from evalkit.anthropic_provider import AnthropicProvider
from evalkit.cost import (
    EXPECTED_ANSWER_TOKENS,
    estimate_answers,
    estimate_judging,
    judge_conversations_before_run,
)
from evalkit.suite import load_suite

SUITES = Path(__file__).parent.parent / "suites"


# Every token-counting call answers 100 tokens; `counted` records each request body.
def counting(counted: list[Any], model: str = "claude-opus-5-5") -> AnthropicProvider:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/messages/count_tokens"  # never a paid call
        counted.append(json.loads(request.content))
        return httpx2.Response(200, json={"input_tokens": 100})

    client = anthropic.Anthropic(
        api_key="k",
        base_url="https://api.anthropic.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return AnthropicProvider(model, client=client)


def test_estimate_answers_counts_each_case_once() -> None:
    counted: list[Any] = []
    suite = load_suite(SUITES / "basic_qa.yaml")  # 5 cases
    estimate = estimate_answers(counting(counted), suite, "plain", repeats=3)
    assert len(counted) == 5  # one count per distinct case, not per repeat
    assert (estimate.calls, estimate.input_tokens) == (15, 1500)
    # expected: 1500 in x $4/M + 15 x 1500 out x $20/M = 0.006 + 0.45
    assert estimate.expected_usd() == pytest.approx(0.456)
    # worst: 1500 in x $4/M + 15 x 16000 out x $20/M = 0.006 + 4.8
    assert estimate.worst_usd() == pytest.approx(4.806)
    assert estimate.lines() == [
        "  claude-opus-5-5 answering: 15 calls, 1,500 input tokens (counted),"
        " ~22,500 output tokens (assumed 1,500/call)",
        "    expected ~$0.46, worst case $4.81 (every call using max_tokens=16,000)",
    ]


def test_estimate_judging_before_a_run() -> None:
    counted: list[Any] = []
    suite = load_suite(SUITES / "open_answers.yaml")  # 8 rubric checks
    conversations = judge_conversations_before_run(suite)
    assert len(conversations) == 8
    estimate = estimate_judging(
        counting(counted, "claude-haiku-5-5"),
        conversations,
        repeats=5,
        answer_tokens=EXPECTED_ANSWER_TOKENS,
    )
    assert estimate.calls == 40
    assert estimate.input_tokens == (100 + EXPECTED_ANSWER_TOKENS) * 8 * 5
    assert estimate.what == "claude-haiku-5-5 as judge"
    # The judge's system prompt is part of what's counted.
    assert "strict, fair grader" in counted[0]["system"]
