import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from evalkit.anthropic_provider import AnthropicProvider
from evalkit.models import Message, ToolCall, ToolDef
from evalkit.providers import ProviderError


def reply(content: list[dict[str, Any]], stop_reason: str = "end_turn", **extra: Any) -> Any:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": None,
        "usage": {"input_tokens": 1000, "output_tokens": 500},
        **extra,
    }


TEXT = reply([{"type": "text", "text": "Bucharest."}])


# A provider whose HTTP calls go to a fake transport: the real SDK parses real-shaped
# replies, with no network and no key. `sent` collects (path, body) of every request.
def fake(
    status: int = 200, body: Any = TEXT, sent: list[tuple[str, Any]] | None = None
) -> AnthropicProvider:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if sent is not None:
            sent.append((request.url.path, json.loads(request.content)))
        return httpx2.Response(status, json=body)

    client = anthropic.Anthropic(
        api_key="test-key",
        base_url="https://api.anthropic.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return AnthropicProvider("claude-opus-5-5", client=client)


# A tool conversation in our (OpenAI-shaped) format, as runner.build_messages makes it.
CONVERSATION = [
    Message(role="system", content="Be brief."),
    Message(role="user", content="Search for Australia's capital."),
    Message(
        role="assistant", tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "x"})]
    ),
    Message(role="tool", content="Canberra.", tool_call_id="call_1"),
]


def test_request_translates_to_claude_format_without_temperature() -> None:
    sent: list[tuple[str, Any]] = []
    tool = ToolDef(name="search", description="Web search.")
    fake(sent=sent).complete(CONVERSATION, tools=[tool], params={"temperature": 0})
    [(path, body)] = sent
    assert path == "/v1/messages"
    assert body == {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": "Be brief.",
        "messages": [
            {"role": "user", "content": "Search for Australia's capital."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "Canberra."}
                ],
            },
        ],
        "tools": [
            {"name": "search", "description": "Web search.", "input_schema": {"type": "object"}}
        ],
    }
    # Never sent: current Claude models reject sampling parameters.
    assert "temperature" not in body


def test_plain_question_sends_no_system_or_tools() -> None:
    sent: list[tuple[str, Any]] = []
    fake(sent=sent).complete([Message(role="user", content="Hi")], params={"max_tokens": 50})
    assert sent[0][1] == {
        "model": "claude-opus-5-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
    }


def test_reply_parsing_ignores_thinking_and_prices_tokens() -> None:
    body = reply(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Let me check. "},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "capital"}},
            {"type": "text", "text": "Done."},
        ],
        stop_reason="tool_use",
    )
    provider = fake(body=body)
    response = provider.complete([Message(role="user", content="Hi")])
    assert response.text == "Let me check. Done."
    assert response.tool_calls == [
        ToolCall(id="toolu_1", name="search", arguments={"q": "capital"})
    ]
    assert (response.input_tokens, response.output_tokens) == (1000, 500)
    # 1000 in x $4/M + 500 out x $20/M = $0.004 + $0.010
    assert response.cost_usd == pytest.approx(0.014)
    assert provider.spent_usd == pytest.approx(0.014)


def test_refusal_is_a_visible_answer() -> None:
    body = reply(
        [],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "x"},
    )
    response = fake(body=body).complete([Message(role="user", content="Hi")])
    assert response.text == "(refused: cyber)"


def test_count_tokens() -> None:
    sent: list[tuple[str, Any]] = []
    provider = fake(body={"input_tokens": 42}, sent=sent)
    assert provider.count_tokens(CONVERSATION) == 42
    assert sent[0][0] == "/v1/messages/count_tokens"
    assert "max_tokens" not in sent[0][1]


def error(kind: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, error("authentication_error", "invalid x-api-key"), "set ANTHROPIC_API_KEY"),
        (404, error("not_found_error", "model: x"), "model 'claude-opus-5-5' not found"),
        (429, error("rate_limit_error", "slow down"), "rate limited"),
        (400, error("invalid_request_error", "bad field"), "Anthropic API error 400: "),
    ],
)
def test_errors_become_provider_errors(status: int, body: Any, expected: str) -> None:
    with pytest.raises(ProviderError, match=expected):
        fake(status=status, body=body).complete([Message(role="user", content="Hi")])


def test_connection_error() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    client = anthropic.Anthropic(
        api_key="k",
        base_url="https://api.anthropic.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(refuse)),
        max_retries=0,
    )
    with pytest.raises(ProviderError, match="cannot reach https://api.anthropic.test"):
        AnthropicProvider("claude-opus-5-5", client=client).complete(
            [Message(role="user", content="Hi")]
        )


def test_unknown_model_has_no_price() -> None:
    with pytest.raises(ProviderError, match="no price known for 'claude-nonexistent'"):
        AnthropicProvider("claude-nonexistent", client=anthropic.Anthropic(api_key="k"))


def test_missing_sdk_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend the optional extra isn't installed: importing `anthropic` fails.
    import builtins

    real_import = builtins.__import__

    def no_anthropic(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    with pytest.raises(ProviderError, match=r"pip install 'evalkit\[anthropic\]'"):
        AnthropicProvider("claude-opus-5-5")


def test_no_credentials_is_caught_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no saved `ant auth login` profile either
    with pytest.raises(ProviderError, match="no Anthropic credentials found"):
        AnthropicProvider("claude-opus-5-5")
