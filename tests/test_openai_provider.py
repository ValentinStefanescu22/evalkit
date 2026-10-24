import json
from typing import Any

import httpx
import pytest

from evalkit.models import Message, ToolDef
from evalkit.providers import OpenAICompatibleProvider, ProviderError

# A typical successful reply in the OpenAI chat-completions format.
REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "Bucharest."}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


# Builds a provider whose HTTP calls go to `handler` instead of the network.
# `sent` collects every request body so tests can inspect what was sent.
def fake(
    status: int = 200, body: Any = REPLY, sent: list[dict[str, Any]] | None = None
) -> OpenAICompatibleProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if sent is not None:
            sent.append(json.loads(request.content))
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider("qwen2.5:3b", client=client)


def ask(provider: OpenAICompatibleProvider, **kwargs: Any) -> Any:
    return provider.complete([Message(role="user", content="Capital of Romania?")], **kwargs)


def test_request_body_shape() -> None:
    sent: list[dict[str, Any]] = []
    tool = ToolDef(name="get_weather", description="Weather for a city.")
    ask(fake(sent=sent), tools=[tool], params={"temperature": 0})
    assert sent == [
        {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "Capital of Romania?"}],
            "temperature": 0,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather for a city.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    ]


def test_no_tools_key_when_no_tools() -> None:
    sent: list[dict[str, Any]] = []
    ask(fake(sent=sent))
    assert "tools" not in sent[0]


def test_parses_text_and_usage() -> None:
    response = ask(fake())
    assert response.text == "Bucharest."
    assert (response.input_tokens, response.output_tokens) == (12, 3)
    assert response.latency_ms >= 0


def test_null_content_and_missing_usage() -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    response = ask(fake(body=body))
    assert response.text == ""
    assert response.input_tokens == 0


def tool_reply(arguments: Any) -> dict[str, Any]:
    call = {"type": "function", "function": {"name": "get_weather", "arguments": arguments}}
    return {"choices": [{"message": {"content": None, "tool_calls": [call]}}]}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"city": "Bucharest"}', {"city": "Bucharest"}),
        ({"city": "Bucharest"}, {"city": "Bucharest"}),
        ("", {}),
        ('{"city": "Buch', {"_raw": '{"city": "Buch'}),
        ('["Bucharest"]', {"_raw": '["Bucharest"]'}),
    ],
)
def test_parses_tool_call_arguments(arguments: Any, expected: dict[str, Any]) -> None:
    call = ask(fake(body=tool_reply(arguments))).tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == expected


def test_cannot_connect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="is Ollama running"):
        ask(OpenAICompatibleProvider("qwen2.5:3b", client=client))


def test_model_not_found() -> None:
    with pytest.raises(ProviderError, match="try: ollama pull qwen2.5:3b"):
        ask(fake(status=404, body={"error": "model not found"}))


def test_other_http_error_shows_server_message() -> None:
    with pytest.raises(ProviderError, match="HTTP 500.*out of memory"):
        ask(fake(status=500, body={"error": "out of memory"}))


def test_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert OpenAICompatibleProvider("m").client.headers["Authorization"] == "Bearer test-key"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "Authorization" not in OpenAICompatibleProvider("m").client.headers
