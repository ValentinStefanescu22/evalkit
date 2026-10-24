# Model providers. Anything with a matching `complete` method is a Provider.

import json
import os
import time
from typing import Any, Protocol

import httpx

from evalkit.models import Message, Response, ToolCall, ToolDef


class Provider(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response: ...


# Raised when the model can't be reached or refuses the request. Stops the run.
class ProviderError(Exception):
    pass


# Returns scripted answers keyed by the last user message.
# Keying by input (not by call order) keeps results identical no matter how
# many times or in which order cases run.
class MockProvider:
    def __init__(self, script: dict[str, str | Response], default: str = "") -> None:
        self.script = script
        self.default = default

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "")
        answer = self.script.get(prompt, self.default)
        return answer if isinstance(answer, Response) else Response(text=answer)


OLLAMA_URL = "http://localhost:11434/v1"


# Talks to any server that speaks the OpenAI chat-completions format:
# Ollama (default), vLLM, LM Studio, Groq, OpenRouter, ...
class OpenAICompatibleProvider:
    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        headers = {}
        # Ollama needs no key; hosted services do. Only ever read from the environment.
        if api_key := os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {api_key}"
        # Local models can take a while on the first call, while they load into memory.
        self.client = client or httpx.Client(timeout=120, headers=headers)

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
            **(params or {}),
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]

        start = time.perf_counter()
        try:
            reply = self.client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.ConnectError as e:
            raise ProviderError(
                f"cannot reach {self.base_url} — is Ollama running? (ollama serve)"
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.base_url} did not answer in time") from e
        latency_ms = (time.perf_counter() - start) * 1000

        if reply.status_code == 404:
            raise ProviderError(f"model {self.model!r} not found — try: ollama pull {self.model}")
        if reply.is_error:
            raise ProviderError(f"HTTP {reply.status_code} from {self.base_url}: {reply.text}")

        data = reply.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return Response(
            text=message.get("content") or "",
            tool_calls=[_parse_tool_call(c) for c in message.get("tool_calls") or []],
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )


# Plain messages stay {role, content}; tool-related fields are added only when present.
def _message_to_openai(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # The OpenAI format wants arguments as a JSON string, not an object.
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    return out


# OpenAI sends arguments as a JSON *string*. If the model wrote broken JSON, keep the
# raw text so the tool_call check fails visibly instead of the whole run crashing.
def _parse_tool_call(call: dict[str, Any]) -> ToolCall:
    function = call["function"]
    raw = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        arguments = {"_raw": raw}
    if not isinstance(arguments, dict):
        arguments = {"_raw": raw}
    return ToolCall(name=function["name"], arguments=arguments, id=call.get("id") or "")
