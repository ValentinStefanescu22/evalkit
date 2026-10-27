# Claude through Anthropic's Messages API, via the official SDK (optional extra:
# pip install 'evalkit[anthropic]'). The SDK is imported only when this provider is built,
# so the rest of evalkit works without it.

import time
from typing import TYPE_CHECKING, Any

from evalkit.models import Message, Response, ToolCall, ToolDef
from evalkit.providers import ProviderError

if TYPE_CHECKING:
    import anthropic
    from anthropic.types import ContentBlockParam, MessageParam, ToolParam

# USD per million tokens (input, output), from Anthropic's price list, October 2026.
# A model missing here can't be cost-estimated before a run, so it's refused.
# (Haiku 5.5 costs more above 100K prompt tokens; evalkit prompts are far below that.)
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Room for a full answer plus the model's thinking (current models always think).
DEFAULT_MAX_TOKENS = 16000


def _import_sdk() -> Any:
    try:
        import anthropic
    except ImportError as e:
        raise ProviderError(
            "the Anthropic provider needs the SDK: pip install 'evalkit[anthropic]'"
        ) from e
    return anthropic


# Our messages follow the OpenAI shape (milestone 4). Claude's API differs:
# - system text is a separate `system` field, not a message
# - an assistant's tool calls are `tool_use` blocks in its content
# - a tool's answer is a `tool_result` block inside a *user* message
def to_anthropic(messages: list[Message]) -> tuple[str, list["MessageParam"]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    chat: list[MessageParam] = []
    for m in messages:
        if m.role == "user":
            chat.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks: list[ContentBlockParam] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for call in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            chat.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            result: ContentBlockParam = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }
            chat.append({"role": "user", "content": [result]})
    return system, chat


def _tools(tools: list[ToolDef] | None) -> list["ToolParam"]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools or []
    ]


class AnthropicProvider:
    def __init__(self, model: str, client: "anthropic.Anthropic | None" = None) -> None:
        if model not in PRICES:
            known = ", ".join(PRICES)
            raise ProviderError(f"no price known for {model!r} (known: {known}); add it to PRICES")
        self.model = model
        self.prices = PRICES[model]
        # Credentials come from the environment (ANTHROPIC_API_KEY), never from a flag.
        self.client = client or _import_sdk().Anthropic()
        # Without any credential the SDK only fails at the first request, with a TypeError.
        # Say it here instead, with the fix, before anything is sent.
        found = (self.client.api_key, self.client.auth_token, self.client.credentials)
        if all(c is None for c in found):
            raise ProviderError("no Anthropic credentials found: export ANTHROPIC_API_KEY=...")
        self.base_url = str(self.client.base_url)
        # Running total of what this provider has spent, printed after a run.
        self.spent_usd = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = self.prices
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    # Only `max_tokens` is read from params. temperature/top_p/top_k are never sent:
    # current Claude models reject them (and so does the API since version 2026-10-21).
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        # Already importable: __init__ checked. A real import keeps mypy's type checks.
        import anthropic
        from anthropic.types import TextBlock, ToolUseBlock

        system, chat = to_anthropic(messages)
        max_tokens = int((params or {}).get("max_tokens", DEFAULT_MAX_TOKENS))
        start = time.perf_counter()
        try:
            reply = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=chat,
                system=system or anthropic.omit,
                tools=_tools(tools) or anthropic.omit,
            )
        except Exception as e:
            raise self._error(e) from e
        latency_ms = (time.perf_counter() - start) * 1000

        if reply.stop_reason == "refusal":
            # A safety refusal is an answer too: keep it visible and gradeable. No fallback
            # to another model, so every answer really comes from the model on the run.
            category = reply.stop_details.category if reply.stop_details else None
            text = f"(refused: {category or 'no category'})"
        else:
            text = "".join(b.text for b in reply.content if isinstance(b, TextBlock))
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in reply.content
            if isinstance(b, ToolUseBlock)
        ]
        cost = self.cost(reply.usage.input_tokens, reply.usage.output_tokens)
        self.spent_usd += cost
        return Response(
            text=text,
            tool_calls=calls,
            input_tokens=reply.usage.input_tokens,
            output_tokens=reply.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    # Exact input size of one request, from the API's free token-counting endpoint.
    def count_tokens(self, messages: list[Message], tools: list[ToolDef] | None = None) -> int:
        import anthropic

        system, chat = to_anthropic(messages)
        try:
            counted = self.client.messages.count_tokens(
                model=self.model,
                messages=chat,
                system=system or anthropic.omit,
                tools=_tools(tools) or anthropic.omit,
            )
        except Exception as e:
            raise self._error(e) from e
        return int(counted.input_tokens)

    # SDK errors become ProviderError with a fix in the message. Most specific first.
    def _error(self, e: Exception) -> Exception:
        import anthropic

        if isinstance(e, anthropic.AuthenticationError | anthropic.CredentialsError):
            return ProviderError("Anthropic rejected the credentials: set ANTHROPIC_API_KEY")
        if isinstance(e, anthropic.NotFoundError):
            return ProviderError(f"model {self.model!r} not found at {self.base_url}")
        if isinstance(e, anthropic.RateLimitError):
            return ProviderError("rate limited by Anthropic even after retries; try again later")
        if isinstance(e, anthropic.APIStatusError):
            return ProviderError(f"Anthropic API error {e.status_code}: {e.message}")
        if isinstance(e, anthropic.APIConnectionError):
            return ProviderError(f"cannot reach {self.base_url}; check the network")
        return e
