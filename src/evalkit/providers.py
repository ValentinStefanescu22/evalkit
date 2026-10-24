# Model providers. Anything with a matching `complete` method is a Provider.

from typing import Any, Protocol

from evalkit.models import Message, Response


class Provider(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response: ...


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
        tools: list[dict[str, Any]] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "")
        answer = self.script.get(prompt, self.default)
        return answer if isinstance(answer, Response) else Response(text=answer)
