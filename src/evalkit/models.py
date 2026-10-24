# Data shapes shared by providers, checks and storage.

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# What every provider returns, whatever model sits behind it.
class Response(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class CheckResult(BaseModel):
    check: str
    passed: bool
    detail: str = ""
