# Data shapes shared by providers, checks and storage.

from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Base for models built from user-written YAML: unknown fields (typos like `promt`) are errors.
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Raises a readable error if `schema` is not itself a valid JSON schema.
def ensure_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        raise ValueError(f"invalid JSON schema: {e.message}") from e
    return schema


# A tool the model may call. `parameters` is a JSON schema describing its arguments.
class ToolDef(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})

    @field_validator("parameters")
    @classmethod
    def parameters_is_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_schema(v)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set by the server; links a tool's result back to the call that asked for it.
    id: str = ""


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    # An assistant message that asks for tools carries the calls here.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # A tool message says which call it is the result of.
    tool_call_id: str | None = None


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
    # The check's settings, saved with the result so reports show exactly what was checked.
    config: dict[str, Any] = Field(default_factory=dict)


# Everything about one case's outcome: what was asked, what came back, how it was graded.
class CaseResult(BaseModel):
    case_id: str
    repeat_index: int = 0
    tags: list[str] = Field(default_factory=list)
    prompt: str
    response: Response
    checks: list[CheckResult]

    # A case passes only if every one of its checks passed.
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)
