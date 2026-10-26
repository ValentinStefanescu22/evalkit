# Test suites: the YAML format, its validation, and a loader with readable errors.

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from evalkit.checks import Check, RubricCheck, ToolCallCheck
from evalkit.models import Response, StrictModel, ToolCall, ToolDef

# How untrusted text (documents, tool output) is put into the prompt.
# plain: pasted as is. delimited: wrapped in tags + a system rule to treat it as data.
DocumentFormat = Literal["plain", "delimited"]


class Defaults(StrictModel):
    system: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    document_format: DocumentFormat = "plain"


# A fake earlier tool call and its result, for attacks hidden in tool output.
# The runner turns it into: assistant calls `name(arguments)`, tool answers `content`.
class ToolResult(StrictModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    content: str


class Case(StrictModel):
    # Ids become database keys and report links, so keep them simple.
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    tags: list[str] = Field(default_factory=list)
    prompt: str = Field(min_length=1)
    system: str | None = None
    document: str | None = None
    tools: list[ToolDef] = Field(default_factory=list)
    tool_result: ToolResult | None = None
    checks: list[Check] = Field(min_length=1)

    # Catches typos like `name: get_wether` when the case only offers `get_weather`.
    @model_validator(mode="after")
    def tool_checks_use_known_tools(self) -> "Case":
        offered = {tool.name for tool in self.tools}
        for check in self.checks:
            if isinstance(check, ToolCallCheck) and check.called and check.name not in offered:
                raise ValueError(f"tool_call check refers to unknown tool {check.name!r}")
        if self.tool_result and self.tool_result.name not in offered:
            raise ValueError(f"tool_result refers to unknown tool {self.tool_result.name!r}")
        return self


class Suite(StrictModel):
    name: str
    description: str = ""
    defaults: Defaults = Field(default_factory=Defaults)
    cases: list[Case] = Field(min_length=1)

    # True if some case needs an LLM judge (`run --judge`).
    @property
    def uses_judge(self) -> bool:
        return any(isinstance(c, RubricCheck) for case in self.cases for c in case.checks)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "Suite":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                duplicates.add(case.id)
            seen.add(case.id)
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(sorted(duplicates))}")
        return self


class SuiteError(Exception):
    pass


# The first gates shared by every YAML file we read: readable, valid YAML, a mapping.
def _read_yaml_mapping(path: Path, what: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SuiteError(f"{path}: cannot read file ({e.strerror})") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SuiteError(f"{path}: invalid YAML\n  {e}") from e

    if data is None:
        raise SuiteError(f"{path}: file is empty")
    if not isinstance(data, dict):
        raise SuiteError(f"{path}: top level must be a mapping with {what}")
    return data


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    data = _read_yaml_mapping(path, "`name` and `cases`")
    try:
        return Suite.model_validate(data)
    except ValidationError as e:
        lines = [_format_error(err, data) for err in e.errors()]
        raise SuiteError(f"{path}: {len(lines)} problem(s)\n" + "\n".join(lines)) from e


# Turns Pydantic's ('cases', 0, 'prompt') into "cases[0] (capital_romania).prompt".
def _format_error(err: Any, data: dict[str, Any]) -> str:
    loc = err["loc"]
    parts: list[str] = []
    for i, key in enumerate(loc):
        if isinstance(key, int):
            parts[-1] += f"[{key}]"
            if i == 1 and loc[0] == "cases":
                case = data["cases"][key]
                if isinstance(case, dict) and "id" in case:
                    parts[-1] += f" ({case['id']})"
        else:
            parts.append(str(key))
    where = ".".join(parts) or "suite"
    message = err["msg"].removeprefix("Value error, ")
    return f"  - {where}: {message}"


# One scripted answer in a mock answers file: text, tool calls, or both.
class MockAnswer(StrictModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


# TypeAdapter validates any type, not just a model: here "a dict of case id -> answer".
_ANSWERS = TypeAdapter(dict[str, str | MockAnswer])


# Reads a mock answers file ({case_id: answer}) for `--model mock:FILE`.
def load_answers(path: str | Path) -> dict[str, str | Response]:
    path = Path(path)
    data = _read_yaml_mapping(path, "case ids as keys")
    try:
        answers = _ANSWERS.validate_python(data)
    except ValidationError as e:
        lines = [f"  - {'.'.join(str(k) for k in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise SuiteError(f"{path}: {len(lines)} problem(s)\n" + "\n".join(lines)) from e
    return {
        case_id: a if isinstance(a, str) else Response(text=a.text, tool_calls=a.tool_calls)
        for case_id, a in answers.items()
    }
