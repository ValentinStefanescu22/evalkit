# Test suites: the YAML format, its validation, and a loader with readable errors.

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, model_validator

from evalkit.checks import Check, ToolCallCheck
from evalkit.models import StrictModel, ToolDef


class Defaults(StrictModel):
    system: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Case(StrictModel):
    # Ids become database keys and report links, so keep them simple.
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    tags: list[str] = Field(default_factory=list)
    prompt: str = Field(min_length=1)
    system: str | None = None
    document: str | None = None
    tools: list[ToolDef] = Field(default_factory=list)
    checks: list[Check] = Field(min_length=1)

    # Catches typos like `name: get_wether` when the case only offers `get_weather`.
    @model_validator(mode="after")
    def tool_checks_use_known_tools(self) -> "Case":
        offered = {tool.name for tool in self.tools}
        for check in self.checks:
            if isinstance(check, ToolCallCheck) and check.called and check.name not in offered:
                raise ValueError(f"tool_call check refers to unknown tool {check.name!r}")
        return self


class Suite(StrictModel):
    name: str
    description: str = ""
    defaults: Defaults = Field(default_factory=Defaults)
    cases: list[Case] = Field(min_length=1)

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


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
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
        raise SuiteError(f"{path}: top level must be a mapping with `name` and `cases`")

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
