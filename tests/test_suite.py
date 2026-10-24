from pathlib import Path

import pytest

from evalkit.checks import (
    ContainsCheck,
    RegexCheck,
    ToolCallCheck,
    ValidJsonCheck,
)
from evalkit.suite import SuiteError, load_suite

VALID = """
name: demo
defaults:
  system: "Be brief."
  params: {temperature: 0}
cases:
  - id: capital_romania
    tags: [geography]
    prompt: "What is the capital of Romania?"
    checks:
      - type: contains
        value: Bucharest
"""


# `tmp_path` is a pytest fixture: pytest sees the argument name and passes in a
# fresh empty folder for each test, deleted afterwards.
def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(text)
    return path


def test_valid_suite_loads(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, VALID))
    assert suite.name == "demo"
    assert suite.defaults.system == "Be brief."
    assert suite.defaults.params == {"temperature": 0}
    case = suite.cases[0]
    assert case.id == "capital_romania"
    assert case.tags == ["geography"]
    assert case.checks == [ContainsCheck(type="contains", value="Bucharest")]


@pytest.mark.parametrize("name", ["basic_qa.yaml", "structured_output.yaml"])
def test_shipped_example_suites_load(name: str) -> None:
    suite = load_suite(Path(__file__).parent.parent / "suites" / name)
    assert len(suite.cases) >= 5


EVERY_CHECK = """
name: demo
cases:
  - id: q1
    prompt: "Weather in Bucharest?"
    tools:
      - name: get_weather
        parameters: {type: object, properties: {city: {type: string}}}
    checks:
      - {type: contains, value: a}
      - {type: not_contains, value: b}
      - {type: regex, pattern: "\\\\d+"}
      - {type: valid_json, json_schema: {type: object}}
      - {type: tool_call, name: get_weather, arguments: {city: Bucharest}}
"""


# The `type` field in YAML must pick the matching check class.
def test_every_check_type_loads_from_yaml(tmp_path: Path) -> None:
    case = load_suite(write(tmp_path, EVERY_CHECK)).cases[0]
    assert [type(c) for c in case.checks] == [
        ContainsCheck,
        ContainsCheck,
        RegexCheck,
        ValidJsonCheck,
        ToolCallCheck,
    ]
    assert [c.type for c in case.checks][:2] == ["contains", "not_contains"]
    assert case.tools[0].name == "get_weather"


ONE_CASE = """
name: demo
cases:
  - id: {id}
    {field}
    checks: {checks}
"""


def case(
    id: str = "q1",
    field: str = 'prompt: "Hi"',
    checks: str = "[{type: contains, value: x}]",
) -> str:
    return ONE_CASE.format(id=id, field=field, checks=checks)


DUPLICATE_IDS = """
name: demo
cases:
  - {id: q1, prompt: "Hi", checks: [{type: contains, value: x}]}
  - {id: q1, prompt: "Bye", checks: [{type: contains, value: x}]}
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (case(field="tags: []"), "cases[0] (q1).prompt: Field required"),
        (case(field='promt: "Hi"'), "cases[0] (q1).promt: Extra inputs are not permitted"),
        (case(field='prompt: ""'), "cases[0] (q1).prompt: String should have at least 1"),
        (case(checks="[{type: contians, value: x}]"), "does not match any of the expected tags"),
        (case(checks="[{type: contains}]"), "checks[0].contains.value: Field required"),
        (case(checks="[]"), "cases[0] (q1).checks: List should have at least 1 item"),
        (case(id="Bad Id!"), "cases[0] (Bad Id!).id: String should match pattern"),
        ("name: demo\ncases: []\n", "cases: List should have at least 1 item"),
        ("cases: []\n", "name: Field required"),
        (DUPLICATE_IDS, "duplicate case ids: q1"),
        ("name: demo\ncases: [\n", "invalid YAML"),
        ("", "file is empty"),
        ("- just\n- a list\n", "top level must be a mapping"),
        (
            case(checks='[{type: regex, pattern: "(unclosed"}]'),
            "checks[0].regex.pattern: invalid regex: missing )",
        ),
        (
            case(checks="[{type: valid_json, json_schema: {type: not-a-type}}]"),
            "invalid JSON schema",
        ),
        (
            case(checks="[{type: tool_call, name: get_wether}]"),
            "cases[0] (q1): tool_call check refers to unknown tool 'get_wether'",
        ),
    ],
)
def test_bad_suites_give_readable_errors(tmp_path: Path, text: str, expected: str) -> None:
    with pytest.raises(SuiteError) as exc:
        load_suite(write(tmp_path, text))
    assert expected in str(exc.value)
    assert "suite.yaml" in str(exc.value)


def test_yaml_error_mentions_line(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="line 3"):
        load_suite(write(tmp_path, "name: demo\ncases:\n  - id: [unclosed\n"))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="cannot read file"):
        load_suite(tmp_path / "nope.yaml")
