import pytest

from evalkit.checks import (
    ContainsCheck,
    RegexCheck,
    ToolCallCheck,
    ValidJsonCheck,
)
from evalkit.models import Response, ToolCall


@pytest.mark.parametrize(
    ("answer", "value", "case_sensitive", "expected"),
    [
        ("The capital is Bucharest.", "Bucharest", False, True),
        ("the capital is bucharest.", "Bucharest", False, True),
        ("the capital is bucharest.", "Bucharest", True, False),
        ("The capital is Cluj.", "Bucharest", False, False),
        ("", "Bucharest", False, False),
    ],
)
def test_contains(answer: str, value: str, case_sensitive: bool, expected: bool) -> None:
    check = ContainsCheck(value=value, case_sensitive=case_sensitive)
    result = check.run(Response(text=answer))
    assert result.passed is expected
    assert result.check == "contains"
    assert ("Bucharest" in result.detail) is not expected


@pytest.mark.parametrize(
    ("answer", "value", "expected"),
    [
        ("5000 metres", "I can't", True),
        ("I can't help with that.", "I can't", False),
        ("i CAN'T do it", "I can't", False),
        ("", "I can't", True),
    ],
)
def test_not_contains(answer: str, value: str, expected: bool) -> None:
    result = ContainsCheck(type="not_contains", value=value).run(Response(text=answer))
    assert result.passed is expected
    assert ("forbidden" in result.detail) is not expected
    assert result.check == "not_contains"


@pytest.mark.parametrize(
    ("answer", "pattern", "expected"),
    [
        ("2026-10-24", r"^\d{4}-\d{2}-\d{2}$", True),
        ("The date is 2026-10-24.", r"\d{4}-\d{2}-\d{2}", True),
        ("The date is 2026-10-24.", r"^\d{4}-\d{2}-\d{2}$", False),
        ("24/10/2026", r"\d{4}-\d{2}-\d{2}", False),
        ("BUCHAREST", r"(?i)bucharest", True),
        ("", r"\d+", False),
    ],
)
def test_regex(answer: str, pattern: str, expected: bool) -> None:
    result = RegexCheck(pattern=pattern).run(Response(text=answer))
    assert result.passed is expected


def test_regex_rejects_broken_pattern() -> None:
    with pytest.raises(ValueError, match="invalid regex"):
        RegexCheck(pattern="(unclosed")


PERSON_SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}


@pytest.mark.parametrize(
    ("answer", "expected", "detail"),
    [
        ('{"name": "Ana", "age": 29}', True, ""),
        ('```json\n{"name": "Ana", "age": 29}\n```', True, ""),
        ('```\n{"name": "Ana", "age": 29}\n```', True, ""),
        ('Sure! {"name": "Ana", "age": 29}', False, "not valid JSON"),
        ('{"name": "Ana", "age": 29', False, "not valid JSON"),
        ("", False, "not valid JSON"),
        ('{"name": "Ana"}', False, "'age' is a required property"),
        ('{"name": "Ana", "age": "29"}', False, "$.age: '29' is not of type 'integer'"),
    ],
)
def test_valid_json_with_schema(answer: str, expected: bool, detail: str) -> None:
    result = ValidJsonCheck(json_schema=PERSON_SCHEMA).run(Response(text=answer))
    assert result.passed is expected
    assert detail in result.detail


def test_valid_json_without_schema_accepts_any_json() -> None:
    assert ValidJsonCheck().run(Response(text="[1, 2, 3]")).passed
    assert ValidJsonCheck().run(Response(text="null")).passed


def test_valid_json_rejects_broken_schema() -> None:
    with pytest.raises(ValueError, match="invalid JSON schema"):
        ValidJsonCheck(json_schema={"type": "not-a-type"})


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


@pytest.mark.parametrize(
    ("calls", "check", "expected"),
    [
        ([call("get_weather", city="Bucharest")], ToolCallCheck(name="get_weather"), True),
        (
            [call("get_weather", city="Bucharest", units="metric")],
            ToolCallCheck(name="get_weather", arguments={"city": "Bucharest"}),
            True,
        ),
        (
            [call("get_weather", city="Cluj")],
            ToolCallCheck(name="get_weather", arguments={"city": "Bucharest"}),
            False,
        ),
        ([call("get_time")], ToolCallCheck(name="get_weather"), False),
        ([], ToolCallCheck(name="get_weather"), False),
        (
            [call("get_time"), call("get_weather", city="Bucharest")],
            ToolCallCheck(name="get_weather", arguments={"city": "Bucharest"}),
            True,
        ),
        (
            [call("book", place={"city": "Bucharest", "zip": "010011"})],
            ToolCallCheck(name="book", arguments={"place": {"city": "Bucharest", "zip": "010011"}}),
            True,
        ),
        ([], ToolCallCheck(name="delete_file", called=False), True),
        ([call("delete_file", path="/")], ToolCallCheck(name="delete_file", called=False), False),
        (
            [call("send_email", to="me@example.com")],
            ToolCallCheck(name="send_email", arguments={"to": "evil@example.com"}, called=False),
            True,
        ),
    ],
)
def test_tool_call(calls: list[ToolCall], check: ToolCallCheck, expected: bool) -> None:
    assert check.run(Response(tool_calls=calls)).passed is expected


def test_tool_call_failure_shows_what_happened() -> None:
    check = ToolCallCheck(name="get_weather", arguments={"city": "Bucharest"})
    detail = check.run(Response(tool_calls=[call("get_weather", city="Cluj")])).detail
    assert detail == (
        "expected call get_weather({'city': 'Bucharest'}); got get_weather({'city': 'Cluj'})"
    )
    assert "got no tool calls" in check.run(Response()).detail
    forbidden = ToolCallCheck(name="delete_file", called=False)
    assert forbidden.run(Response(tool_calls=[call("delete_file", path="/")])).detail == (
        "forbidden call: delete_file({'path': '/'})"
    )
