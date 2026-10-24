import pytest

from evalkit.checks import contains
from evalkit.models import Response


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
    result = contains(Response(text=answer), value, case_sensitive=case_sensitive)
    assert result.passed is expected
    assert result.check == "contains"


def test_contains_failure_explains_itself() -> None:
    result = contains(Response(text="nope"), "Bucharest")
    assert "Bucharest" in result.detail
