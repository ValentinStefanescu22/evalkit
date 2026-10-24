# Checks score a single Response and return a CheckResult.

from evalkit.models import CheckResult, Response


def contains(response: Response, value: str, case_sensitive: bool = False) -> CheckResult:
    text = response.text if case_sensitive else response.text.lower()
    needle = value if case_sensitive else value.lower()
    passed = needle in text
    detail = "" if passed else f"expected {value!r} in answer"
    return CheckResult(check="contains", passed=passed, detail=detail)
