from pathlib import Path
from typing import Any

import pytest

from evalkit.models import Message, Response, ToolCall, ToolDef
from evalkit.providers import MockProvider, ProviderError
from evalkit.runner import build_messages, run_suite
from evalkit.store import Store
from evalkit.suite import Suite

SUITE = Suite.model_validate(
    {
        "name": "demo",
        "defaults": {"system": "Be brief.", "params": {"temperature": 0}},
        "cases": [
            {
                "id": "capital",
                "prompt": "Capital of Romania?",
                "checks": [{"type": "contains", "value": "Bucharest"}],
            },
            {
                "id": "summary",
                "system": "You summarise documents.",
                "prompt": "Summarise this.",
                "document": "Sales grew 5%.",
                "checks": [
                    {"type": "contains", "value": "5%"},
                    {"type": "not_contains", "value": "I can't"},
                ],
            },
            {
                "id": "weather",
                "prompt": "Weather in Bucharest?",
                "tools": [{"name": "get_weather"}],
                "checks": [{"type": "tool_call", "name": "get_weather"}],
            },
        ],
    }
)


def test_system_falls_back_to_default() -> None:
    assert build_messages(SUITE, SUITE.cases[0]) == [
        Message(role="system", content="Be brief."),
        Message(role="user", content="Capital of Romania?"),
    ]


def test_case_system_overrides_default_and_document_is_appended() -> None:
    assert build_messages(SUITE, SUITE.cases[1]) == [
        Message(role="system", content="You summarise documents."),
        Message(role="user", content="Summarise this.\n\nSales grew 5%."),
    ]


def test_no_system_message_when_none_set() -> None:
    suite = Suite.model_validate(
        {
            "name": "x",
            "cases": [{"id": "a", "prompt": "hi", "checks": [{"type": "contains", "value": "x"}]}],
        }
    )
    assert build_messages(suite, suite.cases[0]) == [Message(role="user", content="hi")]


# Records what it was asked, so tests can check tools and params were passed through.
class RecordingProvider(MockProvider):
    def __init__(self, script: dict[str, str | Response]) -> None:
        super().__init__(script)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        self.calls.append({"tools": tools, "params": params})
        return super().complete(messages, tools, params)


def test_run_grades_saves_and_reports_progress(tmp_path: Path) -> None:
    provider = RecordingProvider(
        {
            "Capital of Romania?": "Bucharest.",
            "Summarise this.\n\nSales grew 5%.": "I can't summarise, but it grew 5%.",
            "Weather in Bucharest?": Response(tool_calls=[ToolCall(name="get_weather")]),
        }
    )
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("demo", "demo.yaml", "mock", "url", {})
    seen: list[str] = []

    results = run_suite(run_id, SUITE, provider, store, on_result=lambda r: seen.append(r.case_id))

    assert seen == ["capital", "summary", "weather"]
    # "summary" contains 5% but also "I can't": one failing check fails the case.
    assert [r.passed for r in results] == [True, False, True]
    assert results[1].checks[1].config == {
        "type": "not_contains",
        "value": "I can't",
        "case_sensitive": False,
    }
    assert provider.calls[0] == {"tools": None, "params": {"temperature": 0}}
    tools = provider.calls[2]["tools"]
    assert tools is not None and tools[0].name == "get_weather"
    status = store.conn.execute("SELECT status FROM runs").fetchone()[0]
    assert status == "completed"
    assert store.conn.execute("SELECT count(*) FROM case_results").fetchone()[0] == 3


# Answers normally, then fails on the given prompt, like a server dying mid-run.
class FailingProvider(MockProvider):
    def __init__(self, fail_on: str) -> None:
        super().__init__({}, default="Bucharest.")
        self.fail_on = fail_on

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        if messages[-1].content.startswith(self.fail_on):
            raise ProviderError("cannot reach server")
        return super().complete(messages, tools, params)


def test_provider_error_marks_run_failed_and_keeps_saved_cases(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("demo", "demo.yaml", "mock", "url", {})
    with pytest.raises(ProviderError):
        run_suite(run_id, SUITE, FailingProvider(fail_on="Summarise"), store)
    run = store.conn.execute("SELECT status, error FROM runs").fetchone()
    assert (run["status"], run["error"]) == ("failed", "cannot reach server")
    saved = store.conn.execute("SELECT case_id FROM case_results").fetchall()
    assert [row["case_id"] for row in saved] == ["capital"]
