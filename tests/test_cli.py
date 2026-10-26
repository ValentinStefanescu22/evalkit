import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalkit import __version__, cli
from evalkit.cli import app
from evalkit.models import Message, Response, ToolDef
from evalkit.providers import MockProvider, Provider, ProviderError


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_validate_good_suite() -> None:
    result = CliRunner().invoke(app, ["validate", "suites/basic_qa.yaml"])
    assert result.exit_code == 0
    assert "OK: basic_qa" in result.output


def test_validate_bad_suite(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: demo\ncases: []\n")
    result = CliRunner().invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "cases: List should have at least 1 item" in result.output


SUITE = "suites/basic_qa.yaml"


# `monkeypatch` is a pytest fixture that temporarily replaces something and undoes it
# after the test. Here it swaps the real provider for a scripted one: no Ollama needed.
def use_provider(monkeypatch: pytest.MonkeyPatch, provider: Provider) -> None:
    monkeypatch.setattr(cli, "make_provider", lambda model, base_url: provider)


def test_run_prints_progress_and_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_provider(monkeypatch, MockProvider({"What is the capital of Romania?": "Bucharest."}))
    db = tmp_path / "evalkit.db"
    result = CliRunner().invoke(app, ["run", SUITE, "--model", "mock", "--db", str(db)])
    assert result.exit_code == 0
    assert "run #1  basic_qa  mock  plain  (5 cases)" in result.output
    assert "✓ capital_romania" in result.output
    assert "✗ simple_math" in result.output
    assert "expected '144' in answer" in result.output
    assert "1/5 answers passed (20%)" in result.output
    assert "saved to" in result.output and "(run #1)" in result.output
    assert db.exists()


def test_run_numbers_runs_in_the_same_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_provider(monkeypatch, MockProvider({}))
    args = ["run", SUITE, "--model", "mock", "--db", str(tmp_path / "evalkit.db")]
    CliRunner().invoke(app, args)
    assert "run #2" in CliRunner().invoke(app, args).output


def test_run_bad_suite_exits_1(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: demo\ncases: []\n")
    result = CliRunner().invoke(app, ["run", str(bad), "--model", "mock"])
    assert result.exit_code == 1


class DownProvider(MockProvider):
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        raise ProviderError("cannot reach http://localhost:11434/v1 — is Ollama running?")


def test_run_provider_error_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_provider(monkeypatch, DownProvider({}))
    db = tmp_path / "evalkit.db"
    result = CliRunner().invoke(app, ["run", SUITE, "--model", "mock", "--db", str(db)])
    assert result.exit_code == 2
    assert "is Ollama running?" in result.output
    assert "run #1 marked failed" in result.output


def test_run_document_format_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_provider(monkeypatch, MockProvider({}))
    db = tmp_path / "evalkit.db"
    args = ["run", SUITE, "--model", "mock", "--db", str(db), "--document-format", "delimited"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0
    assert "mock  delimited  (5 cases)" in result.output
    saved = sqlite3.connect(db).execute("SELECT document_format FROM runs").fetchone()[0]
    assert saved == "delimited"


def test_run_repeats_and_temperature(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_provider(monkeypatch, MockProvider({"What is the capital of Romania?": "Bucharest."}))
    db = tmp_path / "evalkit.db"
    args = ["run", SUITE, "--model", "mock", "--db", str(db), "--repeats", "2"]
    result = CliRunner().invoke(app, [*args, "--temperature", "0.7"])
    assert result.exit_code == 0
    assert "(5 cases x 2 repeats)" in result.output
    assert "✓ capital_romania #2" in result.output
    assert "2/10 answers passed (20%)" in result.output
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT params FROM runs").fetchone()[0] == '{"temperature": 0.7}'
    assert conn.execute("SELECT count(*) FROM case_results").fetchone()[0] == 10


def test_run_rejects_zero_repeats() -> None:
    result = CliRunner().invoke(app, ["run", SUITE, "--model", "mock", "--repeats", "0"])
    assert result.exit_code == 2  # Typer's own "invalid value" code
    assert "Invalid value" in result.output


# Two runs of basic_qa in one DB: run 1 answers nothing right, run 2 gets Romania right.
def two_runs(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    base = ["run", SUITE, "--model", "mock", "--db", str(db), "--repeats", "5"]
    use_provider(monkeypatch, MockProvider({}))
    CliRunner().invoke(app, base)
    use_provider(monkeypatch, MockProvider({"What is the capital of Romania?": "Bucharest."}))
    CliRunner().invoke(app, base)


def test_report_writes_markdown_and_html(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "evalkit.db"
    two_runs(monkeypatch, db)
    out = tmp_path / "reports"
    result = CliRunner().invoke(app, ["report", "2", "--db", str(db), "--out", str(out)])
    assert result.exit_code == 0
    md = (out / "run-2.md").read_text()
    assert md.startswith("# Run #2: basic_qa on mock")
    assert "| all answers | 5/25 | 20% |" in md
    assert (out / "run-2.html").read_text().startswith("<!doctype html>")


def test_compare_prints_significance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "evalkit.db"
    two_runs(monkeypatch, db)
    result = CliRunner().invoke(app, ["compare", "1", "2", "--db", str(db)])
    assert result.exit_code == 0
    # capital_romania went 0/5 -> 5/5: the intervals don't overlap.
    assert "cases: 1 significant, 0 noise, 4 unchanged" in result.output
    assert "capital_romania" in result.output


def test_unknown_run_exits_1(tmp_path: Path) -> None:
    db = tmp_path / "evalkit.db"
    for args in (["report", "7"], ["compare", "7", "8"]):
        result = CliRunner().invoke(app, [*args, "--db", str(db)])
        assert result.exit_code == 1
        assert f"no run #7 in {db}" in result.output
