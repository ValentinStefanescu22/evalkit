from pathlib import Path

from typer.testing import CliRunner

from evalkit import __version__
from evalkit.cli import app


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
