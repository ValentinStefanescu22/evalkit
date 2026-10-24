# Command-line entry point. `run`, `report` and friends arrive from milestone 4.

from pathlib import Path

import typer

from evalkit import __version__
from evalkit.suite import SuiteError, load_suite

app = typer.Typer(help="LLM evaluation and prompt-injection test harness.")


@app.command(help="Print the evalkit version.")
def version() -> None:
    typer.echo(__version__)


@app.command(help="Check that a suite file is valid, without running it.")
def validate(suite_path: Path) -> None:
    try:
        suite = load_suite(suite_path)
    except SuiteError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"OK: {suite.name} ({len(suite.cases)} cases)")
