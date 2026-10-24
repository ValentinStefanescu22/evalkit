# Command-line entry point.

from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from evalkit import __version__
from evalkit.models import CaseResult
from evalkit.providers import OLLAMA_URL, OpenAICompatibleProvider, Provider, ProviderError
from evalkit.runner import run_suite
from evalkit.store import Store
from evalkit.suite import Suite, SuiteError, load_suite

app = typer.Typer(help="LLM evaluation and prompt-injection test harness.")


@app.command(help="Print the evalkit version.")
def version() -> None:
    typer.echo(__version__)


def _load_or_exit(suite_path: Path) -> Suite:
    try:
        return load_suite(suite_path)
    except SuiteError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e


@app.command(help="Check that a suite file is valid, without running it.")
def validate(suite_path: Path) -> None:
    suite = _load_or_exit(suite_path)
    typer.echo(f"OK: {suite.name} ({len(suite.cases)} cases)")


# Picks the provider for a model name. Milestone 10 adds `anthropic:...` here.
def make_provider(model: str, base_url: str) -> Provider:
    return OpenAICompatibleProvider(model, base_url=base_url)


@app.command(help="Run a suite against a model and save the results.")
def run(
    suite_path: Path,
    model: Annotated[str, typer.Option(help="Model name, e.g. qwen2.5:3b")],
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible server")] = OLLAMA_URL,
    db: Annotated[Path, typer.Option(help="SQLite file for results")] = Path("evalkit.db"),
) -> None:
    suite = _load_or_exit(suite_path)
    provider = make_provider(model, base_url)
    width = max(len(c.id) for c in suite.cases)

    def show(result: CaseResult) -> None:
        mark = "✓" if result.passed else "✗"
        seconds = result.response.latency_ms / 1000
        failed = next((c.detail for c in result.checks if not c.passed), "")
        typer.echo(f"  {mark} {result.case_id:<{width}}  {seconds:5.1f}s  {failed}".rstrip())

    with closing(Store(db)) as store:
        run_id = store.start_run(
            suite.name, str(suite_path), model, base_url, suite.defaults.params
        )
        typer.echo(f"run #{run_id}  {suite.name}  {model}  ({len(suite.cases)} cases)")
        try:
            results = run_suite(run_id, suite, provider, store, on_result=show)
        except ProviderError as e:
            typer.echo(f"error: {e}", err=True)
            typer.echo(f"run #{run_id} marked failed", err=True)
            raise typer.Exit(code=2) from e

    passed = sum(r.passed for r in results)
    avg = sum(r.response.latency_ms for r in results) / len(results) / 1000
    tokens = sum(r.response.input_tokens + r.response.output_tokens for r in results)
    typer.echo(
        f"{passed}/{len(results)} passed ({passed / len(results):.0%})   avg {avg:.1f}s"
        f"   {tokens} tokens   saved to {db} (run #{run_id})"
    )
