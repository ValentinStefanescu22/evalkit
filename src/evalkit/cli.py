# Command-line entry point.

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from evalkit import __version__
from evalkit.models import CaseResult
from evalkit.providers import OLLAMA_URL, OpenAICompatibleProvider, Provider, ProviderError
from evalkit.report import build_report, compare_runs, to_html, to_markdown
from evalkit.runner import run_suite
from evalkit.store import Store
from evalkit.suite import DocumentFormat, Suite, SuiteError, load_suite

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
    document_format: Annotated[
        DocumentFormat | None,
        typer.Option(help="How documents/tool output are shown to the model [default: suite's]"),
    ] = None,
    repeats: Annotated[int, typer.Option(min=1, help="Ask every case this many times")] = 1,
    temperature: Annotated[
        float | None, typer.Option(help="Override the suite's temperature [default: suite's]")
    ] = None,
) -> None:
    suite = _load_or_exit(suite_path)
    if temperature is not None:
        # Changed before the run starts, so it is saved in runs.params like any other setting.
        suite.defaults.params["temperature"] = temperature
    fmt = document_format or suite.defaults.document_format
    provider = make_provider(model, base_url)
    width = max(len(c.id) for c in suite.cases)

    def show(result: CaseResult) -> None:
        mark = "✓" if result.passed else "✗"
        seconds = result.response.latency_ms / 1000
        failed = next((c.detail for c in result.checks if not c.passed), "")
        repeat = f" #{result.repeat_index + 1}" if repeats > 1 else ""
        name = f"{result.case_id}{repeat}"
        typer.echo(f"  {mark} {name:<{width + 4}}  {seconds:5.1f}s  {failed}".rstrip())

    with closing(Store(db)) as store:
        run_id = store.start_run(
            suite.name, str(suite_path), model, base_url, suite.defaults.params, fmt
        )
        cases = f"{len(suite.cases)} cases" + (f" x {repeats} repeats" if repeats > 1 else "")
        typer.echo(f"run #{run_id}  {suite.name}  {model}  {fmt}  ({cases})")
        try:
            results = run_suite(
                run_id, suite, provider, store, show, document_format=fmt, repeats=repeats
            )
        except ProviderError as e:
            typer.echo(f"error: {e}", err=True)
            typer.echo(f"run #{run_id} marked failed", err=True)
            raise typer.Exit(code=2) from e

    passed = sum(r.passed for r in results)
    avg = sum(r.response.latency_ms for r in results) / len(results) / 1000
    tokens = sum(r.response.input_tokens + r.response.output_tokens for r in results)
    typer.echo(
        f"{passed}/{len(results)} answers passed ({passed / len(results):.0%})   avg {avg:.1f}s"
        f"   {tokens} tokens   saved to {db} (run #{run_id})"
    )


# Reads a run and its answers, or exits 1 with a clear message.
def _load_run_or_exit(store: Store, run_id: int, db: Path) -> tuple[sqlite3.Row, list[CaseResult]]:
    run = store.get_run(run_id)
    if run is None:
        typer.echo(f"no run #{run_id} in {db}", err=True)
        raise typer.Exit(code=1)
    results = store.load_results(run_id)
    if not results:
        typer.echo(f"run #{run_id} has no saved answers", err=True)
        raise typer.Exit(code=1)
    return run, results


@app.command(help="Write a Markdown and an HTML report for a run.")
def report(
    run_id: int,
    db: Annotated[Path, typer.Option(help="SQLite file with results")] = Path("evalkit.db"),
    out: Annotated[Path, typer.Option(help="Folder for the report files")] = Path("reports"),
) -> None:
    with closing(Store(db)) as store:
        run, results = _load_run_or_exit(store, run_id, db)
    blocks = build_report(run, results)
    title = f"Run #{run_id}: {run['suite_name']} on {run['model']}"
    out.mkdir(parents=True, exist_ok=True)
    md_path, html_path = out / f"run-{run_id}.md", out / f"run-{run_id}.html"
    md_path.write_text(to_markdown(title, blocks), encoding="utf-8")
    html_path.write_text(to_html(title, blocks), encoding="utf-8")
    typer.echo(f"wrote {md_path}")
    typer.echo(f"wrote {html_path}")


@app.command(help="Compare two runs: what got better or worse, and whether it's significant.")
def compare(
    run_a: int,
    run_b: int,
    db: Annotated[Path, typer.Option(help="SQLite file with results")] = Path("evalkit.db"),
) -> None:
    with closing(Store(db)) as store:
        a, results_a = _load_run_or_exit(store, run_a, db)
        b, results_b = _load_run_or_exit(store, run_b, db)
    typer.echo(compare_runs(a, b, results_a, results_b))
