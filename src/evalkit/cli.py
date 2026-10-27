# Command-line entry point.

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from evalkit import __version__
from evalkit.baseline import (
    Baseline,
    below_baseline,
    default_path,
    load_baseline,
    make_baseline,
    save_baseline,
)
from evalkit.labels import (
    LabelError,
    agreement,
    export_labels,
    load_labels,
    reference_agreement,
)
from evalkit.models import CaseResult
from evalkit.providers import (
    OLLAMA_URL,
    MockProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
)
from evalkit.report import build_report, compare_runs, format_rate, to_html, to_markdown
from evalkit.runner import judge_question, mock_script, run_suite
from evalkit.store import Store
from evalkit.suite import DocumentFormat, Suite, SuiteError, load_answers, load_suite

app = typer.Typer(help="LLM evaluation and prompt-injection test harness.")
baseline_app = typer.Typer(help="Save a known-good run to compare new runs against.")
app.add_typer(baseline_app, name="baseline")
labels_app = typer.Typer(help="Hand labels for checking the LLM judge.")
app.add_typer(labels_app, name="labels")


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


# Picks the provider for a model name. `mock:FILE` = scripted answers from FILE (used in CI).
# Milestone 10 adds `anthropic:...` here. Raises SuiteError for a bad answers file.
def make_provider(
    model: str, base_url: str, suite: Suite, document_format: DocumentFormat
) -> Provider:
    if model.startswith("mock:"):
        answers = load_answers(model.removeprefix("mock:"))
        return MockProvider(mock_script(suite, answers, document_format))
    return OpenAICompatibleProvider(model, base_url=base_url)


# Reads the baseline file for `run --fail-below-baseline`, or exits 1 with a clear message.
def _load_baseline_or_exit(path: Path, suite_name: str) -> Baseline:
    try:
        baseline = load_baseline(path)
    except OSError as e:
        typer.echo(f"no baseline at {path}; set one with: evalkit baseline set RUN_ID", err=True)
        raise typer.Exit(code=1) from e
    except ValidationError as e:
        typer.echo(f"{path} is not a valid baseline file:\n{e}", err=True)
        raise typer.Exit(code=1) from e
    if baseline.suite != suite_name:
        typer.echo(f"{path} is a baseline for {baseline.suite}, not {suite_name}", err=True)
        raise typer.Exit(code=1)
    return baseline


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
    fail_below_baseline: Annotated[
        bool, typer.Option(help="Exit with code 3 if the pass rate drops significantly")
    ] = False,
    baseline: Annotated[
        Path | None, typer.Option(help="Baseline file [default: baselines/<suite>.json]")
    ] = None,
    judge: Annotated[
        str | None, typer.Option(help="Model that grades rubric checks, e.g. llama3.2:3b")
    ] = None,
) -> None:
    suite = _load_or_exit(suite_path)
    if temperature is not None:
        # Changed before the run starts, so it is saved in runs.params like any other setting.
        suite.defaults.params["temperature"] = temperature
    fmt = document_format or suite.defaults.document_format
    try:
        provider = make_provider(model, base_url, suite, fmt)
    except SuiteError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    judge_provider = None
    if judge is not None:
        if judge.startswith("mock:"):
            # Mock answers are keyed by case id; a judge gets a different prompt per answer.
            typer.echo("--judge can't be a mock: answers files script cases, not a judge", err=True)
            raise typer.Exit(code=1)
        judge_provider = make_provider(judge, base_url, suite, fmt)
    elif suite.uses_judge:
        typer.echo(
            f"{suite.name} has rubric checks; pass --judge MODEL (e.g. llama3.2:3b)", err=True
        )
        raise typer.Exit(code=1)

    # Checked before the run, so a missing baseline doesn't waste a long run.
    expected = None
    if fail_below_baseline:
        expected = _load_baseline_or_exit(baseline or default_path(suite.name), suite.name)
        if expected.model != model:
            typer.echo(f"warning: baseline was {expected.model}, this run is {model}", err=True)
    width = max(len(c.id) for c in suite.cases)

    def show(result: CaseResult) -> None:
        mark = "✓" if result.passed else "✗"
        seconds = result.response.latency_ms / 1000
        # Cut long details (a judge's reasoning) so each case stays on one line.
        failed = next((c.detail for c in result.checks if not c.passed), "")[:100]
        repeat = f" #{result.repeat_index + 1}" if repeats > 1 else ""
        name = f"{result.case_id}{repeat}"
        typer.echo(f"  {mark} {name:<{width + 4}}  {seconds:5.1f}s  {failed}".rstrip())

    with closing(Store(db)) as store:
        run_id = store.start_run(
            suite.name, str(suite_path), model, base_url, suite.defaults.params, fmt, judge
        )
        cases = f"{len(suite.cases)} cases" + (f" x {repeats} repeats" if repeats > 1 else "")
        judged = f"  judge {judge}" if judge else ""
        typer.echo(f"run #{run_id}  {suite.name}  {model}  {fmt}{judged}  ({cases})")
        try:
            results = run_suite(
                run_id,
                suite,
                provider,
                store,
                show,
                document_format=fmt,
                repeats=repeats,
                judge=judge_provider,
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

    if expected:
        dropped = below_baseline(expected, passed, len(results))
        typer.echo(
            f"baseline run #{expected.run_id}: {format_rate(expected.passed, expected.total)}"
            f"   now: {format_rate(passed, len(results))}"
            f"   {'significant drop' if dropped else 'no significant drop'}"
        )
        if dropped:
            raise typer.Exit(code=3)


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


@baseline_app.command("set", help="Save a completed run as the baseline for its suite.")
def set_baseline(
    run_id: int,
    db: Annotated[Path, typer.Option(help="SQLite file with results")] = Path("evalkit.db"),
    baseline: Annotated[
        Path | None, typer.Option(help="Where to write it [default: baselines/<suite>.json]")
    ] = None,
) -> None:
    with closing(Store(db)) as store:
        run, results = _load_run_or_exit(store, run_id, db)
    if run["status"] != "completed":
        typer.echo(
            f"run #{run_id} is {run['status']}; only completed runs can be a baseline", err=True
        )
        raise typer.Exit(code=1)
    new = make_baseline(run, results)
    path = baseline or default_path(new.suite)
    save_baseline(path, new)
    typer.echo(f"baseline for {new.suite}: {format_rate(new.passed, new.total)} (run #{run_id})")
    typer.echo(f"saved to {path}")


@labels_app.command(
    "export", help="Write a run's judged answers to a YAML file for hand labelling."
)
def export(
    run_id: int,
    db: Annotated[Path, typer.Option(help="SQLite file with results")] = Path("evalkit.db"),
    out: Annotated[
        Path | None, typer.Option(help="Labels file to write [default: labels/run-<id>.yaml]")
    ] = None,
) -> None:
    path = out or Path("labels") / f"run-{run_id}.yaml"
    # Never overwrite: that file may hold hours of your labelling.
    if path.exists():
        typer.echo(f"{path} already exists; not overwriting your labels", err=True)
        raise typer.Exit(code=1)
    with closing(Store(db)) as store:
        run, results = _load_run_or_exit(store, run_id, db)
    # The DB stores only the prompt; the suite file also has the document the judge saw.
    try:
        questions = {c.id: judge_question(c) for c in load_suite(run["suite_path"]).cases}
    except SuiteError:
        typer.echo(f"warning: can't read {run['suite_path']}; showing prompts only", err=True)
        questions = {}
    try:
        text, count = export_labels(run, results, questions)
    except LabelError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {path}: {count} answers to label (pass: true/false, score: 1-5)")


@app.command(help="Compare a labels file with the LLM judge: agreement and Cohen's kappa.")
def judge_agreement(
    labels_file: Path,
    db: Annotated[Path, typer.Option(help="SQLite file with results")] = Path("evalkit.db"),
    reference: Annotated[
        Path | None,
        typer.Option(help="Compare with this labels file instead of the judge"),
    ] = None,
) -> None:
    try:
        labels = load_labels(labels_file)
        other = load_labels(reference) if reference else None
    except LabelError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    with closing(Store(db)) as store:
        run, results = _load_run_or_exit(store, labels.run_id, db)
    try:
        if reference and other is not None:
            text = reference_agreement(
                run, other, labels, results, str(reference), str(labels_file)
            )
        else:
            text = agreement(run, labels, results, str(labels_file))
    except LabelError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    typer.echo(text)
