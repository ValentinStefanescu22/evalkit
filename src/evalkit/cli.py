# Command-line entry point. Real commands arrive from milestone 4.

import typer

from evalkit import __version__

app = typer.Typer(help="LLM evaluation and prompt-injection test harness.")


# Keeps evalkit a command group while it has a single command.
@app.callback()
def main() -> None:
    pass


@app.command(help="Print the evalkit version.")
def version() -> None:
    typer.echo(__version__)
