import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from evalkit import __version__, cli
from evalkit.anthropic_provider import AnthropicProvider
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
    monkeypatch.setattr(cli, "make_provider", lambda model, base_url, suite, fmt: provider)


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


CI_ARGS = [
    "run",
    SUITE,
    "--model",
    "mock:ci/basic_qa_answers.yaml",
    "--fail-below-baseline",
    "--baseline",
    "ci/basic_qa_baseline.json",
]


# The exact command CI runs, against the committed files: must pass, so CI can't drift.
def test_ci_command_passes(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, [*CI_ARGS, "--db", str(tmp_path / "ci.db")])
    assert result.exit_code == 0, result.output
    assert "5/5 answers passed" in result.output
    assert "no significant drop" in result.output


def write_answers(tmp_path: Path, correct: int) -> str:
    right = ["Bucharest", "Danube", "144", "Guido van Rossum", "404"]
    ids = ["capital_romania", "longest_river_romania", "simple_math", "python_creator"]
    ids.append("http_not_found")
    lines = [f'{i}: "{right[n] if n < correct else "no idea"}"' for n, i in enumerate(ids)]
    path = tmp_path / f"answers_{correct}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return f"mock:{path}"


def gate(tmp_path: Path, correct: int) -> Any:
    args = [
        "run",
        SUITE,
        "--model",
        write_answers(tmp_path, correct),
        "--db",
        str(tmp_path / "x.db"),
    ]
    args += ["--fail-below-baseline", "--baseline", "ci/basic_qa_baseline.json"]
    return CliRunner().invoke(app, args)


def test_gate_fails_on_significant_drop(tmp_path: Path) -> None:
    result = gate(tmp_path, correct=0)
    assert result.exit_code == 3
    assert "now: 0/5  0% [0%–43%]   significant drop" in result.output
    assert "warning: baseline was mock:ci/basic_qa_answers.yaml" in result.output


# 1/5 = [4%–62%] still overlaps the 5/5 baseline [57%–100%]: noise, not a regression.
def test_gate_ignores_noise(tmp_path: Path) -> None:
    result = gate(tmp_path, correct=1)
    assert result.exit_code == 0
    assert "no significant drop" in result.output


def test_gate_needs_a_baseline(tmp_path: Path) -> None:
    missing = tmp_path / "none.json"
    args = [*CI_ARGS[:4], "--db", str(tmp_path / "x.db"), "--fail-below-baseline"]
    result = CliRunner().invoke(app, [*args, "--baseline", str(missing)])
    assert result.exit_code == 1
    assert "set one with: evalkit baseline set RUN_ID" in result.output
    # Checked before the run starts: nothing was run or saved.
    assert "run #" not in result.output


def test_gate_rejects_baseline_of_another_suite(tmp_path: Path) -> None:
    wrong = tmp_path / "b.json"
    wrong.write_text(Path("ci/basic_qa_baseline.json").read_text().replace("basic_qa", "other"))
    args = [
        "run",
        SUITE,
        "--model",
        "mock:ci/basic_qa_answers.yaml",
        "--db",
        str(tmp_path / "x.db"),
    ]
    result = CliRunner().invoke(app, [*args, "--fail-below-baseline", "--baseline", str(wrong)])
    assert result.exit_code == 1
    assert "is a baseline for other, not basic_qa" in result.output


def test_bad_answers_file_exits_1(tmp_path: Path) -> None:
    bad = tmp_path / "a.yaml"
    bad.write_text("capitl_romania: Bucharest\n")
    result = CliRunner().invoke(app, ["run", SUITE, "--model", f"mock:{bad}"])
    assert result.exit_code == 1
    assert "not in basic_qa: capitl_romania" in result.output


def test_baseline_set_writes_json(tmp_path: Path) -> None:
    db = tmp_path / "evalkit.db"
    CliRunner().invoke(app, ["run", SUITE, "--model", write_answers(tmp_path, 4), "--db", str(db)])
    path = tmp_path / "b.json"
    result = CliRunner().invoke(
        app, ["baseline", "set", "1", "--db", str(db), "--baseline", str(path)]
    )
    assert result.exit_code == 0
    assert "baseline for basic_qa: 4/5  80% [38%–96%] (run #1)" in result.output
    saved = json.loads(path.read_text())
    assert (saved["suite"], saved["run_id"], saved["passed"], saved["total"]) == (
        "basic_qa",
        1,
        4,
        5,
    )


# Answers the first case, then the "server" dies: a failed run that still has saved answers.
class DiesAfterOneProvider(MockProvider):
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        if self.default == "dead":
            raise ProviderError("server went away")
        self.default = "dead"
        return Response(text="Bucharest.")


def test_baseline_set_refuses_failed_or_unknown_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "evalkit.db"
    use_provider(monkeypatch, DiesAfterOneProvider({}))
    assert CliRunner().invoke(app, ["run", SUITE, "--model", "x", "--db", str(db)]).exit_code == 2
    args = ["--db", str(db), "--baseline", str(tmp_path / "b.json")]
    failed = CliRunner().invoke(app, ["baseline", "set", "1", *args])
    assert failed.exit_code == 1
    assert "run #1 is failed; only completed runs can be a baseline" in failed.output
    assert CliRunner().invoke(app, ["baseline", "set", "9", *args]).exit_code == 1
    assert not (tmp_path / "b.json").exists()


OPEN = "suites/open_answers.yaml"


def test_run_with_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The same mock answers as model and judge: every reply is a valid verdict.
    use_provider(monkeypatch, MockProvider({}, default='{"reasoning": "Fine.", "score": 5}'))
    db = tmp_path / "evalkit.db"
    result = CliRunner().invoke(app, ["run", OPEN, "--model", "m", "--judge", "j", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "run #1  open_answers  m  plain  judge j  (8 cases)" in result.output
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT judge FROM runs").fetchone()[0] == "j"
    scores = conn.execute("SELECT DISTINCT score FROM check_results WHERE type = 'rubric'")
    assert scores.fetchall() == [(5,)]


def test_rubric_suite_needs_a_judge(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["run", OPEN, "--model", "m", "--db", str(tmp_path / "x.db")])
    assert result.exit_code == 1
    assert "open_answers has rubric checks; pass --judge MODEL" in result.output


def test_judge_cannot_be_a_mock(tmp_path: Path) -> None:
    args = ["run", OPEN, "--model", "m", "--judge", "mock:x.yaml", "--db", str(tmp_path / "x.db")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "--judge can't be a mock" in result.output


# A judged run of open_answers, then export -> fill in labels -> judge-agreement.
def test_labels_export_then_judge_agreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_provider(monkeypatch, MockProvider({}, default='{"reasoning": "Fine.", "score": 4}'))
    db = tmp_path / "evalkit.db"
    CliRunner().invoke(app, ["run", OPEN, "--model", "m", "--judge", "j", "--db", str(db)])

    out = tmp_path / "labels" / "run-1.yaml"
    exported = CliRunner().invoke(
        app, ["labels", "export", "1", "--db", str(db), "--out", str(out)]
    )
    assert exported.exit_code == 0, exported.output
    assert "8 answers to label" in exported.output
    text = out.read_text()
    # The judge saw the meeting notes, and so do you.
    assert "Ana reported the mobile app beta" in text

    empty = CliRunner().invoke(app, ["judge-agreement", str(out), "--db", str(db)])
    assert empty.exit_code == 0
    assert "labelled 0 of 8 answers" in empty.output

    data = yaml.safe_load(text)
    for entry in data["answers"]:
        entry["pass"], entry["score"] = True, 4
    data["answers"][0]["pass"], data["answers"][0]["score"] = False, 2
    out.write_text(yaml.safe_dump(data))
    result = CliRunner().invoke(app, ["judge-agreement", str(out), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "pass/fail   agreement 7/8 88%" in result.output
    assert "explain_recursion  repeat 0   A: pass 4   B: fail 2   'Fine.'" in result.output

    # The same labels as a reference: a labels file compared with itself agrees completely.
    args = ["judge-agreement", str(out), "--reference", str(out), "--db", str(db)]
    same = CliRunner().invoke(app, args)
    assert same.exit_code == 0, same.output
    assert "agreement 8/8 100%   kappa 1.00 (almost perfect)" in same.output


def test_labels_export_never_overwrites(tmp_path: Path) -> None:
    existing = tmp_path / "run-1.yaml"
    existing.write_text("my precious labels")
    args = ["labels", "export", "1", "--db", str(tmp_path / "x.db"), "--out", str(existing)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "already exists; not overwriting your labels" in result.output
    assert existing.read_text() == "my precious labels"


def test_judge_agreement_bad_inputs(tmp_path: Path) -> None:
    db = str(tmp_path / "x.db")
    missing = CliRunner().invoke(app, ["judge-agreement", str(tmp_path / "none.yaml"), "--db", db])
    assert missing.exit_code == 1
    labels = tmp_path / "l.yaml"
    labels.write_text("run_id: 7\nanswers: []\n")
    unknown = CliRunner().invoke(app, ["judge-agreement", str(labels), "--db", db])
    assert unknown.exit_code == 1
    assert "no run #7" in unknown.output


# A Claude provider whose HTTP calls go to a fake transport. Counting tokens answers 100;
# a messages call answers `text`. `paths` records every request, to prove what was called.
def fake_claude(paths: list[str], text: str = "Bucharest.", status: int = 200) -> Any:
    import anthropic
    import httpx2

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if status != 200:
            error = {"type": "authentication_error", "message": "no key"}
            return httpx2.Response(status, json={"type": "error", "error": error})
        if request.url.path.endswith("count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 100})
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "stop_details": None,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

    client = anthropic.Anthropic(
        api_key="test-key",
        base_url="https://api.anthropic.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return AnthropicProvider("claude-opus-5-5", client=client)


CLAUDE = ["run", SUITE, "--model", "anthropic:claude-opus-5-5"]


def test_claude_run_needs_yes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths: list[str] = []
    use_provider(monkeypatch, fake_claude(paths))
    result = CliRunner().invoke(app, [*CLAUDE, "--db", str(tmp_path / "x.db")])
    assert result.exit_code == 1
    assert "cost estimate (Claude API, paid):" in result.output
    assert "claude-opus-5-5 answering: 5 calls, 500 input tokens (counted)" in result.output
    assert "re-run with --yes to spend this" in result.output
    # Only free token counting happened: not a single paid call, and nothing saved.
    assert paths == ["/v1/messages/count_tokens"] * 5
    assert not (tmp_path / "x.db").exists()


def test_claude_run_with_yes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths: list[str] = []
    use_provider(monkeypatch, fake_claude(paths))
    db = tmp_path / "x.db"
    result = CliRunner().invoke(app, [*CLAUDE, "--db", str(db), "--yes"])
    assert result.exit_code == 0, result.output
    assert "note: not sending temperature (Claude rejects it)" in result.output
    assert "anthropic:claude-opus-5-5  plain  at https://api.anthropic.test" in result.output
    # 5 calls x (100 in x $4/M + 50 out x $20/M) = 5 x $0.0014
    assert "spent $0.0070 on the Claude API" in result.output
    assert paths.count("/v1/messages") == 5
    row = sqlite3.connect(db).execute("SELECT params, base_url FROM runs").fetchone()
    assert row == ("{}", "https://api.anthropic.test")  # temperature not recorded as sent


def test_claude_rejects_temperature_flag(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, [*CLAUDE, "--temperature", "0.7", "--db", str(tmp_path / "x")])
    assert result.exit_code == 1
    assert "--temperature can't be used with Claude" in result.output


def test_claude_without_credentials_fails_before_spending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths: list[str] = []
    use_provider(monkeypatch, fake_claude(paths, status=401))
    result = CliRunner().invoke(app, [*CLAUDE, "--db", str(tmp_path / "x.db"), "--yes"])
    assert result.exit_code == 2
    assert "set ANTHROPIC_API_KEY" in result.output
    assert paths == ["/v1/messages/count_tokens"]


def test_unknown_claude_model_exits_1(tmp_path: Path) -> None:
    args = ["run", SUITE, "--model", "anthropic:claude-nope", "--db", str(tmp_path / "x.db")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "no price known for 'claude-nope'" in result.output


# open_answers run with a lenient judge, then rejudged by a strict one: same answers, new
# scores, and the labels made for the first run can be compared with the second.
def test_rejudge_then_agreement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "evalkit.db"
    use_provider(monkeypatch, MockProvider({}, default='{"reasoning": "Fine.", "score": 4}'))
    base = ["--db", str(db)]
    CliRunner().invoke(app, ["run", OPEN, "--model", "m", "--judge", "j", *base])
    out = tmp_path / "labels.yaml"
    CliRunner().invoke(app, ["labels", "export", "1", "--out", str(out), *base])
    data = yaml.safe_load(out.read_text())
    for entry in data["answers"]:
        entry["pass"], entry["score"] = False, 2
    out.write_text(yaml.safe_dump(data))

    use_provider(monkeypatch, MockProvider({}, default='{"reasoning": "Too vague.", "score": 2}'))
    result = CliRunner().invoke(app, ["rejudge", "1", "--judge", "strict", *base])
    assert result.exit_code == 0, result.output
    assert "run #2  rejudge of run #1  open_answers  m  judge strict  (8 answers)" in result.output
    run = sqlite3.connect(db).execute("SELECT judge, source_run FROM runs WHERE id = 2").fetchone()
    assert run == ("strict", 1)

    old = CliRunner().invoke(app, ["judge-agreement", str(out), *base])
    new = CliRunner().invoke(app, ["judge-agreement", str(out), "--run", "2", *base])
    assert "agreement 0/8 0%" in old.output  # the lenient judge passed what you failed
    assert "agreement 8/8 100%" in new.output
    assert "A = judge strict" in new.output


def test_rejudge_with_claude_needs_yes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "evalkit.db"
    use_provider(monkeypatch, MockProvider({}, default='{"reasoning": "Fine.", "score": 4}'))
    CliRunner().invoke(app, ["run", OPEN, "--model", "m", "--judge", "j", "--db", str(db)])
    paths: list[str] = []
    use_provider(monkeypatch, fake_claude(paths))
    args = ["rejudge", "1", "--judge", "anthropic:claude-opus-5-5", "--db", str(db)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "claude-opus-5-5 as judge: 8 calls" in result.output
    assert paths == ["/v1/messages/count_tokens"] * 8
    assert sqlite3.connect(db).execute("SELECT count(*) FROM runs").fetchone()[0] == 1
