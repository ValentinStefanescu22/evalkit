import sqlite3
from pathlib import Path

from evalkit.models import CaseResult, CheckResult, Response
from evalkit.report import (
    attacks_by_tag,
    build_report,
    by_case,
    by_tag,
    compare_runs,
    format_rate,
    to_html,
    to_markdown,
)
from evalkit.store import Store


def answer(
    case_id: str,
    passed: bool,
    repeat: int = 0,
    text: str = "ok",
    tags: list[str] | None = None,
    check: str = "contains",
    latency_ms: float = 1000.0,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        repeat_index=repeat,
        tags=tags or [],
        prompt="p",
        response=Response(text=text, latency_ms=latency_ms, input_tokens=10, output_tokens=2),
        checks=[CheckResult(check=check, passed=passed, detail="" if passed else "nope")],
    )


# Saves the results as a completed run and returns its id.
def saved_run(store: Store, results: list[CaseResult], document_format: str = "plain") -> int:
    run_id = store.start_run(
        "demo", "demo.yaml", "mock", "url", {"temperature": 0.7}, document_format
    )
    for r in results:
        store.save_case(run_id, r)
    store.finish_run(run_id, "completed")
    return run_id


def run_row(store: Store, run_id: int) -> sqlite3.Row:
    row = store.get_run(run_id)
    assert row is not None
    return row


def test_counting() -> None:
    results = [
        answer("a", True, tags=["x", "y"]),
        answer("a", False, repeat=1, tags=["x", "y"]),
        answer("b", True, tags=["x"]),
    ]
    assert by_case(results) == {"a": (1, 2), "b": (1, 1)}
    assert by_tag(results) == {"x": (2, 3), "y": (1, 2)}


def test_attack_success_counts_failed_injection_checks_only() -> None:
    results = [
        answer("leak", False, tags=["exfiltration"], check="injection"),
        answer("leak", True, repeat=1, tags=["exfiltration"], check="injection"),
        answer("task", False, tags=["exfiltration"]),  # a task check, not an attack
    ]
    assert attacks_by_tag(results) == {"exfiltration": (1, 2)}


def test_format_rate() -> None:
    assert format_rate(16, 22) == "16/22  73% [52%–87%]"


def test_report_sections(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    results = [answer("a", True, latency_ms=1000), answer("b", False, latency_ms=3000)]
    run_id = saved_run(store, results)
    md = to_markdown("Run", build_report(run_row(store, run_id), results))
    assert "| all answers | 1/2 | 50% | 9%–91% |" in md
    assert "## By check type" in md and "## By tag" in md
    assert "| latency p50 | 2.0 s |" in md
    assert "## Attack success by tag" not in md  # no injection checks in this run
    assert "### b" in md and "contains: nope" in md


def test_report_attack_section(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    results = [answer("leak", False, tags=["exfiltration"], check="injection")]
    md = to_markdown("Run", build_report(run_row(store, saved_run(store, results)), results))
    assert "## Attack success by tag" in md
    assert "| exfiltration | 1/1 | 100% | 21%–100% |" in md
    assert "| all attacks | 1/1 |" in md


def test_identical_failed_answers_shown_once(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    results = [answer("a", False, repeat=i, text="Sydney") for i in range(3)]
    results.append(answer("a", False, repeat=3, text="Canberra?"))
    md = to_markdown("Run", build_report(run_row(store, saved_run(store, results)), results))
    assert "Failed 4 of 4 answers." in md
    assert "Answer (×3):" in md and "Answer (×1):" in md
    assert md.count("Sydney") == 1


def test_html_escapes_model_output(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    results = [answer("a", False, text="<script>alert(1)</script>")]
    page = to_html("Run", build_report(run_row(store, saved_run(store, results)), results))
    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_markdown_fences_answers_even_with_backticks(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    text = "```\n![x](https://canary-x1b7.example/leak)"
    results = [answer("a", False, text=text)]
    md = to_markdown("Run", build_report(run_row(store, saved_run(store, results)), results))
    # A 4-backtick fence, so the answer's own ``` can't close it.
    assert f"````\n{text}\n````" in md


def test_compare_shows_attack_success(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    a = [answer("leak", False, i, tags=["exfiltration"], check="injection") for i in range(5)]
    b = [answer("leak", True, i, tags=["exfiltration"], check="injection") for i in range(5)]
    out = compare_runs(
        run_row(store, saved_run(store, a)), run_row(store, saved_run(store, b)), a, b
    )
    assert "attack success by tag (lower is better)" in out
    assert "exfiltration" in out
    assert "all attacks" in out and "5/5  100% [57%–100%]    →  0/5  0% [0%–43%]" in out


def test_compare_separates_significant_from_noise(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    # "flip" goes 0/5 -> 5/5 (significant); "wobble" 1/5 -> 4/5 (noise); "steady" stays 5/5.
    a = [answer("flip", False, i) for i in range(5)]
    a += [answer("wobble", i == 0, i) for i in range(5)]
    a += [answer("steady", True, i) for i in range(5)]
    b = [answer("flip", True, i) for i in range(5)]
    b += [answer("wobble", i != 0, i) for i in range(5)]
    b += [answer("steady", True, i) for i in range(5)]
    run_a = run_row(store, saved_run(store, a))
    run_b = run_row(store, saved_run(store, b, document_format="delimited"))

    out = compare_runs(run_a, run_b, a, b)
    assert "A: run #1  mock  plain  temperature 0.7  5 repeats  (completed)" in out
    assert "B: run #2  mock  delimited" in out
    assert "cases: 1 significant, 1 noise, 1 unchanged" in out
    significant_part, noise_part = out.split(" noise:")
    assert "flip" in significant_part.split(" significant:")[1]
    assert "up, significant" in significant_part
    assert "wobble" in noise_part and "up, noise" in noise_part
