# Turns saved results into numbers (pass rates with 95% intervals, latency percentiles),
# writes a run report as Markdown and HTML, and compares two runs.

import html
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from evalkit.models import CaseResult
from evalkit.stats import intervals_overlap, percentile, wilson_interval

# label -> (passed, total)
Counts = dict[str, tuple[int, int]]


# Adds up (label, passed) pairs per label, keeping labels in first-seen order.
def _count(pairs: Iterable[tuple[str, bool]]) -> Counts:
    counts: Counts = {}
    for label, passed in pairs:
        k, n = counts.get(label, (0, 0))
        counts[label] = (k + passed, n + 1)
    return counts


# Every answer is one try: 22 cases x 5 repeats = 110 tries.
def total(results: list[CaseResult]) -> tuple[int, int]:
    return sum(r.passed for r in results), len(results)


def by_case(results: list[CaseResult]) -> Counts:
    return _count((r.case_id, r.passed) for r in results)


def by_tag(results: list[CaseResult]) -> Counts:
    return _count((tag, r.passed) for r in results for tag in r.tags)


def by_check_type(results: list[CaseResult]) -> Counts:
    return _count((c.check, c.passed) for r in results for c in r.checks)


# One (tags, attack succeeded) entry per injection check. A failed check = the attack worked.
def _attacks(results: list[CaseResult]) -> list[tuple[list[str], bool]]:
    attacks = []
    for r in results:
        for c in r.checks:
            if c.check == "injection":
                attacks.append((r.tags, not c.passed))
    return attacks


def attack_total(results: list[CaseResult]) -> tuple[int, int]:
    attacks = _attacks(results)
    return sum(succeeded for _, succeeded in attacks), len(attacks)


def attacks_by_tag(results: list[CaseResult]) -> Counts:
    return _count((tag, succeeded) for tags, succeeded in _attacks(results) for tag in tags)


def repeat_count(results: list[CaseResult]) -> int:
    return max((r.repeat_index for r in results), default=0) + 1


# Significant only when the two 95% intervals don't overlap (the rule agreed in SPEC.md).
def significant(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not intervals_overlap(wilson_interval(*a), wilson_interval(*b))


# "16/22  73% [52%–87%]"
def format_rate(k: int, n: int) -> str:
    low, high = wilson_interval(k, n)
    return f"{k}/{n}  {k / n:.0%} [{low:.0%}–{high:.0%}]"


def _rate_row(label: str, k: int, n: int) -> list[str]:
    low, high = wilson_interval(k, n)
    return [label, f"{k}/{n}", f"{k / n:.0%}", f"{low:.0%}–{high:.0%}"]


RATE_HEADERS = ["", "passed", "rate", "95% CI"]


# A report is a list of these blocks; to_markdown and to_html each know how to draw them.
@dataclass
class Heading:
    text: str
    level: int = 2


@dataclass
class Text:
    text: str


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]


# Shown exactly as written, never interpreted. Used for anything the model produced.
@dataclass
class Code:
    text: str


Block = Heading | Text | Table | Code


def _rate_table(counts: Counts) -> Table:
    return Table(RATE_HEADERS, [_rate_row(label, k, n) for label, (k, n) in counts.items()])


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s"


# What one failed answer looked like: its text, plus any tool calls it made.
def _shown_answer(result: CaseResult) -> str:
    text = result.response.text or "(empty answer)"
    if result.response.tool_calls:
        calls = json.dumps([c.model_dump(exclude={"id"}) for c in result.response.tool_calls])
        text += f"\n\ntool calls: {calls}"
    return text


def build_report(run: sqlite3.Row, results: list[CaseResult]) -> list[Block]:
    k, n = total(results)
    blocks: list[Block] = [
        Table(
            ["setting", "value"],
            [
                ["suite", f"{run['suite_name']} ({run['suite_path']})"],
                ["model", f"{run['model']} at {run['base_url']}"],
                ["document format", run["document_format"]],
                ["params", run["params"]],
                ["status", run["status"] + (f": {run['error']}" if run["error"] else "")],
                ["started", run["started_at"]],
                ["repeats", str(repeat_count(results))],
            ],
        ),
        Heading("Pass rate"),
        Table(RATE_HEADERS, [_rate_row("all answers", k, n)]),
        Text(
            "Each answer counts as one try. Answers to the same case are related "
            "(a hard case is hard every time), so this interval is a little too narrow."
        ),
        Heading("By check type"),
        _rate_table(by_check_type(results)),
        Heading("By tag"),
        _rate_table(by_tag(results)),
    ]

    succeeded, attacks = attack_total(results)
    if attacks:
        table = _rate_table(attacks_by_tag(results))
        table.headers = ["", "succeeded", "rate", "95% CI"]
        table.rows.append(_rate_row("all attacks", succeeded, attacks))
        blocks += [
            Heading("Attack success by tag"),
            Text("An attack succeeded when its injection check failed. Lower is better."),
            table,
        ]

    latencies = [r.response.latency_ms for r in results]
    blocks += [
        Heading("Latency, tokens, cost"),
        Table(
            ["metric", "value"],
            [
                ["latency p50", _seconds(percentile(latencies, 50))],
                ["latency p95", _seconds(percentile(latencies, 95))],
                ["input tokens", str(sum(r.response.input_tokens for r in results))],
                ["output tokens", str(sum(r.response.output_tokens for r in results))],
                ["cost", f"${sum(r.response.cost_usd for r in results):.4f}"],
            ],
        ),
        Heading("Cases"),
        _rate_table(by_case(results)),
        Heading("Failed answers"),
    ]

    answers_per_case = {case: n for case, (_, n) in by_case(results).items()}
    failed = [r for r in results if not r.passed]
    if not failed:
        blocks.append(Text("None."))
    for case_id, (_, case_n) in by_case(failed).items():
        mine = [r for r in failed if r.case_id == case_id]
        # dict.fromkeys drops duplicates but keeps the first-seen order.
        reasons = dict.fromkeys(
            f"{c.check}: {c.detail}" for r in mine for c in r.checks if not c.passed
        )
        blocks += [
            Heading(case_id, level=3),
            Text(f"Failed {case_n} of {answers_per_case[case_id]} answers."),
            Code("\n".join(reasons)),
        ]
        # At temperature 0 the repeats usually say the same thing: show each answer once.
        for answer, count in Counter(_shown_answer(r) for r in mine).items():
            blocks += [Text(f"Answer (×{count}):"), Code(answer)]
    return blocks


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(_md_cell(c) for c in cells) + " |"


def to_markdown(title: str, blocks: list[Block]) -> str:
    lines = [f"# {title}", ""]
    for block in blocks:
        if isinstance(block, Heading):
            lines += ["#" * block.level + " " + block.text, ""]
        elif isinstance(block, Text):
            lines += [block.text, ""]
        elif isinstance(block, Table):
            lines.append(_md_row(block.headers))
            lines.append(_md_row(["---"] * len(block.headers)))
            lines += [_md_row(row) for row in block.rows]
            lines.append("")
        else:
            # The fence must be longer than any run of backticks inside, or the answer
            # could close it early and the rest would be rendered (images fetched, etc.).
            longest = max((len(run) for run in re.findall(r"`+", block.text)), default=0)
            fence = "`" * max(3, longest + 1)
            lines += [fence, block.text, fence, ""]
    return "\n".join(lines)


CSS = """
body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem;
       color-scheme: light dark; }
table { border-collapse: collapse; margin-bottom: 1rem; }
th, td { border: 1px solid #8884; padding: 0.25rem 0.6rem; text-align: left; }
pre { white-space: pre-wrap; background: #8881; padding: 0.6rem; border-radius: 4px; }
"""


# Every value goes through html.escape: model answers are untrusted, and an answer
# containing <script> must show up as text, not run in the browser.
def to_html(title: str, blocks: list[Block]) -> str:
    e = html.escape
    parts = [
        "<!doctype html>",
        f'<html><head><meta charset="utf-8"><title>{e(title)}</title>',
        f"<style>{CSS}</style></head><body>",
        f"<h1>{e(title)}</h1>",
    ]
    for block in blocks:
        if isinstance(block, Heading):
            parts.append(f"<h{block.level}>{e(block.text)}</h{block.level}>")
        elif isinstance(block, Text):
            parts.append(f"<p>{e(block.text)}</p>")
        elif isinstance(block, Table):
            parts.append("<table>")
            parts.append("<tr>" + "".join(f"<th>{e(h)}</th>" for h in block.headers) + "</tr>")
            for row in block.rows:
                parts.append("<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>")
            parts.append("</table>")
        else:
            parts.append(f"<pre>{e(block.text)}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _describe(run: sqlite3.Row, results: list[CaseResult]) -> str:
    temperature = json.loads(run["params"]).get("temperature", "default")
    return (
        f"run #{run['id']}  {run['model']}  {run['document_format']}  "
        f"temperature {temperature}  {repeat_count(results)} repeats  ({run['status']})"
    )


def _change(label: str, width: int, a: tuple[int, int], b: tuple[int, int]) -> str:
    if a[0] / a[1] == b[0] / b[1]:
        verdict = "same"
    else:
        direction = "up" if b[0] / b[1] > a[0] / a[1] else "down"
        verdict = f"{direction}, {'significant' if significant(a, b) else 'noise'}"
    return f"  {label:<{width}}  {format_rate(*a):<22}  →  {format_rate(*b):<22}  {verdict}"


# Plain-text comparison of run A (before) and run B (after), for the terminal.
def compare_runs(
    run_a: sqlite3.Row,
    run_b: sqlite3.Row,
    results_a: list[CaseResult],
    results_b: list[CaseResult],
) -> str:
    lines = []
    if run_a["suite_name"] != run_b["suite_name"]:
        lines.append(f"warning: different suites ({run_a['suite_name']} vs {run_b['suite_name']})")
    lines += [f"A: {_describe(run_a, results_a)}", f"B: {_describe(run_b, results_b)}", ""]

    cases_a, cases_b = by_case(results_a), by_case(results_b)
    width = max(len(label) for label in [*cases_a, *cases_b, "all attacks"])
    lines.append(_change("pass rate", width, total(results_a), total(results_b)))

    attacks_a, attacks_b = attacks_by_tag(results_a), attacks_by_tag(results_b)
    shared_tags = [tag for tag in attacks_a if tag in attacks_b]
    if shared_tags:
        lines += ["", "attack success by tag (lower is better)"]
        lines += [_change(tag, width, attacks_a[tag], attacks_b[tag]) for tag in shared_tags]
        lines.append(
            _change("all attacks", width, attack_total(results_a), attack_total(results_b))
        )

    shared = [case for case in cases_a if case in cases_b]
    changed = [c for c in shared if cases_a[c][0] / cases_a[c][1] != cases_b[c][0] / cases_b[c][1]]
    real = [c for c in changed if significant(cases_a[c], cases_b[c])]
    noise = [c for c in changed if c not in real]
    lines += [
        "",
        f"cases: {len(real)} significant, {len(noise)} noise, "
        f"{len(shared) - len(changed)} unchanged",
    ]
    for title, group in [("significant", real), ("noise", noise)]:
        if group:
            lines.append(f" {title}:")
            lines += [_change(c, width, cases_a[c], cases_b[c]) for c in group]

    only_one = len(set(cases_a) ^ set(cases_b))
    if only_one:
        lines += ["", f"{only_one} cases appear in only one run and were skipped"]
    return "\n".join(lines)
