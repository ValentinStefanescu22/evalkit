# Hand labels for checking the LLM judge: export a run's answers to label (blind: no judge
# scores in the file), read your labels back, and measure how well the judge agrees with you.

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, model_validator

from evalkit.models import CaseResult, CheckResult, StrictModel
from evalkit.stats import cohens_kappa, kappa_label
from evalkit.suite import SuiteError, read_yaml_mapping


class LabelError(Exception):
    pass


# The rubric check of an answer, or None if it has none. More than one would be ambiguous.
def _rubric(result: CaseResult) -> CheckResult | None:
    rubrics = [c for c in result.checks if c.check == "rubric"]
    if len(rubrics) > 1:
        raise LabelError(f"{result.case_id} has {len(rubrics)} rubric checks; can't label it")
    return rubrics[0] if rubrics else None


# A YAML writer that shows multi-line text as | blocks (readable answers) and empty values
# as nothing at all (`pass:` instead of `pass: null`), so the fields to fill in are obvious.
class _LabelDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, text: str) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", text, style="|" if "\n" in text else None
    )


def _represent_none(dumper: yaml.SafeDumper, _: None) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


_LabelDumper.add_representer(str, _represent_str)
_LabelDumper.add_representer(type(None), _represent_none)


# A trailing newline makes even one-line text a | block, so every entry looks the same.
def _block(text: str) -> str:
    return text.rstrip("\n") + "\n"


# The labels file for a run, and how many answers are in it. `questions` maps case id -> what
# the judge was asked (prompt plus document); missing cases fall back to the stored prompt.
def export_labels(
    run: sqlite3.Row, results: list[CaseResult], questions: dict[str, str]
) -> tuple[str, int]:
    answers = []
    for r in results:
        rubric = _rubric(r)
        if rubric is None:
            continue
        answers.append(
            {
                "case_id": r.case_id,
                "repeat": r.repeat_index,
                "question": _block(questions.get(r.case_id, r.prompt)),
                "rubric": _block(rubric.config["rubric"]),
                "answer": _block(r.response.text or "(empty answer)"),
                "pass": None,
                "score": None,
                "note": "",
            }
        )
    if not answers:
        raise LabelError(f"run #{run['id']} has no rubric checks to label")
    header = (
        f"# Hand labels for run #{run['id']}: {run['suite_name']}, {run['model']}."
        " Judge scores are hidden on purpose.\n"
        "# For each answer fill in pass (true/false) and score (1-5); note is optional.\n"
    )
    body = yaml.dump(
        {"run_id": run["id"], "answers": answers},
        Dumper=_LabelDumper,
        sort_keys=False,
        allow_unicode=True,
    )
    return header + body, len(answers)


# One labelled (or not yet labelled) answer. `pass` is a Python keyword, hence the alias.
class Label(StrictModel):
    case_id: str
    repeat: int
    question: str = ""
    rubric: str = ""
    answer: str = ""
    passed: bool | None = Field(default=None, alias="pass")
    score: int | None = Field(default=None, ge=1, le=5)
    note: str | None = None

    @model_validator(mode="after")
    def both_or_neither(self) -> "Label":
        if (self.passed is None) != (self.score is None):
            raise ValueError("fill in both pass and score, or neither")
        return self

    @property
    def labelled(self) -> bool:
        return self.passed is not None


class LabelFile(StrictModel):
    run_id: int
    answers: list[Label]


def load_labels(path: Path) -> LabelFile:
    try:
        data = read_yaml_mapping(path, "`run_id` and `answers`")
    except SuiteError as e:
        raise LabelError(str(e)) from e
    try:
        return LabelFile.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = err["loc"]
            where = ".".join(str(part) for part in loc)
            # Name the answer, e.g. "answers[3] (explain_padlock, repeat 1)".
            if len(loc) >= 2 and loc[0] == "answers" and isinstance(loc[1], int):
                entry: Any = data["answers"][loc[1]]
                if isinstance(entry, dict):
                    name = f"{entry.get('case_id')}, repeat {entry.get('repeat')}"
                    where = f"answers[{loc[1]}] ({name})" + "".join(f".{p}" for p in loc[2:])
            lines.append(f"  - {where}: {err['msg'].removeprefix('Value error, ')}")
        raise LabelError(f"{path}: {len(lines)} problem(s)\n" + "\n".join(lines)) from e


# One answer graded by two raters: A (the judge, or a reference labels file) and B (a labels
# file). Plain values: only answers both have graded get here.
@dataclass
class Pair:
    case_id: str
    repeat: int
    a_pass: bool
    a_score: int
    a_why: str  # the judge's reasoning, or the reference labeller's note
    b_pass: bool
    b_score: int
    answer: str


def _verdict(passed: bool, score: int) -> str:
    return f"{'pass' if passed else 'fail'} {score}"


def _kappa(pairs: list[tuple[Any, Any]]) -> str:
    kappa = cohens_kappa(pairs)
    shown = "undefined" if kappa is None else f"{kappa:.2f}"
    return f"kappa {shown} ({kappa_label(kappa)})"


def _avg_length(pairs: list[Pair]) -> str:
    if not pairs:
        return "-"
    return f"{sum(len(p.answer) for p in pairs) / len(pairs):.0f} chars"


# Pairs each labelled answer (B) with the judge's grade (A). Also returns how many labelled
# answers were left out because the judge's reply was unusable (no score).
def match_judge(
    labels: LabelFile, results: list[CaseResult], run_id: int
) -> tuple[list[Pair], int]:
    by_key = {(r.case_id, r.repeat_index): r for r in results}
    pairs = []
    unusable = 0
    for label in labels.answers:
        result = by_key.get((label.case_id, label.repeat))
        rubric = _rubric(result) if result else None
        if result is None or rubric is None:
            raise LabelError(
                f"run #{run_id} has no judged answer {label.case_id}, repeat {label.repeat}"
            )
        if label.passed is None or label.score is None:
            continue  # not labelled yet
        if rubric.score is None:
            unusable += 1
            continue
        pairs.append(
            Pair(
                case_id=label.case_id,
                repeat=label.repeat,
                a_pass=rubric.passed,
                a_score=rubric.score,
                # The judge's detail is "score 4/5 (needs 4): <reasoning>"; keep the reasoning.
                a_why=rubric.detail.split(": ", 1)[-1],
                b_pass=label.passed,
                b_score=label.score,
                answer=result.response.text,
            )
        )
    return pairs, unusable


# Pairs answers labelled in both files: reference (A) and labels (B), e.g. Claude vs you.
def match_labels(reference: LabelFile, labels: LabelFile, results: list[CaseResult]) -> list[Pair]:
    if reference.run_id != labels.run_id:
        raise LabelError(f"the files label different runs (#{reference.run_id}, #{labels.run_id})")
    answers = {(r.case_id, r.repeat_index): r.response.text for r in results}
    other = {(lab.case_id, lab.repeat): lab for lab in reference.answers}
    pairs = []
    for label in labels.answers:
        ref = other.get((label.case_id, label.repeat))
        if label.passed is None or label.score is None or ref is None:
            continue
        if ref.passed is None or ref.score is None:
            continue
        pairs.append(
            Pair(
                case_id=label.case_id,
                repeat=label.repeat,
                a_pass=ref.passed,
                a_score=ref.score,
                a_why=ref.note or "",
                b_pass=label.passed,
                b_score=label.score,
                answer=answers.get((label.case_id, label.repeat), ""),
            )
        )
    return pairs


# The comparison as text for the terminal. `a` and `b` say who the two raters are.
def agreement_text(header: list[str], pairs: list[Pair], a: str, b: str) -> str:
    lines = [*header, f"A = {a}", f"B = {b}"]
    if not pairs:
        lines.append("nothing to compare yet: no answer is graded by both")
        return "\n".join(lines)

    n = len(pairs)
    passes = [(p.a_pass, p.b_pass) for p in pairs]
    scores = [(p.a_score, p.b_score) for p in pairs]
    agree = sum(p.a_pass == p.b_pass for p in pairs)
    exact = sum(p.a_score == p.b_score for p in pairs)
    within_one = sum(abs(p.a_score - p.b_score) <= 1 for p in pairs)
    a_mean = sum(p.a_score for p in pairs) / n
    b_mean = sum(p.b_score for p in pairs) / n
    higher = [p for p in pairs if p.a_score > p.b_score]
    rest = [p for p in pairs if p.a_score <= p.b_score]

    lines += [
        "",
        f"pass/fail   agreement {agree}/{n} {agree / n:.0%}   {_kappa(passes)}",
        f"            A pass, B fail: {sum(p.a_pass and not p.b_pass for p in pairs)}"
        f"   A fail, B pass: {sum(p.b_pass and not p.a_pass for p in pairs)}",
        f"score 1-5   exact {exact}/{n} {exact / n:.0%}"
        f"   within 1 point {within_one}/{n} {within_one / n:.0%}   {_kappa(scores)}",
        f"            mean score: A {a_mean:.1f}, B {b_mean:.1f} (A {a_mean - b_mean:+.1f})",
        f"verbosity   answers A scored higher than B: {len(higher)},"
        f" avg {_avg_length(higher)}; the rest: avg {_avg_length(rest)}",
    ]

    disagreements = [p for p in pairs if p.a_pass != p.b_pass]
    if disagreements:
        lines += ["", "pass/fail disagreements:"]
        width = max(len(p.case_id) for p in disagreements)
        for p in disagreements:
            lines.append(
                f"  {p.case_id:<{width}}  repeat {p.repeat}"
                f"   A: {_verdict(p.a_pass, p.a_score)}"
                f"   B: {_verdict(p.b_pass, p.b_score)}"
                f"   {p.a_why[:80]!r}"
            )
    return "\n".join(lines)


# The judge (A) against a labels file (B).
def agreement(run: sqlite3.Row, labels: LabelFile, results: list[CaseResult], name: str) -> str:
    pairs, unusable = match_judge(labels, results, run["id"])
    labelled = sum(label.labelled for label in labels.answers)
    header = [
        f"run #{run['id']} ({run['suite_name']}, {run['model']})",
        f"labelled {labelled} of {len(labels.answers)} answers"
        + (f"   (left out, judge reply unusable: {unusable})" if unusable else ""),
    ]
    return agreement_text(header, pairs, f"judge {run['judge']}", name)


# A reference labels file (A) against a labels file (B), e.g. Claude's labels vs yours.
def reference_agreement(
    run: sqlite3.Row,
    reference: LabelFile,
    labels: LabelFile,
    results: list[CaseResult],
    reference_name: str,
    name: str,
) -> str:
    pairs = match_labels(reference, labels, results)
    header = [
        f"run #{run['id']} ({run['suite_name']}, {run['model']})",
        f"graded in both files: {len(pairs)} answers",
    ]
    return agreement_text(header, pairs, reference_name, name)
