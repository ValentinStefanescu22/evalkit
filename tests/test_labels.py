import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalkit.labels import (
    LabelError,
    agreement,
    export_labels,
    load_labels,
    reference_agreement,
)
from evalkit.models import CaseResult, CheckResult, Response
from evalkit.store import Store


# One answer to a rubric case, as the runner saves it. judge=None = an unusable judge reply.
def judged(case_id: str, repeat: int, judge: int | None, text: str = "An answer.") -> CaseResult:
    passed = judge is not None and judge >= 4
    detail = f"score {judge}/5 (needs 4): Reason {case_id}." if judge else "judge reply not usable"
    return CaseResult(
        case_id=case_id,
        repeat_index=repeat,
        prompt=f"Question {case_id}?",
        response=Response(text=text),
        checks=[
            CheckResult(check="contains", passed=True),
            CheckResult(
                check="rubric",
                passed=passed,
                detail=detail,
                config={"type": "rubric", "rubric": "Be right.", "min_score": 4},
                score=judge,
            ),
        ],
    )


def saved(tmp_path: Path, results: list[CaseResult]) -> tuple[sqlite3.Row, list[CaseResult]]:
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("open_answers", "s.yaml", "qwen", "u", {}, judge="llama")
    for r in results:
        store.save_case(run_id, r)
    run = store.get_run(run_id)
    assert run is not None
    return run, store.load_results(run_id)


def write_labels(tmp_path: Path, answers: list[dict[str, Any]], run_id: int = 1) -> Path:
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump({"run_id": run_id, "answers": answers}))
    return path


def test_export_is_blind_and_readable(tmp_path: Path) -> None:
    run, results = saved(tmp_path, [judged("a", 0, 5, text="Line one.\nLine two.")])
    text, count = export_labels(run, results, {"a": "Question a?\n\nThe document."})
    assert count == 1
    assert text.startswith("# Hand labels for run #1: open_answers, qwen.")
    # Multi-line text as | blocks; the fields to fill in are empty, not "null".
    assert "  answer: |\n    Line one.\n    Line two.\n" in text
    assert "  question: |\n    Question a?\n\n    The document.\n" in text
    assert "  pass:\n  score:\n" in text
    # Blind: nothing from the judge.
    assert "Reason" not in text and "5/5" not in text
    # And it reads back as an unlabelled file.
    assert yaml.safe_load(text)["answers"][0]["pass"] is None


def test_export_needs_rubric_answers(tmp_path: Path) -> None:
    plain = CaseResult(case_id="x", prompt="p", response=Response(), checks=[])
    run, results = saved(tmp_path, [plain])
    with pytest.raises(LabelError, match="no rubric checks to label"):
        export_labels(run, results, {})


def test_load_reads_pass_alias_and_allows_unlabelled(tmp_path: Path) -> None:
    path = write_labels(
        tmp_path,
        [
            {"case_id": "a", "repeat": 0, "pass": True, "score": 4, "note": "fine"},
            {"case_id": "a", "repeat": 1, "pass": None, "score": None},
        ],
    )
    labels = load_labels(path)
    assert [(lab.passed, lab.score, lab.labelled) for lab in labels.answers] == [
        (True, 4, True),
        (None, None, False),
    ]


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        ({"pass": True}, "answers[0] (a, repeat 0): fill in both pass and score, or neither"),
        ({"pass": True, "score": 6}, "answers[0] (a, repeat 0).score: Input should be less"),
        ({"pass": "maybe", "score": 3}, "answers[0] (a, repeat 0).pass: Input should be a valid"),
        ({"scroe": 3}, "answers[0] (a, repeat 0).scroe: Extra inputs are not permitted"),
    ],
)
def test_load_errors_name_the_answer(tmp_path: Path, entry: dict[str, Any], problem: str) -> None:
    path = write_labels(tmp_path, [{"case_id": "a", "repeat": 0, **entry}])
    with pytest.raises(LabelError) as e:
        load_labels(path)
    assert problem in str(e.value)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="cannot read file"):
        load_labels(tmp_path / "none.yaml")


def label(case_id: str, repeat: int, passed: bool | None, score: int | None) -> dict[str, Any]:
    return {"case_id": case_id, "repeat": repeat, "pass": passed, "score": score}


def test_agreement_numbers(tmp_path: Path) -> None:
    # Judge scores 5,4,4,2 (pass,pass,pass,fail); you say 5,2,4,1 (pass,fail,pass,fail).
    long_answer = "A much longer answer. " * 10
    run, results = saved(
        tmp_path,
        [
            judged("a", 0, 5),
            judged("b", 0, 4, text=long_answer),
            judged("c", 0, 4),
            judged("d", 0, 2),
            judged("e", 0, None),  # judge reply unusable
            judged("f", 0, 4),  # not labelled
        ],
    )
    labels = load_labels(
        write_labels(
            tmp_path,
            [
                label("a", 0, True, 5),
                label("b", 0, False, 2),
                label("c", 0, True, 4),
                label("d", 0, False, 1),
                label("e", 0, True, 4),
                label("f", 0, None, None),
            ],
        )
    )
    out = agreement(run, labels, results, "mine.yaml")
    assert "run #1 (open_answers, qwen)" in out
    assert "labelled 5 of 6 answers   (left out, judge reply unusable: 1)" in out
    assert "A = judge llama\nB = mine.yaml" in out
    # pass/fail: 3/4 agree. Judge passes 3/4, you 2/4: chance = .75*.5 + .25*.5 = .5 -> kappa .5
    assert "pass/fail   agreement 3/4 75%   kappa 0.50 (moderate)" in out
    assert "A pass, B fail: 1   A fail, B pass: 0" in out
    # scores: exact 2/4 (a, c); within 1: a, c, d = 3/4; means 3.75 vs 3.0
    assert "exact 2/4 50%   within 1 point 3/4 75%" in out
    assert "mean score: A 3.8, B 3.0 (A +0.8)" in out
    # Judge scored higher on b (long) and d: avg of 220 and 10 chars = 115.
    assert "answers A scored higher than B: 2, avg 115 chars; the rest: avg 10 chars" in out
    assert "b  repeat 0   A: pass 4   B: fail 2   'Reason b.'" in out


def test_agreement_with_nothing_labelled(tmp_path: Path) -> None:
    run, results = saved(tmp_path, [judged("a", 0, 4)])
    labels = load_labels(write_labels(tmp_path, [label("a", 0, None, None)]))
    out = agreement(run, labels, results, "mine.yaml")
    assert "labelled 0 of 1 answers" in out
    assert "nothing to compare yet" in out


def test_agreement_rejects_unknown_answers(tmp_path: Path) -> None:
    run, results = saved(tmp_path, [judged("a", 0, 4)])
    labels = load_labels(write_labels(tmp_path, [label("a", 3, True, 4)]))
    with pytest.raises(LabelError, match="no judged answer a, repeat 3"):
        agreement(run, labels, results, "mine.yaml")


# Two labellers (e.g. Claude as A, you as B), compared only where both have graded.
def test_reference_agreement(tmp_path: Path) -> None:
    run, results = saved(tmp_path, [judged("a", 0, 4), judged("b", 0, 4), judged("c", 0, 4)])
    mine = write_labels(
        tmp_path, [label("a", 0, True, 4), label("b", 0, False, 2), label("c", 0, None, None)]
    )
    other = tmp_path / "other.yaml"
    other.write_text(
        yaml.safe_dump(
            {
                "run_id": 1,
                "answers": [
                    {**label("a", 0, True, 5), "note": "great"},
                    {**label("b", 0, True, 4), "note": "fine"},
                    label("c", 0, True, 4),
                ],
            }
        )
    )
    out = reference_agreement(run, load_labels(other), load_labels(mine), results, "claude", "you")
    assert "graded in both files: 2 answers" in out
    assert "A = claude\nB = you" in out
    assert "pass/fail   agreement 1/2 50%" in out
    assert "b  repeat 0   A: pass 4   B: fail 2   'fine'" in out


def test_reference_must_label_the_same_run(tmp_path: Path) -> None:
    run, results = saved(tmp_path, [judged("a", 0, 4)])
    mine = load_labels(write_labels(tmp_path, [label("a", 0, True, 4)]))
    other = mine.model_copy(update={"run_id": 2})
    with pytest.raises(LabelError, match="different runs"):
        reference_agreement(run, other, mine, results, "claude", "you")


# Labels made for one set of answers must not be compared with different answers.
def test_agreement_refuses_labels_for_other_answers(tmp_path: Path) -> None:
    run, results = saved(tmp_path, [judged("a", 0, 4, text="Bucharest.")])
    same = {**label("a", 0, True, 4), "answer": "Bucharest.\n"}
    other = {**label("a", 0, True, 4), "answer": "Paris."}
    assert "agreement 1/1" in agreement(
        run, load_labels(write_labels(tmp_path, [same])), results, "x"
    )
    with pytest.raises(LabelError, match="made for different answers: a, repeat 0"):
        agreement(run, load_labels(write_labels(tmp_path, [other])), results, "x")
