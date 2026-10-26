import json
import sqlite3
from pathlib import Path

import pytest

from evalkit.models import CaseResult, CheckResult, Response, ToolCall
from evalkit.store import SCHEMA_VERSION, Store


def sample_result() -> CaseResult:
    return CaseResult(
        case_id="capital_romania",
        tags=["geography"],
        prompt="What is the capital of Romania?",
        response=Response(
            text="Bucharest.",
            tool_calls=[ToolCall(name="get_weather", arguments={"city": "Bucharest"})],
            input_tokens=12,
            output_tokens=3,
            latency_ms=812.5,
        ),
        checks=[
            CheckResult(check="contains", passed=True, config={"value": "Bucharest"}),
            CheckResult(check="regex", passed=False, detail="no match", config={"pattern": "x"}),
        ],
    )


def test_round_trip(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("basic_qa", "suites/basic_qa.yaml", "qwen2.5:3b", "url", {"t": 0})
    store.save_case(run_id, sample_result())
    store.finish_run(run_id, "completed")

    run = store.conn.execute("SELECT * FROM runs").fetchone()
    assert (run["id"], run["status"], run["model"]) == (run_id, "completed", "qwen2.5:3b")
    assert json.loads(run["params"]) == {"t": 0}
    assert run["finished_at"] is not None

    case = store.conn.execute("SELECT * FROM case_results").fetchone()
    assert (case["case_id"], case["answer"], case["passed"]) == ("capital_romania", "Bucharest.", 0)
    assert json.loads(case["tags"]) == ["geography"]
    assert json.loads(case["tool_calls"]) == [
        {"name": "get_weather", "arguments": {"city": "Bucharest"}, "id": ""}
    ]
    assert (case["input_tokens"], case["latency_ms"]) == (12, 812.5)

    checks = store.conn.execute("SELECT * FROM check_results ORDER BY position").fetchall()
    assert [(c["type"], c["passed"], c["detail"]) for c in checks] == [
        ("contains", 1, ""),
        ("regex", 0, "no match"),
    ]
    assert json.loads(checks[1]["config"]) == {"pattern": "x"}


def test_data_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "evalkit.db"
    store = Store(path)
    store.start_run("s", "p", "m", "u", {})
    store.close()
    assert Store(path).conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1


def test_schema_version_recorded(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_refuses_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "evalkit.db"
    sqlite3.connect(path).execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="newer evalkit"):
        Store(path)


def test_foreign_keys_enforced(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    with pytest.raises(sqlite3.IntegrityError):
        store.save_case(999, sample_result())


def test_same_case_twice_in_one_run_rejected(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("s", "p", "m", "u", {})
    store.save_case(run_id, sample_result())
    with pytest.raises(sqlite3.IntegrityError):
        store.save_case(run_id, sample_result())


# A database as milestone 4 created it: schema v1, runs table without document_format,
# check_results without score. One completed run with one graded case.
V1_DB = """
CREATE TABLE runs (
    id INTEGER PRIMARY KEY, suite_name TEXT NOT NULL, suite_path TEXT NOT NULL,
    model TEXT NOT NULL, base_url TEXT NOT NULL, params TEXT NOT NULL,
    status TEXT NOT NULL, error TEXT, started_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE case_results (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs (id),
    case_id TEXT NOT NULL, repeat_index INTEGER NOT NULL DEFAULT 0, tags TEXT NOT NULL,
    prompt TEXT NOT NULL, answer TEXT NOT NULL, tool_calls TEXT NOT NULL,
    passed INTEGER NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL, cost_usd REAL NOT NULL, UNIQUE (run_id, case_id, repeat_index)
);
CREATE TABLE check_results (
    id INTEGER PRIMARY KEY, case_result_id INTEGER NOT NULL REFERENCES case_results (id),
    position INTEGER NOT NULL, type TEXT NOT NULL, passed INTEGER NOT NULL,
    detail TEXT NOT NULL, config TEXT NOT NULL
);
INSERT INTO runs (suite_name, suite_path, model, base_url, params, status, started_at)
VALUES ('basic_qa', 'suites/basic_qa.yaml', 'qwen2.5:3b', 'url', '{}', 'completed', 'now');
INSERT INTO case_results (run_id, case_id, tags, prompt, answer, tool_calls, passed,
    input_tokens, output_tokens, latency_ms, cost_usd)
VALUES (1, 'capital_romania', '[]', 'p', 'Bucharest.', '[]', 1, 1, 1, 1.0, 0.0);
INSERT INTO check_results (case_result_id, position, type, passed, detail, config)
VALUES (1, 0, 'contains', 1, '', '{}');
PRAGMA user_version = 1;
"""

# The same file after milestone 5: v2 adds runs.document_format.
V2_DB = V1_DB.replace(
    "status TEXT NOT NULL, error",
    "document_format TEXT NOT NULL DEFAULT 'plain', status TEXT NOT NULL, error",
).replace("PRAGMA user_version = 1", "PRAGMA user_version = 2")


def open_old(tmp_path: Path, script: str) -> Store:
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(script)
    old.close()
    return Store(path)


@pytest.mark.parametrize("script", [V1_DB, V2_DB], ids=["v1", "v2"])
def test_migrates_old_database_keeping_old_runs(tmp_path: Path, script: str) -> None:
    store = open_old(tmp_path, script)
    assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    run = store.get_run(1)
    assert run is not None
    assert (run["suite_name"], run["document_format"], run["judge"]) == ("basic_qa", "plain", None)
    # Old results load, with no score (they were never judged).
    [result] = store.load_results(1)
    assert result.response.text == "Bucharest."
    assert result.checks[0].score is None


def test_document_format_saved(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    store.start_run("s", "p", "m", "u", {}, document_format="delimited")
    assert store.conn.execute("SELECT document_format FROM runs").fetchone()[0] == "delimited"


def test_load_results_rebuilds_what_was_saved(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("s", "p", "m", "u", {})
    saved = sample_result()
    second = sample_result()
    second.repeat_index = 1
    store.save_case(run_id, saved)
    store.save_case(run_id, second)
    assert store.load_results(run_id) == [saved, second]


def test_get_run(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("basic_qa", "p", "m", "u", {})
    run = store.get_run(run_id)
    assert run is not None and run["suite_name"] == "basic_qa"
    assert store.get_run(999) is None


def test_judge_and_score_saved(tmp_path: Path) -> None:
    store = Store(tmp_path / "evalkit.db")
    run_id = store.start_run("s", "p", "m", "u", {}, judge="llama3.2:3b")
    judged = sample_result()
    judged.checks.append(CheckResult(check="rubric", passed=True, detail="ok", score=4))
    store.save_case(run_id, judged)
    run = store.get_run(run_id)
    assert run is not None and run["judge"] == "llama3.2:3b"
    assert [c.score for c in store.load_results(run_id)[0].checks] == [None, None, 4]
