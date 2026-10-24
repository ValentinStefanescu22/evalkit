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
        {"name": "get_weather", "arguments": {"city": "Bucharest"}}
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
