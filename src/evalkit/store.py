# Saves runs and their results in a SQLite file. Schema documented in docs/schema.md.

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalkit.models import CaseResult, CheckResult, Response, ToolCall

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE runs (
    id          INTEGER PRIMARY KEY,
    suite_name  TEXT NOT NULL,
    suite_path  TEXT NOT NULL,
    model       TEXT NOT NULL,
    base_url    TEXT NOT NULL,
    params      TEXT NOT NULL,              -- JSON
    document_format TEXT NOT NULL DEFAULT 'plain',  -- 'plain' or 'delimited'
    judge       TEXT,                       -- model that graded rubric checks (v3)
    status      TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    error       TEXT,
    started_at  TEXT NOT NULL,              -- ISO-8601, UTC
    finished_at TEXT
);

CREATE TABLE case_results (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs (id),
    case_id       TEXT NOT NULL,
    repeat_index  INTEGER NOT NULL DEFAULT 0,
    tags          TEXT NOT NULL,            -- JSON list
    prompt        TEXT NOT NULL,
    answer        TEXT NOT NULL,
    tool_calls    TEXT NOT NULL,            -- JSON list
    passed        INTEGER NOT NULL CHECK (passed IN (0, 1)),
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms    REAL NOT NULL,
    cost_usd      REAL NOT NULL,
    UNIQUE (run_id, case_id, repeat_index)
);

CREATE TABLE check_results (
    id             INTEGER PRIMARY KEY,
    case_result_id INTEGER NOT NULL REFERENCES case_results (id),
    position       INTEGER NOT NULL,        -- order of the check within its case
    type           TEXT NOT NULL,
    passed         INTEGER NOT NULL CHECK (passed IN (0, 1)),
    detail         TEXT NOT NULL,
    config         TEXT NOT NULL,           -- JSON: the check's settings
    score          INTEGER                  -- judge's 1-5 score, rubric checks only (v3)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # SQLite ignores REFERENCES unless this is switched on, per connection.
        self.conn.execute("PRAGMA foreign_keys = ON")
        # user_version is a free integer SQLite keeps in the file: 0 means brand new.
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"{path} was created by a newer evalkit (schema v{version})")
        with self.conn:
            if version == 0:
                self.conn.executescript(SCHEMA)
            # Migrations: upgrade older files one version at a time, keeping their data.
            if version == 1:  # v1 -> v2
                self.conn.execute(
                    "ALTER TABLE runs ADD COLUMN document_format TEXT NOT NULL DEFAULT 'plain'"
                )
            if version in (1, 2):  # v2 -> v3
                self.conn.execute("ALTER TABLE runs ADD COLUMN judge TEXT")
                self.conn.execute("ALTER TABLE check_results ADD COLUMN score INTEGER")
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def start_run(
        self,
        suite_name: str,
        suite_path: str,
        model: str,
        base_url: str,
        params: dict[str, Any],
        document_format: str = "plain",
        judge: str | None = None,
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO runs (suite_name, suite_path, model, base_url, params,"
                " document_format, judge, status, started_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)",
                (
                    suite_name,
                    suite_path,
                    model,
                    base_url,
                    json.dumps(params),
                    document_format,
                    judge,
                    _now(),
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    # One transaction per case: if the run crashes, every case saved so far stays saved.
    def save_case(self, run_id: int, result: CaseResult) -> None:
        r = result.response
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO case_results (run_id, case_id, repeat_index, tags, prompt, answer,"
                " tool_calls, passed, input_tokens, output_tokens, latency_ms, cost_usd)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    result.case_id,
                    result.repeat_index,
                    json.dumps(result.tags),
                    result.prompt,
                    r.text,
                    json.dumps([c.model_dump() for c in r.tool_calls]),
                    result.passed,
                    r.input_tokens,
                    r.output_tokens,
                    r.latency_ms,
                    r.cost_usd,
                ),
            )
            self.conn.executemany(
                "INSERT INTO check_results (case_result_id, position, type, passed, detail,"
                " config, score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        cursor.lastrowid,
                        i,
                        c.check,
                        c.passed,
                        c.detail,
                        json.dumps(c.config),
                        c.score,
                    )
                    for i, c in enumerate(result.checks)
                ],
            )

    def finish_run(self, run_id: int, status: str, error: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, _now(), run_id),
            )

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return row

    # Rebuilds the CaseResult objects the runner saved, so reports work on the same models.
    def load_results(self, run_id: int) -> list[CaseResult]:
        cases = self.conn.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY repeat_index, id", (run_id,)
        ).fetchall()
        results = []
        for c in cases:
            checks = self.conn.execute(
                "SELECT * FROM check_results WHERE case_result_id = ? ORDER BY position",
                (c["id"],),
            ).fetchall()
            response = Response(
                text=c["answer"],
                tool_calls=[ToolCall(**t) for t in json.loads(c["tool_calls"])],
                input_tokens=c["input_tokens"],
                output_tokens=c["output_tokens"],
                latency_ms=c["latency_ms"],
                cost_usd=c["cost_usd"],
            )
            results.append(
                CaseResult(
                    case_id=c["case_id"],
                    repeat_index=c["repeat_index"],
                    tags=json.loads(c["tags"]),
                    prompt=c["prompt"],
                    response=response,
                    checks=[
                        CheckResult(
                            check=k["type"],
                            passed=bool(k["passed"]),
                            detail=k["detail"],
                            config=json.loads(k["config"]),
                            score=k["score"],
                        )
                        for k in checks
                    ],
                )
            )
        return results
