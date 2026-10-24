# Saves runs and their results in a SQLite file. Schema documented in docs/schema.md.

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalkit.models import CaseResult

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE runs (
    id          INTEGER PRIMARY KEY,
    suite_name  TEXT NOT NULL,
    suite_path  TEXT NOT NULL,
    model       TEXT NOT NULL,
    base_url    TEXT NOT NULL,
    params      TEXT NOT NULL,              -- JSON
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
    config         TEXT NOT NULL            -- JSON: the check's settings
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
        if version == 0:
            with self.conn:
                self.conn.executescript(SCHEMA)
                self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif version > SCHEMA_VERSION:
            raise RuntimeError(f"{path} was created by a newer evalkit (schema v{version})")

    def close(self) -> None:
        self.conn.close()

    def start_run(
        self, suite_name: str, suite_path: str, model: str, base_url: str, params: dict[str, Any]
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO runs (suite_name, suite_path, model, base_url, params, status,"
                " started_at) VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (suite_name, suite_path, model, base_url, json.dumps(params), _now()),
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
                " config) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (cursor.lastrowid, i, c.check, c.passed, c.detail, json.dumps(c.config))
                    for i, c in enumerate(result.checks)
                ],
            )

    def finish_run(self, run_id: int, status: str, error: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, _now(), run_id),
            )
