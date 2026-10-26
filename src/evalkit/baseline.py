# Baselines: a known-good run's pass rate, saved as a small JSON file that can be committed,
# so CI can fail when a new run is significantly worse.

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evalkit.models import CaseResult
from evalkit.report import significant, total


class Baseline(BaseModel):
    suite: str
    run_id: int
    model: str
    document_format: str
    params: dict[str, Any]
    passed: int
    total: int
    set_at: str  # ISO-8601, UTC


def default_path(suite_name: str) -> Path:
    return Path("baselines") / f"{suite_name}.json"


def make_baseline(run: sqlite3.Row, results: list[CaseResult]) -> Baseline:
    passed, n = total(results)
    return Baseline(
        suite=run["suite_name"],
        run_id=run["id"],
        model=run["model"],
        document_format=run["document_format"],
        params=json.loads(run["params"]),
        passed=passed,
        total=n,
        set_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def save_baseline(path: Path, baseline: Baseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")


# Raises OSError if the file can't be read, ValidationError if it isn't a valid baseline.
def load_baseline(path: Path) -> Baseline:
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))


# A drop counts only if it's lower AND the two 95% intervals don't overlap (same rule as compare).
def below_baseline(baseline: Baseline, passed: int, n: int) -> bool:
    before = (baseline.passed, baseline.total)
    return passed / n < baseline.passed / baseline.total and significant(before, (passed, n))
