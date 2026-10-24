# Database schema

evalkit stores every run in a SQLite file (default `./evalkit.db`, change with `--db`).
The schema lives in `src/evalkit/store.py`; its version is kept in `PRAGMA user_version` (currently **1**).

```
runs 1 ──── * case_results 1 ──── * check_results
```

A **run** is one `evalkit run` of one suite against one model. It has one **case result** per case
(per repeat, from milestone 6), and each case result has one **check result** per check.

## runs

| column | type | meaning |
|---|---|---|
| `id` | INTEGER PK | run number shown by the CLI (`run #3`) |
| `suite_name` | TEXT | `name` from the suite file |
| `suite_path` | TEXT | path the suite was loaded from |
| `model` | TEXT | model name, e.g. `qwen2.5:3b` |
| `base_url` | TEXT | server the model was called on |
| `params` | TEXT (JSON) | model settings sent with every request, e.g. `{"temperature": 0}` |
| `status` | TEXT | `running`, `completed` or `failed` |
| `error` | TEXT | why the run failed (NULL otherwise) |
| `started_at`, `finished_at` | TEXT | ISO-8601 timestamps in UTC |

## case_results

| column | type | meaning |
|---|---|---|
| `id` | INTEGER PK | |
| `run_id` | INTEGER → `runs.id` | |
| `case_id` | TEXT | `id` from the suite file |
| `repeat_index` | INTEGER | 0 for now; numbers repeated runs of the same case (milestone 6) |
| `tags` | TEXT (JSON list) | the case's tags |
| `prompt` | TEXT | the case's prompt |
| `answer` | TEXT | the model's text answer |
| `tool_calls` | TEXT (JSON list) | `[{"name": ..., "arguments": {...}}]` |
| `passed` | INTEGER 0/1 | 1 only if every check passed |
| `input_tokens`, `output_tokens` | INTEGER | as reported by the server |
| `latency_ms` | REAL | time for the request |
| `cost_usd` | REAL | 0 for local models |

`UNIQUE (run_id, case_id, repeat_index)`: a case can't be saved twice for the same repeat.

## check_results

| column | type | meaning |
|---|---|---|
| `id` | INTEGER PK | |
| `case_result_id` | INTEGER → `case_results.id` | |
| `position` | INTEGER | order of the check within its case (0, 1, ...) |
| `type` | TEXT | `contains`, `not_contains`, `regex`, `valid_json`, `tool_call` |
| `passed` | INTEGER 0/1 | |
| `detail` | TEXT | why it failed (empty if passed) |
| `config` | TEXT (JSON) | the check's settings, so results stay readable after the suite file changes |

## Example queries

```sql
-- pass rate per run
SELECT run_id, round(100.0 * avg(passed), 1) AS pass_pct FROM case_results GROUP BY run_id;

-- every failed check in run 3, with the answer
SELECT c.case_id, k.type, k.detail, c.answer
FROM case_results c JOIN check_results k ON k.case_result_id = c.id
WHERE c.run_id = 3 AND k.passed = 0;
```
