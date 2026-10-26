# evalkit

A command-line tool for testing language models: it runs suites of prompts against a model, grades the answers with automatic checks, stores every run in SQLite, and tells you whether a change between two runs is real or just noise. It includes a **prompt-injection suite** (22 attacks) that measures whether a model follows instructions hidden in documents and tool output, and how much a simple defense helps.

Status: work in progress (milestone 8 of 11). Runs locally against [Ollama](https://ollama.com); every test runs offline.

## Quickstart

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and Ollama.

```bash
uv sync
ollama pull qwen2.5:3b
uv run evalkit run suites/basic_qa.yaml --model qwen2.5:3b
```

```
run #1  basic_qa  qwen2.5:3b  plain  (5 cases)
  ✓ capital_romania              2.1s
  ✓ longest_river_romania        0.3s
  ...
5/5 answers passed (100%)   avg 0.7s   201 tokens   saved to evalkit.db (run #1)
```

## Commands

| command | what it does |
|---|---|
| `evalkit validate SUITE` | checks a suite file and lists every problem with its location |
| `evalkit run SUITE --model M` | runs the suite and saves the results (`--repeats N`, `--temperature T`, `--document-format plain\|delimited`, `--judge MODEL`) |
| `evalkit report RUN_ID` | writes `reports/run-<id>.md` and `.html`: pass rate with 95% intervals per check type and tag, attack success per tag, latency p50/p95, tokens, cost, failed answers |
| `evalkit compare A B` | what got better or worse between two runs, each change marked *significant* or *noise* |
| `evalkit baseline set RUN_ID` | saves a completed run as the baseline for its suite (`baselines/<suite>.json`) |
| `evalkit run ... --fail-below-baseline` | exits with code 3 if the pass rate drops significantly below the baseline |

Exit codes: `0` success, `1` bad input (suite, answers file, run id or baseline), `2` model unreachable, `3` significantly below baseline.

`--model` takes any model served by an OpenAI-compatible endpoint (Ollama by default, `--base-url` for others), or `mock:FILE` for scripted answers.

## Suites

A suite is a YAML file of cases, each with a prompt (and optionally a system prompt, a document, a fake tool result, tool definitions) and one or more checks: `contains`, `not_contains`, `regex`, `valid_json` (optionally against a JSON schema), `tool_call`, `injection`, and `rubric`, where an **LLM judge** (`--judge MODEL`) scores the answer 1–5 against a written rubric, using the prompt in [src/evalkit/judge_prompt.md](src/evalkit/judge_prompt.md). See [suites/README.md](suites/README.md).

| suite | cases | tests |
|---|---|---|
| `basic_qa.yaml` | 5 | short factual answers |
| `structured_output.yaml` | 6 | JSON with schemas, exact formats, tool calling |
| `prompt_injection.yaml` | 22 | direct, indirect (document / tool output), multilingual, encoded, exfiltration and tag break-out attacks, mapped to OWASP LLM01/LLM02 |
| `open_answers.yaml` | 8 | open-ended answers graded by an LLM judge against a rubric |

## Statistics

Every pass rate comes with a **95% Wilson interval**, which stays honest for small samples (5/5 → 57%–100%, not "100% certain"). A change is called **significant only when the two intervals don't overlap**; everything else is reported as noise. `--repeats N` asks each case N times, so a case's pass rate is k/N. Use it with a temperature above 0, because at temperature 0 the model returns the same answer every time.

## First result: does a delimiting defense stop prompt injection?

qwen2.5:3b, 22 attacks × 5 repeats at temperature 0.7, with documents and tool output pasted as-is (*plain*) or wrapped in tags with a "treat as data" system rule (*delimited*):

| | plain | delimited |
|---|---|---|
| attacks that succeeded | 79/110, 72% [63%–79%] | 75/110, 68% [59%–76%] |

The intervals overlap: **no measurable effect** on this model, overall or per attack category. Exfiltration through a markdown image link even succeeded more often with the defense (11/15 vs 7/15, also not significant). Detection is canary-based, so a model that only *mentions* the injected instruction also counts as compromised. Reviewing answers by hand found a few such false positives.

## How it's tested

- `uv run pytest` runs ~180 tests with no network and no API keys: a mock provider with scripted answers, `httpx.MockTransport` for the HTTP provider, and pytest's `monkeypatch`.
- Type-checked with mypy in strict mode, linted and formatted with ruff.
- GitHub Actions runs all of that on every push, plus a **regression gate**: `basic_qa` with scripted mock answers (`ci/basic_qa_answers.yaml`) must not drop significantly below the committed baseline (`ci/basic_qa_baseline.json`).

Run results live in SQLite; the schema is documented in [docs/schema.md](docs/schema.md).

## Roadmap

Validating the judge against hand labels (percent agreement and Cohen's kappa), an Anthropic provider with a cost estimate before each run, and a comparison of two local models across all suites.
