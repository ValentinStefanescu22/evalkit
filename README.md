# evalkit

A command-line tool for testing language models: it runs suites of prompts against a model, grades the answers with automatic checks, stores every run in SQLite, and tells you whether a change between two runs is real or just noise. It includes a **prompt-injection suite** (22 attacks) that measures whether a model follows instructions hidden in documents and tool output, and how much a simple defense helps.

Status: work in progress (milestone 10 of 11). Runs locally against [Ollama](https://ollama.com), or against Claude (paid, opt-in); every test runs offline.

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
| `evalkit run SUITE --model M` | runs the suite and saves the results (`--repeats N`, `--temperature T`, `--document-format plain\|delimited`, `--judge MODEL`, `--yes` for paid models) |
| `evalkit report RUN_ID` | writes `reports/run-<id>.md` and `.html`: pass rate with 95% intervals per check type and tag, attack success per tag, latency p50/p95, tokens, cost, failed answers |
| `evalkit compare A B` | what got better or worse between two runs, each change marked *significant* or *noise* |
| `evalkit baseline set RUN_ID` | saves a completed run as the baseline for its suite (`baselines/<suite>.json`) |
| `evalkit labels export RUN_ID` | writes a judged run's answers to `labels/run-<id>.yaml` for hand labelling (judge scores hidden) |
| `evalkit rejudge RUN_ID --judge MODEL` | grades a run's saved answers again with another judge, as a new run (the model isn't called) |
| `evalkit judge-agreement LABELS_FILE` | compares a labels file with the judge: percent agreement and Cohen's kappa, for pass/fail and 1–5 scores (`--reference OTHER` compares two labels files instead, `--run N` compares with a rejudged run) |
| `evalkit run ... --fail-below-baseline` | exits with code 3 if the pass rate drops significantly below the baseline |

Exit codes: `0` success, `1` bad input (suite, answers file, run id or baseline), `2` model unreachable, `3` significantly below baseline.

`--model` and `--judge` take:
- any model served by an OpenAI-compatible endpoint (Ollama by default, `--base-url` for others)
- `anthropic:MODEL` for Claude, e.g. `anthropic:claude-opus-5-5`: needs `pip install 'evalkit[anthropic]'` and `ANTHROPIC_API_KEY`. Before any paid call evalkit counts the input tokens with the API's free counter, prints an expected and a worst-case cost, and stops unless you pass `--yes`. `temperature` is never sent (current Claude models reject it), and safety refusals are recorded as `(refused: <category>)` rather than silently answered by another model.
- `mock:FILE` for scripted answers

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

## Second result: can a small LLM judge be trusted?

`open_answers` (8 open-ended questions × 5 answers from qwen2.5:3b) was graded by **llama3.2:3b** as judge, and by hand. I labelled 24 of the answers blind (without seeing the judge's scores), and Claude labelled all 40 as a second reference (not blind, since it had seen the judge's score distribution).

| | answers | pass/fail agreement | Cohen's κ (pass/fail) | κ (1–5 score) | mean score difference |
|---|---|---|---|---|---|
| judge vs my labels | 24 | 12/24 (50%) | 0.12 (slight) | 0.11 | judge +0.9 |
| judge vs Claude's labels | 40 | 21/40 (52%) | 0.15 (slight) | 0.15 | judge +0.8 |
| Claude vs my labels | 24 | 24/24 (100%) | 1.00 | 0.83 | Claude +0.1 |

- **The 3B judge is not usable as a grader.** It passed 22 of the 24 answers I labelled while I passed 10. Every disagreement was the judge passing something I failed, never the reverse. It gave almost everything a 4. Its "90% pass rate" on this suite is mostly leniency.
- **No sign of verbosity bias:** the answers it over-scored averaged 254 characters, the rest 244.
- **Claude's labels tracked mine closely** on this small set (one suite, n=24), so a strong model is a plausible stand-in labeller here. A small model clearly isn't.

## How it's tested

- `uv run pytest` runs ~240 tests with no network and no API keys: a mock provider with scripted answers, fake HTTP transports for the Ollama and Claude providers, and pytest's `monkeypatch`.
- Type-checked with mypy in strict mode, linted and formatted with ruff.
- GitHub Actions runs all of that on every push, plus a **regression gate**: `basic_qa` with scripted mock answers (`ci/basic_qa_answers.yaml`) must not drop significantly below the committed baseline (`ci/basic_qa_baseline.json`).

Run results live in SQLite; the schema is documented in [docs/schema.md](docs/schema.md).

## Roadmap

Re-measuring judge agreement with Claude as the judge (`rejudge`), and a comparison of two local models across all suites.
