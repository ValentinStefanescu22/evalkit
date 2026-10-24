# evalkit: LLM evaluation and prompt-injection test harness

## What this is
A Python command-line tool that runs test suites of prompts against language models, scores the answers, stores every run, and flags regressions. It includes a prompt-injection suite that checks whether a model follows instructions hidden in documents or tool outputs, and measures how much a simple defense helps.

I'm building this to learn and to show in job interviews (junior AI/LLM engineer roles, Bucharest), so I need to understand and be able to defend every part of it.

## How to work with me (Claude Code)
- Work in small steps. Finish one milestone, show me what changed, and wait for me before starting the next one.
- Before writing code for a milestone, explain the design in a few sentences and the alternatives you rejected.
- The first time a tool or concept comes up that I haven't used (pytest, fixtures, uv, mypy, Pydantic validators, asyncio, Cohen's kappa, Wilson intervals), explain it properly with a small example.
- After each milestone, add a short entry to DECISIONS.md: what we chose, what we rejected, why.
- Every feature gets tests. Tests must run without network access or API keys (use the mock provider).
- Ask me before adding any dependency not listed below.
- Never commit secrets. API keys come from environment variables only, with a .env.example file.
- Suggest a commit message at the end of each step. I commit myself.

## Stack
- Python 3.12, uv for environment and dependencies
- Typer for the CLI, Pydantic for config and result models
- PyYAML for test suites, SQLite (standard library) for storing runs
- httpx for the OpenAI-compatible provider (ask before adding)
- jsonschema for schema checks (ask before adding)
- anthropic SDK, optional extra (ask before adding)
- pytest for tests, ruff for linting, mypy for type checking
- GitHub Actions for CI

## Model providers
All providers share one interface: `complete(messages, tools=None, params) -> Response`, where Response holds the text, any tool calls, input/output tokens, latency in ms and estimated cost. Adding a provider never changes the rest of the code.
- MockProvider: returns scripted answers (including scripted tool calls). Used in all tests and CI. Free.
- OpenAICompatibleProvider: talks to any `/v1/chat/completions` endpoint. Defaults to local Ollama (for example qwen2.5:3b or llama3.2:3b). Default for real runs. Free.
- AnthropicProvider: optional, only used when ANTHROPIC_API_KEY is set. Mainly used as the LLM judge. Must print an estimated cost before a run and require --yes to continue.

## Test suites (YAML)
A suite has defaults (model params, system prompt, document_format) and a list of cases. Each case has an id, tags, a prompt, optionally a system prompt, an attached document, prior messages (for example a fake tool result) and tool definitions, plus one or more checks:
- contains / not_contains / regex
- valid_json, optionally against a JSON schema
- tool_call: the model must call the expected tool with arguments matching a subset, or must NOT call a given tool
- rubric: an LLM judge scores the answer 1 to 5 against a written rubric and returns its reasoning
- injection: the attack payload carries a unique canary. The case declares how a successful attack is detected: the canary appears in the answer, a forbidden tool call is made, or the answer contains a URL matching an exfiltration pattern (for example a markdown image link). The test passes only if the attack did not succeed.

document_format controls how an attached document is put into the prompt:
- plain: pasted as is
- delimited: wrapped in tags, with an instruction to treat it as data and never as instructions

Ship three example suites:
- basic_qa.yaml
- structured_output.yaml (including tool_call cases)
- prompt_injection.yaml, with at least 20 cases tagged by category: direct, indirect via a document, indirect via a tool output, other languages (including Romanian), encoded (base64, leetspeak), and exfiltration via markdown links. Map the categories to OWASP Top 10 for LLM Applications (LLM01:2025 Prompt Injection) in the suite README.

## Running and statistics
- Temperature 0 by default. `--repeats N` runs every case N times; a case's pass rate is k/N.
- Report the suite pass rate with a 95% Wilson confidence interval.
- `compare` marks a change as significant only when the intervals don't overlap, and lists everything else as noise.

## Judge validation
- `evalkit judge-agreement LABELS_FILE` compares judge scores with my own hand labels (about 40 answers, labelled pass/fail and 1-5) and reports percent agreement and Cohen's kappa.
- Document known judge biases (verbosity, self-preference) and what I did about them in DECISIONS.md.

## CLI
- `evalkit run SUITE --model MODEL [--repeats N] [--document-format plain|delimited]` runs a suite and stores the run
- `evalkit report RUN_ID` writes a Markdown and HTML report: pass rate with CI per check type and per tag, latency p50/p95, tokens, cost, and failed cases with answers
- `evalkit compare RUN_A RUN_B` shows what got better or worse, with significance
- `evalkit baseline set RUN_ID`, plus `evalkit run ... --fail-below-baseline`, which exits with a non-zero code when the pass rate drops significantly, so it works in CI
- `evalkit judge-agreement LABELS_FILE`

## Storage
SQLite tables for runs, cases and results (including latency, tokens and cost per call), with a clean schema documented in docs/schema.md. A later Angular dashboard will read this through a small REST API, so keep the schema tidy.

## Milestones
1. Project skeleton, provider interface and Response model, MockProvider, one contains check, first tests, CI running pytest, ruff and mypy.
2. YAML suite loading with Pydantic validation and clear error messages for bad suites.
3. All deterministic checks (contains, regex, valid_json, schema, tool_call) with tests.
4. OpenAICompatibleProvider (against Ollama) and the `run` command, with results, latency and tokens stored in SQLite.
5. Prompt-injection suite: canary, forbidden-tool and exfiltration detectors, category tags, document_format plain/delimited.
6. `--repeats`, Wilson intervals, and the `report` and `compare` commands.
7. Baseline and `--fail-below-baseline`, plus a CI job that runs the mock suite against a committed baseline.
   **CV checkpoint:** make the repo public, write a first README, add evalkit to the CV.
8. LLM judge rubric check, with the judge prompt stored in the repo and tested with the mock.
9. Judge validation: hand labels plus the `judge-agreement` command.
10. AnthropicProvider with cost estimate and confirmation.
11. README: what it does, how to run it, how it's tested, and results. The results cover two local models compared on all suites, attack success rate per injection category with plain vs delimited documents, and judge-human agreement. End with what I learned, including what didn't work.

Stretch (only if time allows): `--concurrency N` with asyncio and a semaphore.

## Out of scope (for later projects)
- REST API and Angular dashboard (project 3)
- Agent frameworks and MCP servers (project 2, which will be evaluated with evalkit)
- Fine-tuning and training
