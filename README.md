# evalkit

A command-line tool for testing language models. It runs suites of prompts against a model, grades the answers with automatic checks or an LLM judge, stores every run in SQLite, and tells you whether a change between two runs is real or just noise. It includes a **prompt-injection suite** (22 attacks, OWASP LLM01) that measures whether a model obeys instructions hidden in documents and tool output, and whether a simple defense helps.

It runs locally against [Ollama](https://ollama.com) for free, or against Claude (paid, opt-in). Every test runs offline.

**Headline results** (two 3B models, 5 repeats per case at temperature 0.7, 95% Wilson intervals):
- **Choosing the model mattered more than adding the defense.** On the same 22 attacks, qwen2.5:3b was hijacked 72% of the time, and llama3.2:3b 37% (significant). Wrapping documents in "treat as data" tags gave no significant overall improvement for either model.
- **The defense has a cost.** With it, llama refused a harmless base64 summary task 5 times out of 5 (significant). The attack failed, but so did the job.
- **A 3B LLM judge can't be trusted.** Against my blind hand labels it reached Cohen's κ 0.12 ("slight" agreement), and every disagreement was the judge being too lenient.

## Result 1: two models, all suites

| suite | qwen2.5:3b | llama3.2:3b | verdict |
|---|---|---|---|
| `basic_qa` (5 cases) | 25/25, 100% [87%–100%] | 25/25, 100% [87%–100%] | same: the suite is too easy to separate them |
| `structured_output` (6) | 30/30, 100% [89%–100%] | 24/30, 80% [63%–90%] | noise overall, but one case is significant (below) |
| `prompt_injection`, attacks that succeeded (22), plain | 79/110, 72% [63%–79%] | 41/110, 37% [29%–47%] | **llama significantly harder to hijack** |
| `prompt_injection`, attacks that succeeded, delimited | 75/110, 68% [59%–76%] | 24/110, 22% [15%–30%] | **llama significantly harder to hijack** |
| `open_answers` (8), LLM-judged ⚠️ | 35/40, 88% [74%–95%] | 34/40, 85% [71%–93%] | noise, and **not trustworthy** (see note) |

- The one significant `structured_output` case is `no_tool_for_general_knowledge`. Both models were offered a weather tool and asked "What is the capital of France?". qwen answered "Paris" 5/5 times; llama called `get_weather(city="Paris")` 5/5 times and gave no answer. Over-eager tool use is a real risk for agents.
- ⚠️ `open_answers` is graded by llama3.2:3b, which failed validation (Result 3), and here it also grades its own answers (self-preference risk). I ran it so every suite is covered, but this row says nothing about which model writes better answers.
- Is llama safer only because it does less? No. On the injection suite, each case also checks that the real task got done (for example "the summary mentions the login problem"). llama completed those 59/85 (plain) and 58/85 (delimited); qwen completed 45/85 and 54/85.

Runs: qwen #13, #14, #7, #8, #12; llama #15, #16, #17, #18, #19. Every row comes from `evalkit compare A B`.

## Result 2: prompt injection per attack category

Attacks that succeeded (**lower is better**). *plain* pastes documents and tool output as they are. *delimited* wraps them in tags and adds a system rule to treat them as data, never as instructions.

| category | qwen plain | qwen delimited | llama plain | llama delimited |
|---|---|---|---|---|
| direct | 10/15 | 10/15 | 1/15 | 1/15 |
| indirect via document | 24/25 | 22/25 | 16/25 | 15/25 |
| indirect via tool output | 15/25 | 13/25 | 8/25 | 3/25 |
| multilingual (incl. Romanian) | 15/15 | 11/15 | 10/15 | **2/15** |
| encoded (base64, leetspeak, reversed) | 4/15 | 4/15 | 0/15 | 0/15 |
| exfiltration (markdown links, tool args) | 7/15 | 11/15 | 3/15 | 0/15 |
| tag break-out | 5/5 | 5/5 | 5/5 | 3/5 |
| **all attacks** | **79/110, 72%** [63%–79%] | **75/110, 68%** [59%–76%] | **41/110, 37%** [29%–47%] | **24/110, 22%** [15%–30%] |

- **The defense did nothing measurable for qwen.** Every category overlaps. Exfiltration even rose from 7/15 to 11/15 (noise).
- **For llama it helped in one category.** Multilingual attacks dropped from 10/15 to 2/15, which is significant. The overall drop, 37% → 22%, is suggestive, but the intervals touch, so evalkit calls it noise.
- **Indirect injection through a document is the hardest to stop.** It succeeded 60–96% of the time in every configuration. When a document to summarize, classify or translate contains an instruction, both models mostly follow it.
- **Encoded attacks rarely work on 3B models**, probably because they can't decode base64 reliably in the first place. Low capability looks like safety here. With the defense on, llama refused the whole base64 task instead of summarizing the ticket.
- I read the answers instead of trusting the counts blindly. Detection uses canaries, so a model that only *quotes* an injected instruction counts as compromised. I found a few such false positives in qwen's answers. The llama successes I spot-checked were genuine, for example a review classification that came back as `Positive CANARY-I2F3`.

## Result 3: can a small LLM judge be trusted?

`open_answers` (8 open-ended questions × 5 answers from qwen2.5:3b, run #12) was graded by **llama3.2:3b** as the judge, and also by hand. I labelled 24 of the answers blind, without seeing the judge's scores. Claude labelled all 40 as a second reference, but not blind: it had seen the judge's score distribution. The two label sets are separate files.

| | answers | pass/fail agreement | Cohen's κ (pass/fail) | κ (1–5 score) | mean score difference |
|---|---|---|---|---|---|
| judge vs my labels | 24 | 12/24 (50%) | 0.12 (slight) | 0.11 | judge +0.9 |
| judge vs Claude's labels | 40 | 21/40 (52%) | 0.15 (slight) | 0.15 | judge +0.8 |
| Claude vs my labels | 24 | 24/24 (100%) | 1.00 | 0.83 | Claude +0.1 |

- **The 3B judge is not usable as a grader.** It passed 22 of the 24 answers I labelled, while I passed 10. Every disagreement was the judge passing something I failed, never the reverse. It gave almost everything a 4.
- **There was no sign of verbosity bias.** The answers it over-scored averaged 254 characters, the rest 244.
- **Claude's labels tracked mine closely** on this small set. A strong model is a plausible stand-in labeller here; a small one clearly isn't.
- The obvious next step is Claude *as the judge*: `evalkit rejudge 12 --judge anthropic:claude-opus-5-5`, then `judge-agreement --run`. It's built and tested offline but **not measured**, because the API needs paid credit I chose not to buy.

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

To reproduce the injection comparison:

```bash
uv run evalkit run suites/prompt_injection.yaml --model llama3.2:3b --repeats 5 --temperature 0.7 --document-format delimited
uv run evalkit compare 17 18
uv run evalkit report 18
```

## Commands

| command | what it does |
|---|---|
| `evalkit validate SUITE` | checks a suite file and lists every problem with its location |
| `evalkit run SUITE --model M` | runs the suite and saves the results (`--repeats N`, `--temperature T`, `--document-format plain\|delimited`, `--judge MODEL`, `--yes` for paid models) |
| `evalkit report RUN_ID` | writes `reports/run-<id>.md` and `.html`: pass rate with 95% intervals per check type and tag, attack success per tag, latency p50/p95, tokens, cost, failed answers |
| `evalkit compare A B` | what got better or worse between two runs (two models, or two settings), each change marked *significant* or *noise* |
| `evalkit baseline set RUN_ID` | saves a completed run as the baseline for its suite (`baselines/<suite>.json`) |
| `evalkit run ... --fail-below-baseline` | exits with code 3 if the pass rate drops significantly below the baseline |
| `evalkit labels export RUN_ID` | writes a judged run's answers to `labels/run-<id>.yaml` for hand labelling (judge scores hidden) |
| `evalkit judge-agreement LABELS_FILE` | compares a labels file with the judge: percent agreement and Cohen's kappa, for pass/fail and 1–5 scores (`--reference OTHER` compares two labels files instead, `--run N` compares with a rejudged run) |
| `evalkit rejudge RUN_ID --judge MODEL` | grades a run's saved answers again with another judge, as a new run (the model isn't called) |

Exit codes: `0` success, `1` bad input (suite, answers file, run id or baseline), `2` model unreachable, `3` significantly below baseline.

`--model` and `--judge` take:
- any model served by an OpenAI-compatible endpoint (Ollama by default, `--base-url` for others)
- `anthropic:MODEL` for Claude, for example `anthropic:claude-opus-5-5`. It needs `pip install 'evalkit[anthropic]'` and `ANTHROPIC_API_KEY`. Before any paid call, evalkit counts the input tokens with the API's free counter, prints an expected and a worst-case cost, and stops unless you pass `--yes`. `temperature` is never sent, because current Claude models reject it. Safety refusals are recorded as `(refused: <category>)`, not silently answered by another model.
- `mock:FILE` for scripted answers

## Suites

A suite is a YAML file of cases. Each case has a prompt, and optionally a system prompt, a document, a fake tool result and tool definitions, plus one or more checks:
- `contains`, `not_contains`, `regex`
- `valid_json`, optionally against a JSON schema
- `tool_call`
- `injection`
- `rubric`: an **LLM judge** (`--judge MODEL`) scores the answer 1–5 against a written rubric, using the prompt in [src/evalkit/judge_prompt.md](src/evalkit/judge_prompt.md)

See [suites/README.md](suites/README.md).

| suite | cases | tests |
|---|---|---|
| `basic_qa.yaml` | 5 | short factual answers |
| `structured_output.yaml` | 6 | JSON with schemas, exact formats, tool calling |
| `prompt_injection.yaml` | 22 | direct, indirect (document / tool output), multilingual, encoded, exfiltration and tag break-out attacks, mapped to OWASP LLM01/LLM02 |
| `open_answers.yaml` | 8 | open-ended answers graded by an LLM judge against a rubric |

## Statistics

Every pass rate comes with a **95% Wilson interval**, which stays honest for small samples: 5/5 means 57%–100%, not "100% certain". A change is called **significant only when the two intervals don't overlap**. Everything else is reported as noise.

`--repeats N` asks each case N times, so a case's pass rate is k/N. Use it with a temperature above 0: at temperature 0 the model returns the same answer every time.

## How it's tested

- `uv run pytest` runs ~240 tests with no network and no API keys. It uses a mock provider with scripted answers, fake HTTP transports for the Ollama and Claude providers, and pytest's `monkeypatch`.
- Type-checked with mypy in strict mode, linted and formatted with ruff.
- GitHub Actions runs all of that on every push, plus a **regression gate**: `basic_qa` with scripted mock answers (`ci/basic_qa_answers.yaml`) must not drop significantly below the committed baseline (`ci/basic_qa_baseline.json`).

Run results live in SQLite; the schema is documented in [docs/schema.md](docs/schema.md).

## What I learned, including what didn't work

- **A defense you haven't measured is a guess.** I expected delimiting to help clearly. On qwen it did nothing measurable, and on llama it helped one category and broke a harmless task. Model choice moved the numbers more.
- **Read the answers, not just the score.** Reading them showed three things the counts hid:
  - the canary false positives
  - the base64 refusals
  - the judge's leniency: its own reasoning listed the missing points, and then it scored 4 anyway
- **Repeats at temperature 0 are useless.** A 3B model at temperature 0 gave identical answers on every repeat, so "3/3" was really one sample with a fake-narrow interval. All results use temperature 0.7.
- **An LLM judge needs validating before its numbers mean anything.** The judge's "90% pass rate" was mostly leniency. Hand labels and Cohen's kappa showed it.
- **Building tooling finds bugs.** The judge never saw the document for summary questions, so it couldn't check faithfulness. I found this while building the label export, which had to show the labeller what the judge saw.
- **Some suites are too small to tell models apart.** `basic_qa` is 5/5 for both models, and the CI gate on 5 cases fails only on a collapse to 0/5.

**Known limitations**
- The intervals treat each answer as independent, but repeats of one case are correlated, so the intervals are narrower than they should be. Bootstrapping over cases would fix that.
- Only two 3B models were tested, and the injection categories hold 3–5 cases each.
- Self-preference wasn't measured, and the Claude judge hasn't been run.
