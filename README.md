# evalkit

A command-line tool I built to test language models. You write test questions in a YAML file, evalkit asks a model, checks the answers, and saves every run in a SQLite database so you can compare runs later.

I built it to learn how LLM evaluation works in practice, and especially **prompt injection**: whether a model can be tricked by instructions hidden inside a document or a tool result.

## What it can do

- Run a test suite against a local model ([Ollama](https://ollama.com)) or Claude.
- Check answers automatically:
  - text matches and regex
  - valid JSON (optionally against a schema)
  - the right tool calls
  - whether a prompt-injection attack worked
- Grade open-ended answers with a second model as a "judge" (LLM-as-judge).
- Ask each question several times and show every pass rate with a 95% confidence interval, so a lucky run doesn't look like an improvement.
- Compare two runs and mark each change as *significant* or *noise*.
- Fail a CI build when results drop significantly below a saved baseline.

It comes with four example suites:
- `basic_qa`: short factual questions
- `structured_output`: JSON answers and tool calling
- `prompt_injection`: 22 attacks
- `open_answers`: questions graded by a judge

## How to use it

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and Ollama.

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

Some more commands:

```bash
# ask every question 5 times
uv run evalkit run suites/prompt_injection.yaml --model qwen2.5:3b --repeats 5 --temperature 0.7

# wrap documents in tags ("treat this as data")
uv run evalkit run suites/prompt_injection.yaml --model qwen2.5:3b --document-format delimited

# grade open answers with a judge model
uv run evalkit run suites/open_answers.yaml --model qwen2.5:3b --judge llama3.2:3b

# what changed between two runs
uv run evalkit compare 7 8

# write a Markdown + HTML report
uv run evalkit report 7

# check that a suite file is valid
uv run evalkit validate suites/basic_qa.yaml
```

`--model` also accepts `anthropic:MODEL` for Claude (needs `pip install 'evalkit[anthropic]'` and an `ANTHROPIC_API_KEY`). evalkit prints an estimated cost first and doesn't spend anything unless you add `--yes`.

How to write your own suite: [suites/README.md](suites/README.md). How the database is laid out: [docs/schema.md](docs/schema.md).

## What I found

I tested two small local models, qwen2.5:3b and llama3.2:3b, asking each question 5 times.

**Prompt injection.** How many of the attacks worked (lower is better):

| | documents pasted as-is | documents wrapped in "treat as data" tags |
|---|---|---|
| qwen2.5:3b | 79/110 (72%) | 75/110 (68%) |
| llama3.2:3b | 41/110 (37%) | 24/110 (22%) |

- **Which model you pick mattered more than the defense.** The difference between the two models is significant.
- **The tags didn't make a significant difference overall for either model.** For llama they did help a lot against attacks in other languages.
- **They also had a side effect:** llama started refusing a normal task, summarizing a ticket that contained base64 text.
- **Attacks hidden inside a document worked most often**, 60–96% of the time for both models.

**Other suites.**
- Both models got every `basic_qa` question right, so that suite is too easy to tell them apart.
- In `structured_output`, llama answered "What is the capital of France?" by calling a weather tool, every time.

**The LLM judge.** I hand-labelled 24 answers myself without looking at the judge's scores. Then I compared my labels with the llama3.2:3b judge:
- it agreed with me only 50% of the time on pass/fail (Cohen's kappa 0.12, "slight" agreement)
- every disagreement was the judge being too generous

So I wouldn't trust a 3B model as a grader. Claude also labelled the answers as a second opinion; that's kept in a separate file, and its labels matched mine on all 24.

## What I learned

- **Test a defense before you trust it.** I expected the "treat as data" tags to help a lot. Measuring showed they mostly didn't.
- **Read the answers, not just the score.** Three things only showed up when I read what the models actually wrote:
  - a model that just *repeated* an attack got counted as hacked
  - llama's refusals
  - the judge giving 4/5 to answers it had just criticised
- **One run proves very little.** With 5 questions, 5/5 still means "somewhere between 57% and 100%". That's why I added repeats and confidence intervals.
- **Repeats only help with temperature above 0.** At temperature 0 the model gave the exact same answer every time.
- **An LLM judge needs to be checked against humans.** My judge's "90% pass rate" was mostly the judge being lenient.
- **Building the tools found a bug.** While making the export for hand labelling, I noticed the judge had never been shown the document it was supposed to grade against.

## What didn't work / limitations

- Only two small models were tested, and each attack category has just 3–5 cases.
- The confidence intervals treat every answer as independent. Repeats of the same question aren't, so the real uncertainty is a bit bigger.
- I built Claude support but never ran it on real requests, because the API needs paid credit. Testing Claude as a better judge is the next experiment.

## How it's tested

- About 240 tests with pytest. They all run offline: a fake model gives scripted answers, and the HTTP calls are mocked.
- mypy (strict) and ruff check the code.
- GitHub Actions runs everything on every push, plus a small regression check against a saved baseline.

```bash
uv run pytest
```
