# Decisions

## Milestone 1: skeleton, provider interface, MockProvider, contains check

- **src layout** (`src/evalkit/`). Tests import the installed package, so a missing file in packaging shows up in tests instead of in production. Rejected: flat layout.
- **Provider is a `typing.Protocol`**, not an abstract base class. Any class with a matching `complete()` is a provider; mypy verifies the match. Rejected: ABC, which forces inheritance for no benefit.
- **One `Response` model for every provider** (text, tool calls, tokens, latency, cost) so checks, storage and reports never care which model answered.
- **MockProvider keys answers by the last user message.** Same input, same output, regardless of order or repeats. Rejected: a queue of answers in call order, which breaks when case order or repeat count changes.
- **Checks are plain functions** returning `CheckResult`. Rejected for now: a class hierarchy or registry; revisit in milestone 3 when there are five checks.
- **Tooling:** ruff (lint + format), mypy in strict mode, pytest; CI runs all four on every push with `uv sync --locked` so CI uses exactly the locked versions.

## Milestone 2: YAML suite loading and validation

- **Explicit check syntax** (`- type: contains` / `value: Bucharest`) parsed as a Pydantic *discriminated union* on `type`: Pydantic reads `type` and picks the matching model, so `type: contians` gives "does not match any of the expected tags: 'contains'". Rejected: short form `- contains: Bucharest`; needs custom parsing and makes options like `case_sensitive` awkward. Could be added later as sugar.
- **`extra="forbid"` on every suite model.** A typo like `promt:` is an error instead of being silently ignored, which would otherwise make a check quietly never run.
- **Rules enforced in the models:** at least one case and one check per case, non-empty prompt, unique case ids made of `a-z 0-9 _ -` (they become DB keys and report links).
- **One `SuiteError` with readable messages** for every failure: missing file, YAML syntax (with line/column), empty file, wrong top-level shape, and validation errors rewritten from `('cases', 0, 'prompt')` to `cases[0] (capital_romania).prompt`. All problems are listed at once, not just the first. Rejected: YAML line numbers on validation errors; needs a position-tracking loader, too much code for the gain.
- **Known limit:** the duplicate-id rule runs only after every field is valid (Pydantic "after" validator), so it can appear on a second run.
- **`evalkit validate SUITE`** added (not in the spec's command list): ~10 lines, makes this milestone usable outside tests and can gate suite files in CI.
- **`yaml.safe_load`, never `yaml.load`**: plain `load` can construct arbitrary Python objects from a malicious file.
- **Added `types-PyYAML`** (dev): mypy strict rejects untyped imports; a `# type: ignore` would hide real errors.
