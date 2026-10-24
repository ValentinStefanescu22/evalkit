# Decisions

## Milestone 1: skeleton, provider interface, MockProvider, contains check

- **src layout** (`src/evalkit/`). Tests import the installed package, so a missing file in packaging shows up in tests instead of in production. Rejected: flat layout.
- **Provider is a `typing.Protocol`**, not an abstract base class. Any class with a matching `complete()` is a provider; mypy verifies the match. Rejected: ABC, which forces inheritance for no benefit.
- **One `Response` model for every provider** (text, tool calls, tokens, latency, cost) so checks, storage and reports never care which model answered.
- **MockProvider keys answers by the last user message.** Same input, same output, regardless of order or repeats. Rejected: a queue of answers in call order, which breaks when case order or repeat count changes.
- **Checks are plain functions** returning `CheckResult`. Rejected for now: a class hierarchy or registry; revisit in milestone 3 when there are five checks.
- **Tooling:** ruff (lint + format), mypy in strict mode, pytest; CI runs all four on every push with `uv sync --locked` so CI uses exactly the locked versions.
