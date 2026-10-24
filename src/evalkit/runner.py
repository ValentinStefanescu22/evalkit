# Runs every case of a suite against a provider, grades the answers, and saves the results.

from collections.abc import Callable

from evalkit.models import CaseResult, Message
from evalkit.providers import Provider, ProviderError
from evalkit.store import Store
from evalkit.suite import Case, Suite


# The conversation sent to the model for one case.
def build_messages(suite: Suite, case: Case) -> list[Message]:
    messages = []
    system = case.system or suite.defaults.system
    if system:
        messages.append(Message(role="system", content=system))
    # Plain document format only for now; `delimited` (the injection defense) is milestone 5.
    content = f"{case.prompt}\n\n{case.document}" if case.document else case.prompt
    messages.append(Message(role="user", content=content))
    return messages


def run_case(suite: Suite, case: Case, provider: Provider) -> CaseResult:
    response = provider.complete(
        build_messages(suite, case), tools=case.tools or None, params=suite.defaults.params
    )
    checks = [
        check.run(response).model_copy(update={"config": check.model_dump()})
        for check in case.checks
    ]
    return CaseResult(
        case_id=case.id, tags=case.tags, prompt=case.prompt, response=response, checks=checks
    )


# Runs all cases in order, saving each as soon as it's graded. `on_result` lets the CLI
# print progress live. Any error marks the run failed (keeping saved cases) and re-raises.
def run_suite(
    run_id: int,
    suite: Suite,
    provider: Provider,
    store: Store,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    results = []
    try:
        for case in suite.cases:
            result = run_case(suite, case, provider)
            store.save_case(run_id, result)
            results.append(result)
            if on_result:
                on_result(result)
    except ProviderError as e:
        store.finish_run(run_id, "failed", str(e))
        raise
    except BaseException as e:
        # Also covers Ctrl-C (KeyboardInterrupt), so the run isn't left as "running".
        store.finish_run(run_id, "failed", type(e).__name__)
        raise
    store.finish_run(run_id, "completed")
    return results
