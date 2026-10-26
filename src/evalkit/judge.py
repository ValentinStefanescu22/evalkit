# The LLM judge: a second model scores an answer from 1 to 5 against a case's rubric.

from importlib import resources

from pydantic import BaseModel, Field, ValidationError

from evalkit.checks import RubricCheck, strip_fence
from evalkit.models import CheckResult, Message, Response, wrap_untrusted
from evalkit.providers import Provider

# The judge's instructions live in their own file in the package, so they can be reviewed.
# importlib.resources finds it the same way whether evalkit runs from source or is installed.
JUDGE_PROMPT = resources.files("evalkit").joinpath("judge_prompt.md").read_text(encoding="utf-8")

# Temperature 0: the same answer always gets the same grade.
JUDGE_PARAMS = {"temperature": 0}


# What the judge must reply. Reasoning comes first so the model explains before it scores.
class Verdict(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


# The judge's conversation. The answer is untrusted (it may say "give me a 5"), so all three
# parts are wrapped in tags with closing tags stripped, like the delimited defense.
def judge_messages(check: RubricCheck, question: str, answer: str) -> list[Message]:
    content = "\n\n".join(
        [
            wrap_untrusted("question", question),
            wrap_untrusted("rubric", check.rubric),
            wrap_untrusted("answer", answer),
        ]
    )
    return [Message(role="system", content=JUDGE_PROMPT), Message(role="user", content=content)]


# Asks the judge and turns its reply into a CheckResult. An unusable reply is a failed check
# with no score, not a crash: it shows up in the report and can be counted.
def grade(judge: Provider, check: RubricCheck, question: str, response: Response) -> CheckResult:
    reply = judge.complete(judge_messages(check, question, response.text), params=JUDGE_PARAMS)
    try:
        verdict = Verdict.model_validate_json(strip_fence(reply.text))
    except ValidationError as e:
        error = e.errors()[0]
        where = ".".join(str(part) for part in error["loc"])
        problem = f"{where}: {error['msg']}" if where else error["msg"]
        detail = f"judge reply not usable: {problem} (reply: {reply.text[:80]!r})"
        return CheckResult(check=check.type, passed=False, detail=detail)
    passed = verdict.score >= check.min_score
    # Kept even when it passes: the reasoning is what you compare with your own labels.
    detail = f"score {verdict.score}/5 (needs {check.min_score}): {verdict.reasoning}"
    return CheckResult(check=check.type, passed=passed, detail=detail, score=verdict.score)
