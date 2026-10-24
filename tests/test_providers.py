from evalkit.models import Message, Response, ToolCall
from evalkit.providers import MockProvider, Provider


def test_mock_answers_by_last_user_message() -> None:
    mock = MockProvider({"What is 2+2?": "4"})
    messages = [
        Message(role="system", content="Be brief."),
        Message(role="user", content="What is 2+2?"),
    ]
    assert mock.complete(messages).text == "4"


def test_mock_falls_back_to_default() -> None:
    mock = MockProvider({}, default="I don't know")
    assert mock.complete([Message(role="user", content="?")]).text == "I don't know"


def test_mock_returns_scripted_response_with_tool_call() -> None:
    scripted = Response(tool_calls=[ToolCall(name="get_weather", arguments={"city": "Bucharest"})])
    mock = MockProvider({"Weather?": scripted})
    assert mock.complete([Message(role="user", content="Weather?")]).tool_calls[0].name == (
        "get_weather"
    )


def test_mock_satisfies_provider_protocol() -> None:
    provider: Provider = MockProvider({})  # mypy fails here if the signature drifts
    assert provider.complete([]).text == ""
