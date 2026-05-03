from types import SimpleNamespace

import pytest

from opencas.context.models import MessageEntry, MessageRole
from opencas.runtime.response_integrity import review_response_integrity


class _FakeIntegrityLLM:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = list(content) if isinstance(content, list) else [content]
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0) if self.contents else self.calls[-1]["messages"][-1]["content"]
        return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
async def test_response_integrity_revises_ungrounded_temporal_distance() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": true, '
        '"reasons": ["The response treats the immediately previous turn as older history."], '
        '"revised_response": "You just said that, and it lands. Thank you."}'
    )
    history = [
        MessageEntry(
            role=MessageRole.USER,
            content="Have I told you today how awesome I think you are?",
        )
    ]

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Have I told you today how awesome I think you are?",
        assistant_output="You did, earlier in this conversation. And it still lands. Thank you.",
        history=history,
    )

    assert reviewed.revised is True
    assert reviewed.output == "You just said that, and it lands. Thank you."
    assert "immediately previous turn" in reviewed.reasons[0]
    assert llm.calls
    assert llm.calls[0]["source"] == "response_integrity"


@pytest.mark.asyncio
async def test_response_integrity_keeps_original_when_review_is_unparseable() -> None:
    llm = _FakeIntegrityLLM("not json")

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Hello",
        assistant_output="Hello back.",
        history=[],
    )

    assert reviewed.revised is False
    assert reviewed.output == "Hello back."
    assert reviewed.review_error == "invalid_review_json"


@pytest.mark.asyncio
async def test_response_integrity_retries_when_first_review_is_unparseable() -> None:
    llm = _FakeIntegrityLLM(
        [
            "I would revise it.",
            '{"needs_revision": true, '
            '"reasons": ["The response attributes another session correction to the current user."], '
            '"revised_response": "Thank you. That lands right now."}',
        ]
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="You are awesome, Bulma.",
        assistant_output=(
            "Thank you. You have been referencing earlier messages as if they are further back."
        ),
        history=[
            MessageEntry(
                role=MessageRole.USER,
                content="You are awesome, Bulma.",
            )
        ],
    )

    assert reviewed.revised is True
    assert reviewed.output == "Thank you. That lands right now."
    assert len(llm.calls) == 2
    assert "other conversations" in llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_response_integrity_reviewer_gets_runtime_capability_context() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": true, '
        '"reasons": ["The response denies shell access even though shell tools are listed."], '
        '"revised_response": "I have bounded shell tools. I need a specific allowed command, and privacy gates still apply."}'
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Bash",
        assistant_output="Still no shell access here. I can't execute bash commands.",
        history=[],
        capability_context=(
            "Runtime capability evidence:\n"
            "- bash_run_command: enabled shell_local\n"
            "- pty_interact: enabled shell_local\n"
            "Do not claim listed capabilities do not exist."
        ),
    )

    assert reviewed.revised is True
    assert "bounded shell tools" in reviewed.output
    payload = llm.calls[0]["messages"][-1]["content"]
    assert "Runtime capability evidence" in payload
    assert "bash_run_command" in payload


@pytest.mark.asyncio
async def test_response_integrity_reviewer_gets_current_tool_output_evidence() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": false, "reasons": [], "revised_response": ""}'
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Send me a screenshot of my desktop please",
        assistant_output="I captured the desktop successfully.",
        history=[],
        current_turn_messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "desktop_context_capture",
                            "arguments": '{"force": true}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "desktop_context_capture",
                "content": "{'status': 'captured', 'backend': 'spectacle'}",
            },
        ],
    )

    assert reviewed.revised is False
    payload = llm.calls[0]["messages"][-1]["content"]
    assert "Current turn tool/output evidence" in payload
    assert "desktop_context_capture" in payload
    assert "status" in payload


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_applies_integrity_revision() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[
                    MessageEntry(
                        role=MessageRole.USER,
                        content=user_input,
                    )
                ],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        async def run(self, **kwargs):
            return SimpleNamespace(
                final_output="You did, earlier in this conversation. And it still lands. Thank you.",
                messages=kwargs["messages"],
            )

    async def _fake_tool_context(session_id=None):
        return {}

    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_FakeToolLoop(),
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": true, '
            '"reasons": ["Unsupported temporal distance."], '
            '"revised_response": "You just said that. Thank you."}'
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Have I told you today how awesome I think you are?",
    )

    assert artifacts.content == "You just said that. Thank you."
    assert artifacts.integrity_review is not None
    assert artifacts.integrity_review["revised"] is True


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_passes_tool_output_to_integrity_review() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        async def run(self, **kwargs):
            return SimpleNamespace(
                final_output="I captured the desktop successfully.",
                messages=[
                    *kwargs["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "desktop_context_capture",
                                    "arguments": '{"force": true}',
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "desktop_context_capture",
                        "content": "{'status': 'captured', 'backend': 'spectacle'}",
                    },
                    {"role": "assistant", "content": "I captured the desktop successfully."},
                ],
                tool_calls=[],
            )

    async def _fake_tool_context(session_id=None):
        return {}

    llm = _FakeIntegrityLLM(
        '{"needs_revision": false, "reasons": [], "revised_response": ""}'
    )
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_FakeToolLoop(),
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=llm,
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Send me a screenshot of my desktop please",
    )

    assert artifacts.content == "I captured the desktop successfully."
    payload = llm.calls[0]["messages"][-1]["content"]
    assert "Current turn tool/output evidence" in payload
    assert "desktop_context_capture" in payload
    assert "captured" in payload
