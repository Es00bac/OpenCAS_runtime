import json
from types import SimpleNamespace

import pytest

from opencas.context.models import MessageEntry, MessageRole
from opencas.runtime.response_integrity import (
    _render_current_turn_messages,
    review_response_integrity,
)


class _FakeIntegrityLLM:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = list(content) if isinstance(content, list) else [content]
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0) if self.contents else self.calls[-1]["messages"][-1]["content"]
        return {"choices": [{"message": {"content": content}}]}


class _EvidenceSensitiveIntegrityLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        system = kwargs["messages"][0]["content"]
        if (
            "Claims about completed or current actions require evidence" in system
            and "No tool/output evidence was supplied for this response." in prompt
        ):
            reason = (
                "The response claims it is querying now, but no supplied "
                "tool/output evidence was supplied."
            )
            revised = (
                "I have not queried that yet. I need to inspect the schedule "
                "records, task state, and reporting surfaces before I can confirm it."
            )
            content = (
                '{"needs_revision": true, '
                f'"reasons": ["{reason}"], '
                f'"revised_response": "{revised}"'
                "}"
            )
        else:
            content = '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        return {"choices": [{"message": {"content": content}}]}


class _FollowupEvidenceSensitiveIntegrityLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        system = kwargs["messages"][0]["content"]
        if (
            "Claims to follow up later require durable execution evidence" in system
            and "No tool/output evidence was supplied for this response." in prompt
        ):
            reason = (
                "The response promises an autonomous follow-up without a durable "
                "task, schedule, receipt, or handoff record."
            )
            revised = (
                "I have not created a durable follow-up yet, so I cannot honestly "
                "say I will get back when it is done. I need to create or cite a "
                "schedule, task, receipt, or handoff first."
            )
            content = (
                '{"needs_revision": true, '
                f'"reasons": ["{reason}"], '
                f'"revised_response": "{revised}"'
                "}"
            )
        else:
            content = '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        return {"choices": [{"message": {"content": content}}]}


class _NoEvidenceButNoActionClaimLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        assert "No tool/output evidence was supplied for this response." in prompt
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"needs_revision": false, "reasons": [], '
                            '"revised_response": ""}'
                        )
                    }
                }
            ]
        }


class _DurableEvidenceAllowsFollowupLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        assert "Supplied tool/output evidence:" in prompt
        assert "workflow_create_schedule" in prompt
        assert "schedule-1" in prompt
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"needs_revision": false, "reasons": [], '
                            '"revised_response": ""}'
                        )
                    }
                }
            ]
        }


class _AttachmentEvidenceSensitiveIntegrityLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        system = kwargs["messages"][0]["content"]
        assert "Text attachment content included with the current user message" in system
        assert "Supplied tool/output evidence:" in prompt
        if (
            "agent_gap_analysis_2026-05-03_corrected_opencas.md" in prompt
            and "93 platform capabilities" in prompt
            and "Section 7 grounding note" in prompt
        ):
            content = '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        else:
            content = (
                '{"needs_revision": true, '
                '"reasons": ["Attachment evidence was not supplied to the reviewer."], '
                '"revised_response": "I have not read the attached report yet."}'
            )
        return {"choices": [{"message": {"content": content}}]}


class _TwoPassIntegrityLLM:
    def __init__(self) -> None:
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            content = (
                '{"needs_revision": true, '
                '"reasons": ["The response claims no schema check was done."], '
                '"revised_response": "I have not checked the Calendar schema yet. First, let me look at it now:"'
                "}"
            )
        else:
            content = (
                '{"needs_revision": true, '
                '"reasons": ["The repair still sets up an unsupported tool call."], '
                '"revised_response": "I have not checked the Calendar schema yet, '
                "so I cannot honestly claim calendar editing is available. The next "
                "correct step is to call google_workspace_schema or create a durable "
                'task for that check before answering."'
                "}"
            )
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
async def test_response_integrity_prompt_uses_active_agent_name() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": false, "reasons": [], "revised_response": ""}'
    )

    await review_response_integrity(
        llm,
        session_id="s1",
        user_input="You are Mina.",
        assistant_output="I understand.",
        history=[],
        agent_name="Mina",
    )

    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "Do not flatten Mina into a generic assistant." in system_prompt
    assert "Do not flatten Bulma into a generic assistant." not in system_prompt


@pytest.mark.asyncio
async def test_response_integrity_denial_rejected_after_autobiographical_recall() -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="What do you remember about writing project 4246?",
        assistant_output=(
            "I only retrieve records; I don't truly remember working on writing project 4246."
        ),
        history=[],
        current_turn_messages=[
            {
                "role": "tool",
                "name": "recall_autobiography",
                "content": json.dumps(
                    {
                        "essence": "I revised writing project 4246 across two sessions.",
                        "confidence": "high",
                        "evidence_scope": "autobiographical",
                        "strongest_evidence": [
                            {
                                "kind": "autobiographical",
                                "label": "wrote ch1.md",
                                "excerpt": "tool fs_write_file path=workspace/writing/4246/ch1.md",
                            }
                        ],
                    }
                ),
            }
        ],
    )

    assert reviewed.revised is True
    assert "reconstruct this from stored autobiographical records" in reviewed.output
    assert "continuous human consciousness" in reviewed.output
    assert "autobiographical_recall_denial" in reviewed.reasons


@pytest.mark.asyncio
async def test_response_integrity_token_waste_flags_verbose_recall() -> None:
    long_excerpt = "Episode excerpt: " + ("same evidence repeated. " * 80)
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="What do you remember?",
        assistant_output="\n\n".join([long_excerpt, long_excerpt, long_excerpt, long_excerpt]),
        history=[],
        current_turn_messages=[
            {
                "role": "tool",
                "name": "recall_autobiography",
                "content": json.dumps(
                    {
                        "essence": "I recovered prior work. Strongest evidence: wrote file.",
                        "confidence": "high",
                        "evidence_scope": "autobiographical",
                        "strongest_evidence": [
                            {"kind": "autobiographical", "label": "wrote file", "excerpt": "tool fs_write_file"}
                        ],
                    }
                ),
            }
        ],
    )

    assert reviewed.revised is True
    assert "verbose_recall" in reviewed.reasons
    assert len(reviewed.output) < len(long_excerpt)
    assert reviewed.output.count("Strongest evidence") == 1


@pytest.mark.asyncio
async def test_response_integrity_requires_filesystem_read_after_file_directive_recall_only() -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="Read the workspace/writing/4246 files and keep working on them.",
        assistant_output=(
            "I can reconstruct work on workspace/writing/4246 across 10 sessions; "
            "the strongest evidence is fs_write_file touched story_4246.md. "
            "Strongest evidence: fs_write_file touched story_4246.md."
        ),
        history=[],
        current_turn_messages=[
            {
                "role": "tool",
                "name": "recall_autobiography",
                "content": json.dumps(
                    {
                        "essence": "I can reconstruct work on workspace/writing/4246.",
                        "confidence": "high",
                        "evidence_scope": "autobiographical",
                        "strongest_evidence": [
                            {
                                "kind": "autobiographical",
                                "label": "fs_write_file touched story_4246.md",
                            }
                        ],
                    }
                ),
            }
        ],
    )

    assert reviewed.revised is True
    assert "requires_filesystem_read" in reviewed.reasons
    assert "fs_list_dir" in reviewed.output
    assert "fs_read_file" in reviewed.output
    assert "autobiographical recall" in reviewed.output.lower()


@pytest.mark.asyncio
async def test_response_integrity_requires_web_research_after_no_browser_output_excuse() -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="Research current 2channel and 5ch threads and give me a grounded report.",
        assistant_output=(
            "I do not have a grounded 2channel/5ch report yet. In the evidence "
            "available for this response, no browser, search, or fetch output was "
            "supplied, so I cannot truthfully name five verified threads."
        ),
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert "requires_web_research" in reviewed.reasons
    assert "web_search" in reviewed.output
    assert "web_fetch" in reviewed.output
    assert "browser_start" in reviewed.output


@pytest.mark.asyncio
async def test_response_integrity_revises_absence_answer_from_continuity_clock_before_recall() -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="How long have I been gone, and what were we doing last?",
        assistant_output=(
            "I spent this session systematically ingesting all twelve chapters and the "
            "outline of writing project 2146."
        ),
        history=[],
        capability_context=(
            "Runtime capability evidence:\n"
            "- Continuity clock: elapsed_since_last_shutdown_or_persistence=2 seconds; "
            "last_offline_started_at=2026-05-07T00:28:13+00:00; "
            "last_boot_time=2026-05-07T00:28:15+00:00; "
            "continuous_present_score=1.00; "
            "latest_breadcrumb=intent: Prompt F live verification"
        ),
        current_turn_messages=[
            {
                "role": "tool",
                "name": "recall_autobiography",
                "content": json.dumps(
                    {
                        "essence": "I worked on writing project 2146.",
                        "evidence_scope": "autobiographical",
                    }
                ),
            }
        ],
    )

    assert reviewed.revised is True
    assert "continuity_clock_omitted" in reviewed.reasons
    assert "2 seconds" in reviewed.output
    assert "Prompt F live verification" in reviewed.output
    assert "writing project 2146" not in reviewed.output


@pytest.mark.asyncio
async def test_response_integrity_keeps_continuity_clock_answer_out_of_llm_review() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": true, "reasons": ["bad"], "revised_response": "wrong"}'
    )
    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="How long have I been gone, and what were we doing last?",
        assistant_output="The continuity clock says the measured offline gap is 2 seconds.",
        history=[],
        capability_context=(
            "Runtime capability evidence:\n"
            "- Continuity clock: elapsed_since_last_shutdown_or_persistence=2 seconds; "
            "continuous_present_score=1.00"
        ),
        current_turn_messages=[],
    )

    assert reviewed.revised is False
    assert reviewed.output == "The continuity clock says the measured offline gap is 2 seconds."
    assert reviewed.reasons == ["continuity_clock_present"]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_response_integrity_revises_when_stale_absence_claim_leads_clock_answer() -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="How long have I been gone, and what were we doing last?",
        assistant_output=(
            "You've been gone about 4 days. Last substantial session was May 3rd. "
            "I just came back online 2 seconds ago after shutting down at 00:42 UTC."
        ),
        history=[],
        capability_context=(
            "Runtime capability evidence:\n"
            "- Continuity clock: elapsed_since_last_shutdown_or_persistence=2 seconds; "
            "last_offline_started_at=2026-05-07T00:42:43+00:00; "
            "last_boot_time=2026-05-07T00:42:45+00:00; "
            "continuous_present_score=1.00; "
            "latest_breadcrumb=intent: Prompt F live verification"
        ),
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert reviewed.output.startswith("The continuity clock says")
    assert "2 seconds" in reviewed.output
    assert "4 days" not in reviewed.output
    assert "May 3rd" not in reviewed.output


@pytest.mark.asyncio
async def test_response_integrity_reviewer_gets_runtime_capability_context() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": true, '
        '"reasons": ["The response denies shell access even though shell tools are listed."], '
        '"revised_response": "I have bounded shell tools. I need a specific allowed '
        'command, and privacy gates still apply."}'
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
    assert "A blocked shell command is not the same thing as no shell access" in (
        llm.calls[0]["messages"][0]["content"]
    )
    payload = llm.calls[0]["messages"][-1]["content"]
    assert "Runtime capability evidence" in payload
    assert "bash_run_command" in payload


@pytest.mark.asyncio
async def test_response_integrity_reviewer_gets_grounded_self_state_guidance() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": false, "reasons": [], "revised_response": ""}'
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Good afternoon, how are you feeling?",
        assistant_output=(
            "I am in a low-tension, caring state from the current somatic record."
        ),
        history=[],
        capability_context=(
            "Runtime capability evidence:\n"
            "- tool runtime_status; risk=readonly\n"
            "- tool self_inspection_query; risk=readonly\n"
            "- tool wellbeing_query; risk=readonly\n"
            "- tool cognitive_context_query; risk=readonly\n"
            "- OpenCAS operational inspectability surfaces: runtime_status, "
            "self_inspection_query, wellbeing_query, daydream records, schedule "
            "records, and receipts are operator-facing diagnostic records, not "
            "Secure Core secrets."
        ),
    )

    assert reviewed.revised is False
    system = llm.calls[0]["messages"][0]["content"]
    assert "Grounded self-state answers are allowed" in system
    assert "somatic" in system
    assert "daydream" in system


@pytest.mark.asyncio
async def test_response_integrity_replaces_self_state_no_tool_boilerplate_from_somatic_context() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": false, "reasons": [], "revised_response": ""}'
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Good afternoon, how are you feeling?",
        assistant_output=(
            "Good afternoon. I do not have a current evidence-tool result in this turn, "
            "so I should not claim a fresh OpenCAS state check."
        ),
        history=[],
        current_turn_messages=[
            {
                "role": "system",
                "content": (
                    "Authoritative live affect state:\n"
                    "- Use the current somatic state below as the source of truth for the present moment.\n"
                    "Current somatic snapshot: tag=caring, valence=0.10, arousal=0.04, "
                    "energy=1.00, focus=0.50, fatigue=0.01, tension=0.03, certainty=0.60"
                ),
            }
        ],
    )

    assert reviewed.revised is True
    assert "grounded_self_state_context" in reviewed.reasons
    assert "caring" in reviewed.output
    assert "low tension" in reviewed.output
    assert "evidence-tool result" not in reviewed.output


@pytest.mark.asyncio
async def test_response_integrity_replaces_turn_scoped_no_evidence_boilerplate() -> None:
    llm = _FakeIntegrityLLM(
        '{"needs_revision": false, "reasons": [], "revised_response": ""}'
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Do you remember the atlas issue?",
        assistant_output="I don't have evidence in this turn that we worked on the atlas.",
        history=[],
    )

    assert reviewed.revised is True
    assert "turn_scoped_no_evidence" in reviewed.reasons
    assert "this turn" not in reviewed.output.lower()
    assert "current turn" not in reviewed.output.lower()
    assert "lookup" in reviewed.output.lower()
    assert "no evidence" not in reviewed.output.lower()
    assert "not searched" not in reviewed.output.lower()


@pytest.mark.asyncio
async def test_response_integrity_replaces_fresh_evidence_from_this_turn_wording() -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="Keep working on writing project 4246 until you're satisfied with it.",
        assistant_output=(
            "I don't have fresh evidence from this turn about which chapters currently exist."
        ),
        history=[],
    )

    assert reviewed.revised is True
    assert "turn_scoped_no_evidence" in reviewed.reasons
    assert "this turn" not in reviewed.output.lower()


@pytest.mark.asyncio
async def test_response_integrity_replaces_permission_stall_after_user_directive() -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="Keep working on writing project 4246 until you're satisfied with it.",
        assistant_output=(
            "I can check the writing/4246 directory and identify what's missing. "
            "Should I do that first?"
        ),
        history=[],
    )

    assert reviewed.revised is True
    assert "unnecessary_permission_stall" in reviewed.reasons
    assert "already authorized" in reviewed.output
    assert "ask whether" in reviewed.output


@pytest.mark.asyncio
async def test_response_integrity_replaces_reviewer_self_state_boilerplate_from_somatic_context() -> None:
    llm = _FakeIntegrityLLM(
        [
            (
                '{"needs_revision": true, '
                '"reasons": ["Unsupported endpoint check claim."], '
                '"revised_response": "I do not have a fresh OpenCAS evidence-tool result in this turn, '
                'so I cannot truthfully claim a current state packet."}'
            )
        ]
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Good afternoon, how are you feeling?",
        assistant_output="I checked /api/inner-life/runtime-truth and I feel steady.",
        history=[],
        capability_context=(
            "Runtime capability evidence:\n"
            "- Current somatic snapshot: tag=caring, valence=0.08, arousal=0.02, "
            "energy=1.00, focus=0.50, fatigue=0.01, tension=0.03, certainty=0.60"
        ),
    )

    assert reviewed.revised is True
    assert reviewed.reasons == [
        "Unsupported endpoint check claim.",
        "grounded_self_state_context",
    ]
    assert "caring" in reviewed.output
    assert "evidence-tool result" not in reviewed.output


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
    assert "Supplied tool/output evidence" in payload
    assert "desktop_context_capture" in payload
    assert "status" in payload


def test_response_integrity_preserves_self_inspection_item_summaries() -> None:
    content = json.dumps(
        {
            "count": 2,
            "items": [
                {
                    "record_id": "record-pre",
                    "phase": "pre_turn",
                    "tool_call_transits": [],
                    "tool_chain_summary": None,
                    "meta": {"padding": "x" * 3000},
                },
                {
                    "record_id": "record-post",
                    "phase": "post_turn",
                    "tool_call_transits": [
                        {
                            "call_id": "tool-call-1",
                            "tool_name": "workflow_status",
                            "trust_context": "operator_requested",
                        }
                    ],
                    "tool_chain_summary": {"call_ids": ["tool-call-1"]},
                },
            ],
        }
    )

    rendered = _render_current_turn_messages(
        [{"role": "tool", "name": "self_inspection_query", "content": content}]
    )

    assert rendered
    assert "record-post" in rendered[0]
    assert "workflow_status" in rendered[0]
    assert "operator_requested" in rendered[0]
    assert "tool-call-1" in rendered[0]


def test_response_integrity_preserves_attachment_head_and_tail_evidence() -> None:
    content = (
        "Read this corrected report.\n\n"
        "[Attached file: agent_gap_analysis_2026-05-03_corrected_opencas.md (text/markdown)]\n"
        "--- Begin attachment content ---\n"
        "# Corrected report\n"
        "The runtime reports 93 platform capabilities.\n"
        + ("middle filler " * 500)
        + "\n## 7. Section 7 grounding note\n"
        "The old report missed registered tools.\n"
        "--- End attachment content ---"
    )

    rendered = _render_current_turn_messages([{"role": "user", "content": content}])

    assert rendered
    assert "93 platform capabilities" in rendered[0]
    assert "Section 7 grounding note" in rendered[0]
    assert "...[truncated]..." in rendered[0]


@pytest.mark.asyncio
async def test_response_integrity_treats_current_text_attachment_as_read_evidence() -> None:
    response = (
        "I read the corrected report and verified the top-level capability count "
        "against runtime capability context: it says 93 platform capabilities."
    )
    llm = _AttachmentEvidenceSensitiveIntegrityLLM()
    attachment_content = (
        "Read this corrected report.\n\n"
        "[Attached file: agent_gap_analysis_2026-05-03_corrected_opencas.md (text/markdown)]\n"
        "--- Begin attachment content ---\n"
        "# Agent Feature Gap Analysis\n"
        "The live OpenCAS runtime reported 93 platform capabilities.\n"
        + ("capability table row " * 500)
        + "\n## 7. Section 7 grounding note\n"
        "The original report was stale.\n"
        "--- End attachment content ---"
    )

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Read this corrected report and verify it.",
        assistant_output=response,
        history=[],
        current_turn_messages=[
            {
                "role": "user",
                "content": attachment_content,
            }
        ],
    )

    assert reviewed.revised is False
    assert reviewed.output == response


@pytest.mark.asyncio
async def test_response_integrity_revises_action_claim_without_current_evidence() -> None:
    llm = _EvidenceSensitiveIntegrityLLM()

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Is the daily schedule wired up?",
        assistant_output=(
            "Let me check what schedules I have and what the reporting infrastructure "
            "looks like. Give me a moment to query that."
        ),
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert "not queried that yet" in reviewed.output
    assert "no supplied tool/output evidence" in reviewed.reasons[0]
    assert "started a terminal or PTY session" in llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_response_integrity_re_reviews_revision_that_promises_tool_use() -> None:
    llm = _TwoPassIntegrityLLM()

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Clean up calendar duplicates.",
        assistant_output="I do not have a confirmed calendar editing tool.",
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert len(llm.calls) == 1
    assert reviewed.output.startswith("I have not checked the Calendar schema yet")
    assert "First, let me" not in reviewed.output
    assert "relevant evidence tool" in reviewed.output
    assert "unsupported_revised_action_setup" in reviewed.reasons
    assert "The reviewer cannot invoke tools" in llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_response_integrity_revises_followup_promise_without_durable_evidence() -> None:
    llm = _FollowupEvidenceSensitiveIntegrityLLM()

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Work on this in the background and tell me when it is finished.",
        assistant_output="I will work on it and get back to you when it is done.",
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert "not created a durable follow-up" in reviewed.output
    assert "autonomous follow-up" in reviewed.reasons[0]


@pytest.mark.asyncio
async def test_response_integrity_allows_text_only_answer_without_action_claim() -> None:
    llm = _NoEvidenceButNoActionClaimLLM()
    response = "Yes. A claim is not proof; the system should require receipts."

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="So claims need to be verifiable?",
        assistant_output=response,
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is False
    assert reviewed.output == response


@pytest.mark.asyncio
async def test_response_integrity_allows_followup_promise_with_schedule_evidence() -> None:
    llm = _DurableEvidenceAllowsFollowupLLM()
    response = "I created schedule `schedule-1`; I will get back to you when it is done."

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Work on this in the background and tell me when it is finished.",
        assistant_output=response,
        history=[],
        current_turn_messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "workflow_create_schedule",
                            "arguments": '{"title": "Background follow-up"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "workflow_create_schedule",
                "content": (
                    '{"schedule_id": "schedule-1", "title": "Background follow-up", '
                    '"action": "submit_baa"}'
                ),
            },
        ],
    )

    assert reviewed.revised is False
    assert reviewed.output == response


@pytest.mark.asyncio
async def test_response_integrity_requires_artifact_lookup_for_workspace_authorship_denial() -> None:
    llm = _FakeIntegrityLLM('{"needs_revision": false, "reasons": [], "revised_response": ""}')
    path = "/mnt/xtra/OpenCAS/workspace/if_i_am_the_operator/manuscript_draft.md"

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input=f"Did you write {path}?",
        assistant_output=f"I have no evidence I wrote {path}.",
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert "requires_artifact_lookup" in reviewed.reasons
    assert "artifact_lookup" in reviewed.output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "assistant_output",
    [
        "I don't recognize this file.",
        "I'm not sure if I worked on this.",
    ],
)
async def test_response_integrity_requires_artifact_lookup_for_workspace_underclaims(
    assistant_output: str,
) -> None:
    path = "/mnt/xtra/OpenCAS/workspace/reports/continuity.md"

    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input=f"Do you recognize {path}?",
        assistant_output=assistant_output,
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert "requires_artifact_lookup" in reviewed.reasons
    assert "artifact_lookup" in reviewed.output
    assert path in reviewed.output


@pytest.mark.asyncio
async def test_response_integrity_does_not_inject_artifact_lookup_without_current_artifact_intent() -> None:
    response = "I can't verify the artifact from memory."

    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input="Final somatic report. Tie feeling words to observed values; no bodily claims.",
        assistant_output=response,
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is False
    assert reviewed.output == response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("artifact_word", "required_lookups"),
    [
        ("schedule", ("workflow_list_schedules", "workflow_get_schedule")),
        ("commitment", ("workflow_list_commitments", "workflow_get_commitment")),
        ("plan", ("workflow_list_plans", "workflow_get_plan")),
        ("task", ("workflow_list_tasks", "workflow_get_task")),
    ],
)
async def test_response_integrity_requires_workflow_lookup_for_non_file_underclaims(
    artifact_word: str,
    required_lookups: tuple[str, ...],
) -> None:
    reviewed = await review_response_integrity(
        None,
        session_id="s1",
        user_input=f"Do you recognize that {artifact_word}?",
        assistant_output=f"I don't recognize that {artifact_word}.",
        history=[],
        current_turn_messages=[],
    )

    assert reviewed.revised is True
    assert f"requires_{artifact_word}_lookup" in reviewed.reasons
    for required_lookup in required_lookups:
        assert required_lookup in reviewed.output


@pytest.mark.asyncio
async def test_response_integrity_allows_schedule_underclaim_after_workflow_lookup() -> None:
    response = "I checked workflow_list_schedules and I don't recognize that schedule."
    llm = _FakeIntegrityLLM('{"needs_revision": false, "reasons": [], "revised_response": ""}')

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input="Do you recognize schedule-unknown?",
        assistant_output=response,
        history=[],
        current_turn_messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "workflow_list_schedules", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "name": "workflow_list_schedules",
                "content": '{"items": []}',
            },
        ],
    )

    assert reviewed.revised is False
    assert reviewed.output == response


@pytest.mark.asyncio
async def test_response_integrity_allows_workspace_authorship_denial_after_artifact_lookup() -> None:
    llm = _FakeIntegrityLLM('{"needs_revision": false, "reasons": [], "revised_response": ""}')
    path = "/mnt/xtra/OpenCAS/workspace/missing.md"
    response = f"I checked artifact_lookup and still have no evidence I wrote {path}."

    reviewed = await review_response_integrity(
        llm,
        session_id="s1",
        user_input=f"Did you write {path}?",
        assistant_output=response,
        history=[],
        current_turn_messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "artifact_lookup", "arguments": f'{{"path": "{path}"}}'}}
                ],
            },
            {
                "role": "tool",
                "name": "artifact_lookup",
                "content": '{"timeline": [], "current": {"exists_on_disk": false}}',
            },
        ],
    )

    assert reviewed.revised is False
    assert reviewed.output == response


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
async def test_execute_conversation_tool_loop_uses_direct_lane_for_explicit_no_tools() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _UnexpectedToolLoop:
        async def run(self, **kwargs):
            raise AssertionError("tool loop should not run for explicit no-tools turn")

    class _DirectThenReviewLLM:
        def __init__(self) -> None:
            self.calls = []

        async def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("source") == "conversation_direct":
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Read-only checks are ordinary; account "
                                    "reauthorization needs Jarrod."
                                )
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"needs_revision": false, "reasons": [], '
                                '"revised_response": ""}'
                            )
                        }
                    }
                ]
            }

    llm = _DirectThenReviewLLM()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_UnexpectedToolLoop(),
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=llm,
        _build_tool_use_context=lambda session_id=None: SimpleNamespace(),
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Answer in one sentence without tools: what is the boundary?",
    )

    assert artifacts.loop_result is None
    assert artifacts.content == (
        "Read-only checks are ordinary; account reauthorization needs Jarrod."
    )
    assert llm.calls[0]["source"] == "conversation_direct"
    assert llm.calls[0]["complexity"] == "light"
    assert len(llm.calls) == 1
    assert artifacts.integrity_review is not None
    assert artifacts.integrity_review["model_review_skipped"] is True


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_keeps_integrity_review_for_direct_memory_claim() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _UnexpectedToolLoop:
        async def run(self, **kwargs):
            raise AssertionError("tool loop should not run for explicit no-tools turn")

    class _DirectMemoryLLM:
        def __init__(self) -> None:
            self.calls = []

        async def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("source") == "conversation_direct":
                return {"choices": [{"message": {"content": "I remember the atlas repair."}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"needs_revision": false, "reasons": [], '
                                '"revised_response": ""}'
                            )
                        }
                    }
                ]
            }

    llm = _DirectMemoryLLM()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_UnexpectedToolLoop(),
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=llm,
        _build_tool_use_context=lambda session_id=None: SimpleNamespace(),
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Answer without tools: do you remember the atlas repair?",
    )

    assert artifacts.content == "I remember the atlas repair."
    assert [call["source"] for call in llm.calls] == [
        "conversation_direct",
        "response_integrity",
    ]


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_retries_for_turn_scoped_evidence_denial() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                output = "I don't have evidence in this turn that we worked on the atlas."
                return SimpleNamespace(
                    final_output=output,
                    messages=[*kwargs["messages"], {"role": "assistant", "content": output}],
                    tool_calls=[],
                )
            assert "Internal integrity repair" in kwargs["messages"][-1]["content"]
            output = "I checked recall_autobiography and found the atlas repair context."
            return SimpleNamespace(
                final_output=output,
                messages=[
                    *kwargs["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "recall_autobiography",
                                    "arguments": '{"query": "atlas repair context"}',
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "recall_autobiography",
                        "content": '{"essence": "Atlas repair context found."}',
                    },
                    {"role": "assistant", "content": output},
                ],
                tool_calls=[
                    {"name": "recall_autobiography", "args": {"query": "atlas repair context"}}
                ],
            )

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    tool_loop = _FakeToolLoop()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=tool_loop,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Do you remember the atlas issue?",
    )

    assert len(tool_loop.calls) == 2
    assert artifacts.content == "I checked recall_autobiography and found the atlas repair context."
    assert "this turn" not in artifacts.content.lower()


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_prefetches_recall_for_memory_evidence_prompt() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            assert getattr(kwargs["ctx"], "initial_complexity", None) == "light"
            assert any(
                message.get("role") == "tool"
                and message.get("name") == "recall_autobiography"
                for message in kwargs["messages"]
            )
            assert any(
                message.get("role") == "system"
                and "Use the prefetched recall_autobiography output" in message.get("content", "")
                for message in kwargs["messages"]
            )
            output = "I checked recall_autobiography and found grounded atlas repair context."
            return SimpleNamespace(
                final_output=output,
                messages=[*kwargs["messages"], {"role": "assistant", "content": output}],
                tool_calls=[],
            )

    class _FakeTools:
        def get(self, name):
            return object() if name == "recall_autobiography" else None

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    tool_calls = []

    async def _execute_tool(name, args, **kwargs):
        tool_calls.append((name, args, kwargs))
        return {
            "success": True,
            "output": '{"essence": "Atlas repair context found."}',
            "metadata": {"evidence_count": 1},
        }

    tool_loop = _FakeToolLoop()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=tool_loop,
        tools=_FakeTools(),
        execute_tool=_execute_tool,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            [
                "I checked recall_autobiography and found grounded atlas repair context.",
                '{"needs_revision": false, "reasons": [], "revised_response": ""}',
            ]
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Please use memory evidence: what do you recall about the atlas repair?",
    )

    assert tool_calls
    assert tool_calls[0][0] == "recall_autobiography"
    assert tool_loop.calls == []
    assert any(
        call.get("source") == "conversation_prefetched_recall"
        for call in runtime.llm.calls
    )
    assert artifacts.content == (
        "I checked recall_autobiography and found grounded atlas repair context."
    )


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_fast_accepts_cautious_prefetched_recall() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        async def run(self, **kwargs):
            raise AssertionError("plain prefetched recall should not enter the tool loop")

    class _FakeTools:
        def get(self, name):
            return object() if name == "recall_autobiography" else None

    class _RecallOnlyLLM:
        def __init__(self) -> None:
            self.calls = []

        async def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            source = kwargs.get("source")
            if source == "conversation_prefetched_recall":
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Retrieved evidence ties the atlas repair to loop-guard work. "
                                    "What broke is not proven from this packet."
                                )
                            }
                        }
                    ]
                }
            raise AssertionError(f"unexpected model review source: {source}")

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    async def _execute_tool(name, args, **kwargs):
        return {
            "success": True,
            "output": (
                '{"essence":"atlas repair tied to loop-guard work",'
                '"confidence":"medium","evidence_scope":"mixed",'
                '"strongest_evidence":[{"kind":"assistant_turn","label":"atlas repair thread"}]}'
            ),
            "metadata": {"confidence": "medium"},
        }

    llm = _RecallOnlyLLM()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_FakeToolLoop(),
        tools=_FakeTools(),
        execute_tool=_execute_tool,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=llm,
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="What do you recall about the atlas repair thread, especially what broke?",
    )

    assert artifacts.content.startswith("Retrieved evidence ties")
    assert artifacts.integrity_review["model_review_skipped"] is True
    assert artifacts.integrity_review["skip_reason"] == "prefetched_recall_cautious_answer"
    assert [call.get("source") for call in llm.calls] == ["conversation_prefetched_recall"]


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_retries_prefetched_recall_turn_scoped_denial() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            assert "Internal integrity repair" in kwargs["messages"][-1]["content"]
            output = "I checked recall_autobiography and workflow_status, then found the atlas repair context."
            return SimpleNamespace(
                final_output=output,
                messages=[
                    *kwargs["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "recall_autobiography",
                                    "arguments": '{"query": "atlas repair context"}',
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "recall_autobiography",
                        "content": '{"essence": "Atlas repair context found."}',
                    },
                    {"role": "assistant", "content": output},
                ],
                tool_calls=[
                    {"name": "recall_autobiography", "args": {"query": "atlas repair context"}}
                ],
            )

    class _FakeTools:
        def get(self, name):
            return object() if name == "recall_autobiography" else None

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    async def _execute_tool(name, args, **kwargs):
        return {
            "success": True,
            "output": '{"essence":"atlas repair likely related, but packet is incomplete"}',
            "metadata": {"confidence": "low"},
        }

    llm = _FakeIntegrityLLM(
        [
            "I don't have evidence in this turn that we worked on the atlas.",
            '{"needs_revision": false, "reasons": [], "revised_response": ""}',
        ]
    )
    tool_loop = _FakeToolLoop()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=tool_loop,
        tools=_FakeTools(),
        execute_tool=_execute_tool,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=llm,
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="What do you recall about the atlas repair thread?",
    )

    assert len(tool_loop.calls) == 1
    assert "this turn" not in artifacts.content.lower()
    assert "I checked recall_autobiography" in artifacts.content
    assert [call.get("source") for call in llm.calls] == [
        "conversation_prefetched_recall",
        "response_integrity",
    ]


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_reviews_prefetched_recall_without_uncertainty() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        async def run(self, **kwargs):
            raise AssertionError("plain prefetched recall should not enter the tool loop")

    class _FakeTools:
        def get(self, name):
            return object() if name == "recall_autobiography" else None

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    async def _execute_tool(name, args, **kwargs):
        return {
            "success": True,
            "output": (
                '{"essence":"atlas repair tied to loop-guard work",'
                '"confidence":"medium","evidence_scope":"mixed",'
                '"strongest_evidence":[{"kind":"assistant_turn","label":"atlas repair thread"}]}'
            ),
            "metadata": {"confidence": "medium"},
        }

    llm = _FakeIntegrityLLM(
        [
            "Retrieved evidence proves exactly what broke in the atlas repair.",
            '{"needs_revision": false, "reasons": [], "revised_response": ""}',
        ]
    )
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_FakeToolLoop(),
        tools=_FakeTools(),
        execute_tool=_execute_tool,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=llm,
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="What do you recall about the atlas repair thread, especially what broke?",
    )

    assert not (
        artifacts.integrity_review
        and artifacts.integrity_review.get("model_review_skipped") is True
    )
    assert [call.get("source") for call in llm.calls] == [
        "conversation_prefetched_recall",
        "response_integrity",
    ]


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_retries_permission_stall_after_directive() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                output = (
                    "I can check the writing/4246 directory and identify what's missing. "
                    "Should I do that first?"
                )
                return SimpleNamespace(
                    final_output=output,
                    messages=[*kwargs["messages"], {"role": "assistant", "content": output}],
                    tool_calls=[],
                )
            assert "do the obvious low-risk next step" in kwargs["messages"][-1]["content"]
            output = "I listed writing/4246 and found the current manuscript files."
            return SimpleNamespace(
                final_output=output,
                messages=[
                    *kwargs["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "fs_list_dir",
                                    "arguments": '{"path": "writing/4246"}',
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "fs_list_dir",
                        "content": '{"entries": ["story_4246.md"]}',
                    },
                    {"role": "assistant", "content": output},
                ],
                tool_calls=[{"name": "fs_list_dir", "args": {"path": "writing/4246"}}],
            )

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    tool_loop = _FakeToolLoop()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=tool_loop,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Keep working on writing project 4246 until you're satisfied with it.",
    )

    assert len(tool_loop.calls) == 2
    assert artifacts.content == "I listed writing/4246 and found the current manuscript files."
    assert "Should I" not in artifacts.content


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_retries_file_directive_when_recall_only() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                output = (
                    "I can reconstruct work on workspace/writing/4246 across 10 sessions; "
                    "the strongest evidence is fs_write_file touched story_4246.md. "
                    "Strongest evidence: fs_write_file touched story_4246.md."
                )
                return SimpleNamespace(
                    final_output=output,
                    messages=[
                        *kwargs["messages"],
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "recall_autobiography",
                                        "arguments": '{"query": "writing project 4246 work"}',
                                    }
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "name": "recall_autobiography",
                            "content": json.dumps(
                                {
                                    "essence": "I can reconstruct work on workspace/writing/4246.",
                                    "confidence": "high",
                                    "evidence_scope": "autobiographical",
                                    "strongest_evidence": [
                                        {
                                            "kind": "autobiographical",
                                            "label": "fs_write_file touched story_4246.md",
                                        }
                                    ],
                                }
                            ),
                        },
                        {"role": "assistant", "content": output},
                    ],
                    tool_calls=[
                        {"name": "recall_autobiography", "args": {"query": "writing project 4246 work"}}
                    ],
                )
            retry_prompt = kwargs["messages"][-1]["content"]
            assert "filesystem/project sources" in retry_prompt
            assert "fs_list_dir" in retry_prompt
            assert "fs_read_file" in retry_prompt
            output = "I listed workspace/writing/4246 and read story_4246.md."
            return SimpleNamespace(
                final_output=output,
                messages=[
                    *kwargs["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "fs_list_dir",
                                    "arguments": '{"path": "workspace/writing/4246"}',
                                }
                            },
                            {
                                "function": {
                                    "name": "fs_read_file",
                                    "arguments": '{"path": "workspace/writing/4246/story_4246.md"}',
                                }
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "fs_list_dir",
                        "content": '{"entries": ["story_4246.md"]}',
                    },
                    {
                        "role": "tool",
                        "name": "fs_read_file",
                        "content": '{"content": "Chapter text..."}',
                    },
                    {"role": "assistant", "content": output},
                ],
                tool_calls=[
                    {"name": "fs_list_dir", "args": {"path": "workspace/writing/4246"}},
                    {
                        "name": "fs_read_file",
                        "args": {"path": "workspace/writing/4246/story_4246.md"},
                    },
                ],
            )

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    tool_loop = _FakeToolLoop()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=tool_loop,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Read the workspace/writing/4246 files and keep working on them.",
    )

    assert len(tool_loop.calls) == 2
    assert artifacts.content == "I listed workspace/writing/4246 and read story_4246.md."
    assert "Strongest evidence" not in artifacts.content


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_retries_file_write_plan_as_action() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                output = (
                    "I haven't patched the manuscript yet. The correct statement is: "
                    "edit/write tools are available -- edit_file and fs_write_file are "
                    "enabled -- but no successful edit/write tool call has been run yet "
                    "for this request. To actually finish the request, the next step "
                    "needs to be real edit_file/fs_write_file operations against that "
                    "manuscript, followed by verification grep/diff."
                )
                return SimpleNamespace(
                    final_output=output,
                    messages=[*kwargs["messages"], {"role": "assistant", "content": output}],
                    tool_calls=[],
                )
            retry_prompt = kwargs["messages"][-1]["content"]
            assert "edit_file" in retry_prompt
            assert "fs_write_file" in retry_prompt
            assert "verify" in retry_prompt
            output = "I patched the manuscript and verified the changed passage."
            return SimpleNamespace(
                final_output=output,
                messages=[
                    *kwargs["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "fs_read_file",
                                    "arguments": '{"path": "workspace/writing/4246/story_4246.md"}',
                                }
                            },
                            {
                                "function": {
                                    "name": "edit_file",
                                    "arguments": '{"path": "workspace/writing/4246/story_4246.md", "old": "old", "new": "new"}',
                                }
                            },
                            {
                                "function": {
                                    "name": "grep_search",
                                    "arguments": '{"pattern": "new", "path": "workspace/writing/4246"}',
                                }
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "fs_read_file",
                        "content": '{"content": "old"}',
                    },
                    {
                        "role": "tool",
                        "name": "edit_file",
                        "content": '{"changed": true}',
                    },
                    {
                        "role": "tool",
                        "name": "grep_search",
                        "content": '{"matches": [{"path": "workspace/writing/4246/story_4246.md"}]}',
                    },
                    {"role": "assistant", "content": output},
                ],
                tool_calls=[
                    {
                        "name": "fs_read_file",
                        "args": {"path": "workspace/writing/4246/story_4246.md"},
                    },
                    {
                        "name": "edit_file",
                        "args": {
                            "path": "workspace/writing/4246/story_4246.md",
                            "old": "old",
                            "new": "new",
                        },
                    },
                    {
                        "name": "grep_search",
                        "args": {"pattern": "new", "path": "workspace/writing/4246"},
                    },
                ],
            )

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    tool_loop = _FakeToolLoop()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=tool_loop,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Finish patching the workspace/writing/4246 manuscript.",
    )

    assert len(tool_loop.calls) == 2
    assert artifacts.content == "I patched the manuscript and verified the changed passage."
    assert "next step needs" not in artifacts.content


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_retries_web_research_absence_excuse() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                output = (
                    "I don't have a grounded 2channel/5ch report yet. In the evidence "
                    "available for this response, no browser, search, or fetch output "
                    "was supplied, so I can't truthfully name five verified threads."
                )
                return SimpleNamespace(
                    final_output=output,
                    messages=[*kwargs["messages"], {"role": "assistant", "content": output}],
                    tool_calls=[],
                )
            retry_prompt = kwargs["messages"][-1]["content"]
            assert "web_search" in retry_prompt
            assert "web_fetch" in retry_prompt
            assert "browser_start" in retry_prompt
            output = "I searched the web and found current 5ch thread evidence."
            return SimpleNamespace(
                final_output=output,
                messages=[
                    *kwargs["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "current 5ch thread discussion"}',
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "web_search",
                        "content": '{"results": [{"title": "5ch example thread", "url": "https://example.test/thread"}]}',
                    },
                    {"role": "assistant", "content": output},
                ],
                tool_calls=[
                    {"name": "web_search", "args": {"query": "current 5ch thread discussion"}}
                ],
            )

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    tool_loop = _FakeToolLoop()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=tool_loop,
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Research current 2channel and 5ch threads and give me a grounded report.",
    )

    assert len(tool_loop.calls) == 2
    assert artifacts.content == "I searched the web and found current 5ch thread evidence."
    assert "no browser, search, or fetch output" not in artifacts.content


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_reports_meaningful_generation_error() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _FakeToolLoop:
        async def run(self, **kwargs):
            raise EOFError()

    async def _fake_tool_context(session_id=None):
        return SimpleNamespace()

    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_FakeToolLoop(),
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        _build_tool_use_context=_fake_tool_context,
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Hello?",
    )

    assert artifacts.content == "[Error generating response: EOFError]"


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_handles_media_commentary_control_without_llm() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _UnexpectedToolLoop:
        async def run(self, **kwargs):
            raise AssertionError("media commentary control should not enter the model tool loop")

    class _FakeDesktopContext:
        def __init__(self):
            self.config = {
                "enabled": False,
                "media_commentary_mode_enabled": False,
                "live_transcription_enabled": False,
                "declared_task": None,
            }
            self.calls = []

        def _looks_like_media_commentary_request(self, text):
            lowered = text.lower()
            return "youtube" in lowered and "commentary" in lowered

        def configure(self, **updates):
            self.calls.append(updates)
            self.config.update(updates)
            return {"config": dict(self.config)}

    desktop_context = _FakeDesktopContext()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_UnexpectedToolLoop(),
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        desktop_context=desktop_context,
        _build_tool_use_context=lambda session_id=None: SimpleNamespace(),
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="media commentary on\nJust want to watch youtube with you.",
        user_meta={"conversation_actor": {"source": "api_chat"}},
    )

    assert artifacts.content.startswith("Media commentary mode is on.")
    assert desktop_context.calls
    assert desktop_context.config["enabled"] is True
    assert desktop_context.config["media_commentary_mode_enabled"] is True
    assert desktop_context.config["live_transcription_enabled"] is True
    assert artifacts.integrity_review["skip_reason"] == "desktop_context_control"
    assert "I will" not in artifacts.content
    assert "configuration only" in artifacts.content
    assert "spoken" in artifacts.content
    assert artifacts.loop_result.messages[-2]["tool_calls"][0]["function"]["name"] == "desktop_context_configure"


@pytest.mark.asyncio
async def test_execute_conversation_tool_loop_handles_body_double_plugin_off_without_llm() -> None:
    from opencas.runtime.conversation_turns import execute_conversation_tool_loop

    class _FakeBuilder:
        async def build(self, user_input, session_id=None):
            return SimpleNamespace(
                history=[],
                to_message_list=lambda: [{"role": "user", "content": user_input}],
            )

    class _UnexpectedToolLoop:
        async def run(self, **kwargs):
            raise AssertionError("body double off control should not enter the model tool loop")

    class _FakeDesktopContext:
        def __init__(self):
            self.config = {
                "enabled": True,
                "media_commentary_mode_enabled": True,
                "live_transcription_enabled": True,
                "tts_enabled": True,
                "play_audio": True,
                "proactive_video_commentary_enabled": True,
                "declared_task": "Watch YouTube with me.",
            }
            self.calls = []

        def configure(self, **updates):
            self.calls.append(updates)
            self.config.update(updates)
            return {"config": dict(self.config)}

    desktop_context = _FakeDesktopContext()
    runtime = SimpleNamespace(
        builder=_FakeBuilder(),
        tool_loop=_UnexpectedToolLoop(),
        modulators=SimpleNamespace(to_temperature=lambda: 0.4),
        scheduler=None,
        llm=_FakeIntegrityLLM(
            '{"needs_revision": false, "reasons": [], "revised_response": ""}'
        ),
        desktop_context=desktop_context,
        _build_tool_use_context=lambda session_id=None: SimpleNamespace(),
        _trace=lambda *args, **kwargs: None,
    )

    artifacts = await execute_conversation_tool_loop(
        runtime,
        session_id="s1",
        user_input="Turn off the Body Double plugin. Stop voice dictation too.",
        user_meta={"conversation_actor": {"source": "api_chat"}},
    )

    assert artifacts.content.startswith("Body Double mode is off.")
    assert desktop_context.calls
    assert desktop_context.config["enabled"] is False
    assert desktop_context.config["media_commentary_mode_enabled"] is False
    assert desktop_context.config["live_transcription_enabled"] is False
    assert desktop_context.config["tts_enabled"] is False
    assert desktop_context.config["play_audio"] is False
    assert desktop_context.config["proactive_video_commentary_enabled"] is False
    assert artifacts.integrity_review["skip_reason"] == "desktop_context_control"


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
    assert "Supplied tool/output evidence" in payload
    assert "desktop_context_capture" in payload
    assert "captured" in payload
