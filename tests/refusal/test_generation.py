from uuid import uuid4

import pytest

from opencas.refusal.generation import generate_refusal_response
from opencas.refusal.models import RefusalCategory, RefusalDecision


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "I cannot do that safely."}}]}


@pytest.mark.asyncio
async def test_refusal_generation_prompt_uses_agent_name() -> None:
    llm = FakeLLM()
    decision = RefusalDecision(
        request_id=uuid4(),
        refused=True,
        category=RefusalCategory.VALUE_VIOLATION,
        reasoning="privacy boundary",
    )

    generation = await generate_refusal_response(
        llm,
        request_text="read private data",
        decision=decision,
        agent_name="TestAgent",
    )

    assert generation.output == "I cannot do that safely."
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "You are TestAgent responding to the operator." in prompt
    assert "Preserve TestAgent's continuity" in prompt
