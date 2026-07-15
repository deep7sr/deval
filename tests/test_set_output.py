"""Unit tests for the hooks' _set_output helper.

Verifies that when a guardrail overwrites the delivered answer (remediation or
fallback), it also clears any stale chain-of-thought (reasoning /
reasoning_content) so the response can't ship a trace that contradicts the new
content. Uses a plain fake response - no litellm, no network.
"""

from types import SimpleNamespace

from guardrail.hook import HallucinationGuardrail
from guardrail.relevancy_hook import AnswerRelevancyGuardrail
from guardrail.contextual_relevancy_hook import ContextualRelevancyGuardrail

import pytest

_HOOKS = [
    HallucinationGuardrail,
    AnswerRelevancyGuardrail,
    ContextualRelevancyGuardrail,
]


def _fake_response(content, **extra):
    message = SimpleNamespace(content=content, role="assistant", **extra)
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


@pytest.mark.parametrize("guardrail", _HOOKS)
def test_set_output_overwrites_content_and_clears_reasoning(guardrail):
    resp = _fake_response(
        "off-topic original answer",
        reasoning="the model's discarded chain-of-thought",
        reasoning_content="also discarded",
    )
    guardrail._set_output(resp, "corrected answer")
    msg = resp.choices[0].message
    assert msg.content == "corrected answer"
    assert msg.reasoning is None
    assert msg.reasoning_content is None


@pytest.mark.parametrize("guardrail", _HOOKS)
def test_set_output_without_reasoning_field_is_fine(guardrail):
    # A response that never carried a reasoning field must still be overwritten
    # cleanly (no crash from clearing an absent attribute).
    resp = _fake_response("original answer")
    guardrail._set_output(resp, "corrected answer")
    assert resp.choices[0].message.content == "corrected answer"
