"""Unit tests for the contextual-relevancy hook's _set_output helper.

Verifies that when the guardrail overwrites the delivered answer (block mode),
it also clears any stale chain-of-thought (reasoning / reasoning_content) so the
response can't ship a trace that contradicts the new content. Uses a plain fake
response - no litellm, no network.
"""

from types import SimpleNamespace

from guardrail.contextual_relevancy_hook import ContextualRelevancyGuardrail


def _fake_response(content, **extra):
    message = SimpleNamespace(content=content, role="assistant", **extra)
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def test_set_output_overwrites_content_and_clears_reasoning():
    resp = _fake_response(
        "answer built on irrelevant context",
        reasoning="the model's discarded chain-of-thought",
        reasoning_content="also discarded",
    )
    ContextualRelevancyGuardrail._set_output(resp, "fallback message")
    msg = resp.choices[0].message
    assert msg.content == "fallback message"
    assert msg.reasoning is None
    assert msg.reasoning_content is None


def test_set_output_without_reasoning_field_is_fine():
    # A response that never carried a reasoning field must still be overwritten
    # cleanly (no crash from clearing an absent attribute).
    resp = _fake_response("original answer")
    ContextualRelevancyGuardrail._set_output(resp, "fallback message")
    assert resp.choices[0].message.content == "fallback message"
