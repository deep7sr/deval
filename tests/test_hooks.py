"""Integration-style tests for the hooks' _run logic (no network).

Uses real litellm.ModelResponse objects and a stubbed evaluator, so the skip /
block / remediate decision paths are exercised exactly as the proxy would drive
them - minus the judge and served-model calls.
"""

import contextvars

import pytest
import litellm

from guardrail import config
from guardrail.hook import HallucinationGuardrail
from guardrail.relevancy_hook import AnswerRelevancyGuardrail
from guardrail.contextual_relevancy_hook import ContextualRelevancyGuardrail
from guardrail.grounding import GroundingVerdict
from guardrail.relevancy import RelevancyVerdict
from guardrail.contextual_relevancy import ContextualRelevancyVerdict
from guardrail import remediation

MARKER = config.EVIDENCE_MARKER


def _response(content="The refund window is 30 days."):
    return litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": content}}]
    )


def _marker_data():
    return {
        "model": "test-model",
        "messages": [
            {"role": "assistant", "content": MARKER + "\nRefund window is 30 days."},
            {"role": "user", "content": "What is the refund policy?"},
        ],
    }


class _FakeGroundingEvaluator:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    async def a_evaluate(self, input, actual_output, retrieval_context):
        self.calls += 1
        return self.verdict


class _FakeRelevancyEvaluator:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    async def a_evaluate(self, input, actual_output):
        self.calls += 1
        return self.verdict


class _FakeContextualEvaluator:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    async def a_evaluate(self, input, retrieval_context):
        self.calls += 1
        return self.verdict


def _grounding_hook(verdict, mode="block"):
    hook = HallucinationGuardrail(
        guardrail_name="hallucination-guardrail", event_hook="post_call"
    )
    hook.evaluator = _FakeGroundingEvaluator(verdict)
    hook.mode = mode
    return hook


@pytest.mark.asyncio
async def test_passing_verdict_leaves_response_untouched():
    hook = _grounding_hook(GroundingVerdict(1.0, True, "grounded"))
    resp = _response()
    out = await hook._run(_marker_data(), resp)
    assert out.choices[0].message.content == "The refund window is 30 days."
    assert hook.evaluator.calls == 1


@pytest.mark.asyncio
async def test_failing_verdict_block_mode_replaces_with_fallback():
    hook = _grounding_hook(GroundingVerdict(0.1, False, "ungrounded", ["bad claim"]))
    resp = _response("You also get a free phone.")
    out = await hook._run(_marker_data(), resp)
    assert out.choices[0].message.content == config.FALLBACK_MESSAGE


@pytest.mark.asyncio
async def test_hook_skips_while_in_remediation_context():
    # The recursion guard: if the hook is ever re-entered while a remediation
    # loop is running in this context, it must pass the response through
    # without scoring.
    hook = _grounding_hook(GroundingVerdict(0.0, False, "would block"))
    resp = _response()

    async def _call_inside_guard():
        token = remediation._IN_REMEDIATION.set(True)
        try:
            return await hook._run(_marker_data(), resp)
        finally:
            remediation._IN_REMEDIATION.reset(token)

    # Run in a copied context so the var manipulation stays local to the test.
    out = await contextvars.copy_context().run(_call_inside_guard)
    assert out.choices[0].message.content == "The refund window is 30 days."
    assert hook.evaluator.calls == 0


@pytest.mark.asyncio
async def test_forged_metadata_flag_does_not_bypass_guardrail():
    # A client putting the old retry flag into request metadata must NOT be
    # able to skip the check (that was a real bypass vector; the guard is a
    # ContextVar now).
    hook = _grounding_hook(GroundingVerdict(0.0, False, "ungrounded"))
    data = _marker_data()
    data["metadata"] = {"guardrail_retry": True, "guardrail_relevancy_retry": True}
    resp = _response("You also get a free phone.")
    out = await hook._run(data, resp)
    assert hook.evaluator.calls == 1
    assert out.choices[0].message.content == config.FALLBACK_MESSAGE


@pytest.mark.asyncio
async def test_hook_skips_content_already_replaced_by_another_guardrail():
    hook = _grounding_hook(GroundingVerdict(0.0, False, "would block"))
    resp = _response(config.CONTEXTUAL_RELEVANCY_FALLBACK_MESSAGE)
    out = await hook._run(_marker_data(), resp)
    assert hook.evaluator.calls == 0
    assert (
        out.choices[0].message.content
        == config.CONTEXTUAL_RELEVANCY_FALLBACK_MESSAGE
    )


@pytest.mark.asyncio
async def test_relevancy_hook_skips_fallback_content():
    hook = AnswerRelevancyGuardrail(
        guardrail_name="answer-relevancy-guardrail", event_hook="post_call"
    )
    hook.evaluator = _FakeRelevancyEvaluator(RelevancyVerdict(0.0, False, "off"))
    hook.mode = "block"
    resp = _response(config.FALLBACK_MESSAGE)
    data = {"model": "m", "messages": [{"role": "user", "content": "q?"}]}
    out = await hook._run(data, resp)
    assert hook.evaluator.calls == 0
    assert out.choices[0].message.content == config.FALLBACK_MESSAGE


@pytest.mark.asyncio
async def test_contextual_hook_observe_mode_never_alters_response():
    hook = ContextualRelevancyGuardrail(
        guardrail_name="contextual-relevancy-guardrail", event_hook="post_call"
    )
    hook.evaluator = _FakeContextualEvaluator(
        ContextualRelevancyVerdict(0.0, False, "irrelevant", ["off-topic chunk"])
    )
    hook.mode = "observe"
    resp = _response("Some answer.")
    out = await hook._run(_marker_data(), resp)
    assert hook.evaluator.calls == 1
    assert out.choices[0].message.content == "Some answer."


@pytest.mark.asyncio
async def test_contextual_hook_block_mode_replaces_with_fallback():
    hook = ContextualRelevancyGuardrail(
        guardrail_name="contextual-relevancy-guardrail", event_hook="post_call"
    )
    hook.evaluator = _FakeContextualEvaluator(
        ContextualRelevancyVerdict(0.0, False, "irrelevant", ["off-topic chunk"])
    )
    hook.mode = "block"
    resp = _response("Answer built on bad retrieval.")
    out = await hook._run(_marker_data(), resp)
    assert (
        out.choices[0].message.content
        == config.CONTEXTUAL_RELEVANCY_FALLBACK_MESSAGE
    )


def test_unknown_mode_falls_back_to_default_with_warning():
    saved = config.MODE
    try:
        config.MODE = "blok"  # typo'd env value, already normalised lowercase
        hook = HallucinationGuardrail(
            guardrail_name="hallucination-guardrail", event_hook="post_call"
        )
        assert hook.mode == "remediate"
    finally:
        config.MODE = saved
