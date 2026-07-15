"""Unit tests for the self-correction retry loop.

No network / no models - the evaluate and regenerate collaborators are fakes,
so the loop's control flow (early stop, retry counting, fallback, time budget,
correction-prompt content) is verified deterministically.
"""

import pytest

from guardrail.grounding import GroundingVerdict
from guardrail.relevancy import RelevancyVerdict
from guardrail.remediation import (
    remediate,
    build_correction_prompt,
    build_relevancy_correction_prompt,
    OUTCOME_PASSED_FIRST_TRY,
    OUTCOME_PASSED_AFTER_RETRY,
    OUTCOME_FALLBACK,
)


def _verdict(score, passed, claims=None):
    return GroundingVerdict(
        score=score, passed=passed, reason="", unsupported_claims=claims or []
    )


async def _run(initial_verdict, evaluate, regenerate, max_retries=3, time_budget=30.0):
    return await remediate(
        question="What is the refund policy?",
        retrieval_context=["30 day refund."],
        original_output="original answer",
        initial_verdict=initial_verdict,
        evaluate=evaluate,
        regenerate=regenerate,
        max_retries=max_retries,
        time_budget=time_budget,
        fallback_message="FALLBACK",
    )


@pytest.mark.asyncio
async def test_passes_first_try_does_not_regenerate():
    calls = []

    async def regenerate(correction, prev):
        calls.append(prev)
        return "should not happen"

    async def evaluate(q, o, c):
        raise AssertionError("evaluate should not be called on a first-try pass")

    result = await _run(_verdict(1.0, True), evaluate, regenerate)
    assert result.outcome == OUTCOME_PASSED_FIRST_TRY
    assert result.final_output == "original answer"
    assert result.attempts == 0
    assert calls == []


@pytest.mark.asyncio
async def test_passes_after_first_retry_stops_early():
    regen_calls = []

    async def regenerate(correction, prev):
        regen_calls.append((correction, prev))
        return "corrected answer"

    async def evaluate(q, o, c):
        return _verdict(1.0, True)  # the retry is grounded

    result = await _run(_verdict(0.2, False, ["bad claim"]), evaluate, regenerate)
    assert result.outcome == OUTCOME_PASSED_AFTER_RETRY
    assert result.attempts == 1
    assert result.final_output == "corrected answer"
    assert len(regen_calls) == 1
    # regeneration receives the previous (ungrounded) output to correct.
    assert regen_calls[0][1] == "original answer"
    # and the correction prompt carries the specific unsupported claim.
    assert "bad claim" in regen_calls[0][0]


@pytest.mark.asyncio
async def test_all_retries_fail_falls_back():
    regen_calls = []

    async def regenerate(correction, prev):
        regen_calls.append(prev)
        return "still ungrounded"

    async def evaluate(q, o, c):
        return _verdict(0.1, False, ["still bad"])

    result = await _run(_verdict(0.1, False, ["bad"]), evaluate, regenerate, max_retries=3)
    assert result.outcome == OUTCOME_FALLBACK
    assert result.final_output == "FALLBACK"
    assert result.attempts == 3
    assert len(regen_calls) == 3
    assert result.passed is False


@pytest.mark.asyncio
async def test_feedback_chains_previous_output_each_retry():
    seen_prev = []
    outputs = iter(["retry-1", "retry-2", "retry-3"])

    async def regenerate(correction, prev):
        seen_prev.append(prev)
        return next(outputs)

    async def evaluate(q, o, c):
        return _verdict(0.0, False, [o])  # never passes

    await _run(_verdict(0.0, False, ["orig-bad"]), evaluate, regenerate, max_retries=3)
    # Each retry corrects the immediately-preceding output.
    assert seen_prev == ["original answer", "retry-1", "retry-2"]


@pytest.mark.asyncio
async def test_time_budget_exceeded_short_circuits_to_fallback():
    calls = []

    async def regenerate(correction, prev):
        calls.append(1)
        return "x"

    async def evaluate(q, o, c):
        return _verdict(0.0, False)

    # Negative budget => elapsed always exceeds it => no regeneration attempted.
    result = await _run(
        _verdict(0.1, False), evaluate, regenerate, max_retries=3, time_budget=-1.0
    )
    assert result.outcome == OUTCOME_FALLBACK
    assert result.attempts == 0
    assert calls == []


def test_correction_prompt_with_claims():
    prompt = build_correction_prompt(["claim A", "claim B"])
    assert "claim A" in prompt
    assert "claim B" in prompt
    assert "ONLY" in prompt


def test_correction_prompt_without_claims_is_still_valid():
    prompt = build_correction_prompt([])
    assert "not supported" in prompt
    assert "ONLY" in prompt


# --- relevancy correction prompt + injectable build_prompt ---------------

def test_relevancy_correction_prompt_with_statements():
    prompt = build_relevancy_correction_prompt(["off-topic A", "tangent B"])
    assert "off-topic A" in prompt
    assert "tangent B" in prompt
    assert "directly" in prompt


def test_relevancy_correction_prompt_without_statements_is_still_valid():
    prompt = build_relevancy_correction_prompt([])
    assert "did not directly address" in prompt
    assert "directly" in prompt


@pytest.mark.asyncio
async def test_injected_build_prompt_is_used_over_default():
    # A relevancy-style verdict (no unsupported_claims attribute) must be able
    # to drive the shared loop via an injected build_prompt.
    seen_prompts = []

    async def regenerate(correction, prev):
        seen_prompts.append(correction)
        return "corrected"

    async def evaluate(q, o, c):
        return RelevancyVerdict(score=1.0, passed=True, reason="")

    initial = RelevancyVerdict(
        score=0.2, passed=False, reason="", irrelevant_statements=["off-topic X"]
    )

    result = await remediate(
        question="What is the capital of France?",
        retrieval_context=[],
        original_output="off-topic answer",
        initial_verdict=initial,
        evaluate=evaluate,
        regenerate=regenerate,
        max_retries=3,
        time_budget=30.0,
        fallback_message="FALLBACK",
        build_prompt=lambda v: build_relevancy_correction_prompt(
            v.irrelevant_statements
        ),
    )
    assert result.outcome == OUTCOME_PASSED_AFTER_RETRY
    assert len(seen_prompts) == 1
    assert "off-topic X" in seen_prompts[0]
