"""Self-correction retry loop (remediation).

Pure orchestration, deliberately decoupled from LiteLLM and DeepEval so it can
be unit-tested with fakes. Given an ungrounded response, it re-prompts the SAME
model with the specific unsupported claims, re-scores, and repeats up to a
bound - stopping early on success, and falling back to a safe message if all
attempts still fail (or the time budget is exceeded).

The two collaborators are injected as async callables:
  * ``evaluate(question, output, retrieval_context) -> GroundingVerdict``
  * ``regenerate(correction_prompt, previous_output) -> str``
"""

import contextvars
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from .grounding import GroundingVerdict

# --- outcome codes (stable strings for logging) --------------------------
OUTCOME_PASSED_FIRST_TRY = "passed_first_try"
OUTCOME_PASSED_AFTER_RETRY = "passed_after_retry"
OUTCOME_FALLBACK = "fallback"

# Recursion guard for the retry loop. Retry regenerations go through the proxy
# Router, which does not re-enter guardrail hooks - but as defence in depth the
# hooks short-circuit if they are ever invoked while a remediation is running
# in the same request context. A ContextVar is used (not request metadata)
# because metadata is client-supplied: a caller could forge a metadata flag in
# the request body and silently bypass the guardrail. A ContextVar cannot be
# influenced from outside the process, and each request task gets its own
# context, so concurrent requests never see each other's flag.
_IN_REMEDIATION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "guardrail_in_remediation", default=False
)


def in_remediation() -> bool:
    """True while a guardrail remediation loop is running in this context."""
    return _IN_REMEDIATION.get()


@dataclass
class RemediationResult:
    final_output: str
    passed: bool
    attempts: int  # number of retry regenerations actually performed
    outcome: str
    final_verdict: Optional[GroundingVerdict]


def build_correction_prompt(unsupported_claims: List[str]) -> str:
    """Turn the judge's unsupported-claim list into a targeted correction."""
    if unsupported_claims:
        bullets = "\n".join(f"- {claim}" for claim in unsupported_claims)
        problem = (
            "Your previous answer contained statements that are NOT supported "
            "by the retrieved evidence:\n" + bullets + "\n\n"
        )
    else:
        problem = (
            "Your previous answer contained statements that are not supported "
            "by the retrieved evidence.\n\n"
        )
    return (
        problem
        + "Answer the user's question again using ONLY the retrieved evidence "
        "provided earlier in this conversation. Do not add, infer, or assume "
        "any fact that is not explicitly stated in that evidence."
    )


def build_relevancy_correction_prompt(irrelevant_statements: List[str]) -> str:
    """Turn the judge's irrelevant-statement list into a targeted correction.

    The relevancy counterpart to ``build_correction_prompt``: instead of "stick
    to the evidence", it says "answer the question directly and drop the
    off-topic content".
    """
    if irrelevant_statements:
        bullets = "\n".join(f"- {stmt}" for stmt in irrelevant_statements)
        problem = (
            "Your previous answer included statements that do NOT directly "
            "address the user's question:\n" + bullets + "\n\n"
        )
    else:
        problem = (
            "Your previous answer did not directly address the user's "
            "question.\n\n"
        )
    return (
        problem
        + "Answer the user's question again, directly and concisely. Address "
        "exactly what was asked and leave out any off-topic, tangential, or "
        "irrelevant content."
    )


def _default_build_prompt(verdict: "GroundingVerdict") -> str:
    """Default prompt builder (faithfulness): correct unsupported claims."""
    return build_correction_prompt(verdict.unsupported_claims)


async def remediate(
    *,
    question: str,
    retrieval_context: List[str],
    original_output: str,
    initial_verdict: GroundingVerdict,
    evaluate: Callable[[str, str, List[str]], Awaitable[GroundingVerdict]],
    regenerate: Callable[[str, str], Awaitable[str]],
    max_retries: int,
    time_budget: float,
    fallback_message: str,
    build_prompt: Optional[Callable[[Any], str]] = None,
    on_event: Optional[Callable[..., None]] = None,
) -> RemediationResult:
    # Which correction prompt to emit each retry. Defaults to the faithfulness
    # prompt (unsupported claims); the relevancy guardrail injects its own
    # (irrelevant statements). This is the ONE metric-specific bit of the loop -
    # everything else (early stop, retry counting, budget, fallback) is shared.
    make_prompt = build_prompt if build_prompt is not None else _default_build_prompt

    def emit(event: str, **kwargs):
        if on_event is not None:
            on_event(event, **kwargs)

    # Already grounded - nothing to do.
    if initial_verdict.passed:
        return RemediationResult(
            final_output=original_output,
            passed=True,
            attempts=0,
            outcome=OUTCOME_PASSED_FIRST_TRY,
            final_verdict=initial_verdict,
        )

    start = time.monotonic()
    current_output = original_output
    current_verdict = initial_verdict
    attempts = 0

    guard_token = _IN_REMEDIATION.set(True)
    try:
        for attempt in range(1, max_retries + 1):
            if time.monotonic() - start > time_budget:
                emit("time_budget_exceeded", attempt=attempt)
                break

            # The initial verdict already failed, so the original output must
            # NOT reach the user. If a retry's regeneration or re-scoring call
            # errors (rate limit, judge outage, ...), swallowing the attempt
            # and proceeding to the fallback is the safe direction - letting
            # the exception escape would hit the hook's fail-open handler and
            # deliver the known-bad answer.
            try:
                correction = make_prompt(current_verdict)
                regenerated = await regenerate(correction, current_output)
                attempts = attempt
                new_verdict = await evaluate(
                    question, regenerated, retrieval_context
                )
            except Exception as exc:
                attempts = attempt
                emit("retry_error", attempt=attempt, error=repr(exc))
                continue

            current_output = regenerated
            current_verdict = new_verdict
            emit(
                "retry",
                attempt=attempt,
                score=current_verdict.score,
                passed=current_verdict.passed,
            )

            if current_verdict.passed:
                return RemediationResult(
                    final_output=current_output,
                    passed=True,
                    attempts=attempt,
                    outcome=OUTCOME_PASSED_AFTER_RETRY,
                    final_verdict=current_verdict,
                )
    finally:
        _IN_REMEDIATION.reset(guard_token)

    # Still ungrounded after all attempts (or budget exceeded) -> safe fallback.
    return RemediationResult(
        final_output=fallback_message,
        passed=False,
        attempts=attempts,
        outcome=OUTCOME_FALLBACK,
        final_verdict=current_verdict,
    )
