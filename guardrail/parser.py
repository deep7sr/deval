"""Contract parser / adapter for the hallucination guardrail.

This module owns 100% of the marker-contract logic in isolation. It knows
NOTHING about DeepEval or LiteLLM - it only turns a raw ``messages`` array
into either:

  * a compliant ParseResult (``input`` + ``retrieval_context``), ready to be
    handed to the DeepEval wrapper, or
  * a non-compliant "skip" result explaining why the guardrail will not run.

It is purely deterministic string/substring matching - no LLM call, minimal
latency - and is exhaustively unit-tested, because this is where most of the
correctness risk lives.

The contract (locked in):
  1. Retrieved evidence goes in a ``role: assistant`` message.
  2. That message's ``content`` must START WITH the exact EVIDENCE_MARKER.
  3. Everything after the marker is evidence, one chunk per (non-blank) line.
  4. The question is the final ``role: user`` message.
  5. ``content`` must be a plain string (not a content-parts array) for both
     the evidence message and the final user message.
  6. Multi-turn: if several marker messages exist, only the one immediately
     preceding the final user message is used; older ones are ignored.
  7. No valid marker found -> guardrail does not run (silent skip).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import EVIDENCE_MARKER


@dataclass
class ParseResult:
    """Outcome of parsing a request's ``messages`` against the contract.

    When ``compliant`` is True, ``input`` and ``retrieval_context`` are
    populated and the guardrail should run. When False, ``skip_reason`` says
    why the request falls under rule 7 (silent skip / pass-through).
    """

    compliant: bool
    input: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    # Index of the assistant message whose marker was used - logged so teams
    # can self-diagnose "which evidence block did the guardrail check?".
    marker_message_index: Optional[int] = None
    # Machine-readable reason for a skip (also useful in structured logs).
    skip_reason: Optional[str] = None


# --- skip reason codes (stable strings for logging/alerting) -------------
SKIP_NO_MESSAGES = "no_messages"
SKIP_NO_USER_MESSAGE = "no_user_message"
SKIP_USER_CONTENT_NOT_STRING = "user_content_not_string"
SKIP_NO_MARKER = "no_marker_before_user"
SKIP_EMPTY_EVIDENCE = "marker_present_but_no_evidence"


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


def _extract_evidence_lines(content: str) -> List[str]:
    """Return the non-blank, stripped lines that follow the marker."""
    after_marker = content[len(EVIDENCE_MARKER):]
    lines = [line.strip() for line in after_marker.splitlines()]
    return [line for line in lines if line]


def parse_messages(messages: Optional[List[Dict[str, Any]]]) -> ParseResult:
    """Parse a raw ``messages`` array into a ParseResult.

    Deterministic and side-effect free. ``actual_output`` is NOT this module's
    concern - it comes from the LLM response and is supplied by the hook.
    """
    if not messages:
        return ParseResult(compliant=False, skip_reason=SKIP_NO_MESSAGES)

    # Rule 4: the question is the FINAL user message.
    last_user_index = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_index = i
            break

    if last_user_index is None:
        return ParseResult(compliant=False, skip_reason=SKIP_NO_USER_MESSAGE)

    user_content = messages[last_user_index].get("content")
    # Rule 5: the user message content must be a plain string.
    if not _is_string(user_content):
        return ParseResult(compliant=False, skip_reason=SKIP_USER_CONTENT_NOT_STRING)

    # Rule 6: scan BACKWARDS from the final user message for the nearest
    # qualifying assistant marker message. The first one found wins; older
    # marker blocks are ignored.
    for i in range(last_user_index - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        # Rule 5: evidence content must be a plain string (not content-parts).
        if not _is_string(content):
            continue
        # Rule 2: exact, case-sensitive, no-normalisation startswith match.
        if not content.startswith(EVIDENCE_MARKER):
            continue

        # Rule 3: everything after the marker is evidence, one chunk per line.
        evidence = _extract_evidence_lines(content)
        if not evidence:
            # Marker present but nothing to check against -> skip rather than
            # score against empty context (which would false-flag everything).
            return ParseResult(
                compliant=False,
                marker_message_index=i,
                skip_reason=SKIP_EMPTY_EVIDENCE,
            )

        return ParseResult(
            compliant=True,
            input=user_content,
            retrieval_context=evidence,
            marker_message_index=i,
        )

    # Rule 7: no valid marker before the final user message.
    return ParseResult(compliant=False, skip_reason=SKIP_NO_MARKER)
