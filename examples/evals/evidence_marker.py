"""Deterministic parser for the faithfulness "Retrieved Evidence" contract.

This is the Path B core (see MLFLOW_LITELLM_HANDOFF.md §9). It owns 100% of the
marker-contract logic in isolation: it knows nothing about MLflow, LiteLLM, or any
judge model. Given a raw OpenAI-style ``messages`` array it returns either a clean
``ParsedRequest`` (question + evidence lines) or ``None`` to signal "non-compliant,
skip the faithfulness check". Being pure string logic with no LLM call, it is
cheap, precise, and independently unit-testable (see ``test_evidence_marker.py``) —
which is exactly why production faithfulness should lean on it rather than asking
the judge to extract the marker itself (Path A).

The contract (locked in — see litellm-deepeval-guardrail-context.md and
MLFLOW_LITELLM_HANDOFF.md §5):

1. Retrieved evidence lives in a message whose ``content`` STARTS WITH the exact
   marker string ``--- Retrieved Evidence ---`` (this exact spelling/spacing).
2. Everything after the marker is the evidence, one chunk per line.
3. The current question is the final ``role: user`` message.
4. ``content`` must be a plain string (not a content-parts array) for both the
   evidence message and the final user message.
5. Multi-turn rule: if several marker-tagged messages exist, only the one
   immediately preceding the final user message is used; older ones are ignored.
6. No compliant marker found -> return ``None`` (silent skip, no fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The exact marker string. Note the correct spelling "Retrieved" (an early
# reference screenshot had a "Retreived" typo — do not reintroduce it).
EVIDENCE_MARKER = "--- Retrieved Evidence ---"

# Roles whose messages may carry the marker. The contract puts evidence in an
# ``assistant`` message; MLFLOW_LITELLM_HANDOFF.md §5 also allows ``user``. The
# final user message (the question) is always excluded from evidence candidacy.
EVIDENCE_ROLES = ("assistant", "user")


@dataclass
class ParsedRequest:
    """Clean, contract-compliant faithfulness inputs extracted from a request."""

    question: str
    evidence: list[str] = field(default_factory=list)
    # Index of the message the evidence was taken from (for logging/diagnostics —
    # teams ask "which message matched?").
    evidence_message_index: int | None = None

    @property
    def evidence_text(self) -> str:
        """Evidence as a single newline-joined block (handy for judge prompts)."""
        return "\n".join(self.evidence)


def _content_str(message: object) -> str | None:
    """Return a message's content iff it is a plain string, else ``None``.

    Content-parts arrays (``[{"type": "text", ...}]``) violate contract rule 4 and
    are treated as non-compliant for our purposes.
    """
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _final_user_index(messages: list) -> int | None:
    """Index of the last ``role: user`` message with plain-string content."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user" and _content_str(msg) is not None:
            return i
    return None


def parse_evidence(messages: list) -> ParsedRequest | None:
    """Apply the contract to a raw ``messages`` array.

    Returns a :class:`ParsedRequest` when a compliant marker message precedes the
    final user question, otherwise ``None`` (skip the faithfulness check).
    """
    if not isinstance(messages, list) or not messages:
        return None

    final_idx = _final_user_index(messages)
    if final_idx is None:
        return None  # no usable question -> nothing to judge

    question = _content_str(messages[final_idx])
    if question is None:
        return None
    question = question.strip()
    if not question:
        return None

    # Nearest-preceding-marker rule: scan backwards from just before the question
    # and take the first compliant marker message we hit.
    for i in range(final_idx - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") not in EVIDENCE_ROLES:
            continue
        content = _content_str(msg)
        if content is None or not content.startswith(EVIDENCE_MARKER):
            continue

        after = content[len(EVIDENCE_MARKER):]
        evidence = [line.strip() for line in after.splitlines() if line.strip()]
        if not evidence:
            # Marker present but no actual evidence lines -> not usable; keep the
            # marker semantics strict and treat as skip rather than an empty pass.
            return None
        return ParsedRequest(
            question=question,
            evidence=evidence,
            evidence_message_index=i,
        )

    return None
