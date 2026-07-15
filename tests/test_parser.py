"""Unit tests for the contract parser.

This is the highest-risk component (many edge cases, no LLM to smooth over
mistakes), so coverage here is deliberately exhaustive. No network, no LLM -
these run instantly and deterministically.
"""

from guardrail.parser import (
    parse_messages,
    parse_final_user_message,
    SKIP_NO_MESSAGES,
    SKIP_NO_USER_MESSAGE,
    SKIP_USER_CONTENT_NOT_STRING,
    SKIP_NO_MARKER,
    SKIP_EMPTY_EVIDENCE,
)

MARKER = "--- Retrieved Evidence ---"


# --- happy path ----------------------------------------------------------

def test_single_turn_compliant():
    messages = [
        {"role": "system", "content": "You are a factual assistant."},
        {
            "role": "assistant",
            "content": (
                MARKER
                + "\nAll customers are eligible for a 30 day full refund at no extra cost."
                + "\nRefunds are processed within 5 business days."
            ),
        },
        {"role": "user", "content": "What is the refund policy?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is True
    assert result.input == "What is the refund policy?"
    assert result.retrieval_context == [
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Refunds are processed within 5 business days.",
    ]
    assert result.marker_message_index == 1


def test_blank_lines_between_evidence_are_dropped():
    messages = [
        {"role": "assistant", "content": MARKER + "\n\nfact one\n\n\nfact two\n"},
        {"role": "user", "content": "q?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is True
    assert result.retrieval_context == ["fact one", "fact two"]


def test_evidence_lines_are_stripped():
    messages = [
        {"role": "assistant", "content": MARKER + "\n   padded fact   \n\ttabbed fact\t"},
        {"role": "user", "content": "q?"},
    ]
    result = parse_messages(messages)
    assert result.retrieval_context == ["padded fact", "tabbed fact"]


# --- multi-turn "nearest preceding marker wins" (rule 6) -----------------

def test_multi_turn_uses_nearest_preceding_marker():
    messages = [
        {"role": "user", "content": "What's your return policy?"},
        {"role": "assistant", "content": MARKER + "\nAll customers get a 30 day full refund."},
        {"role": "assistant", "content": "You get a full refund within 30 days."},
        {"role": "user", "content": "What about the warranty on premium plans?"},
        {"role": "assistant", "content": MARKER + "\nPremium plan customers get a 2-year extended warranty."},
        {"role": "user", "content": "Does that cover accidental damage too?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is True
    # The stale refund evidence must be ignored; only the warranty block used.
    assert result.retrieval_context == [
        "Premium plan customers get a 2-year extended warranty."
    ]
    assert result.marker_message_index == 4
    assert result.input == "Does that cover accidental damage too?"


def test_marker_only_after_final_user_is_ignored():
    # A marker that appears AFTER the last user message must not be used.
    messages = [
        {"role": "user", "content": "q?"},
        {"role": "assistant", "content": MARKER + "\nsome evidence"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER


# --- malformed marker -> treated as "no marker" (rule 2, rule 7) ---------

def test_typo_marker_is_not_recognised():
    messages = [
        {"role": "assistant", "content": "--- Retreived Evidence ---\nfact"},
        {"role": "user", "content": "q?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER


def test_leading_whitespace_before_marker_is_not_recognised():
    messages = [
        {"role": "assistant", "content": " " + MARKER + "\nfact"},
        {"role": "user", "content": "q?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER


def test_marker_in_wrong_role_is_ignored():
    # Same marker text but in a user/system message must not qualify.
    messages = [
        {"role": "system", "content": MARKER + "\nfact"},
        {"role": "user", "content": MARKER + "\nfact"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER


# --- content-parts array must not qualify (rule 5) -----------------------

def test_evidence_content_as_list_is_ignored():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": MARKER + "\nfact"}],
        },
        {"role": "user", "content": "q?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER


def test_user_content_as_list_is_skipped():
    messages = [
        {"role": "assistant", "content": MARKER + "\nfact"},
        {"role": "user", "content": [{"type": "text", "text": "q?"}]},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_USER_CONTENT_NOT_STRING


# --- no marker at all (rule 7) -------------------------------------------

def test_no_marker_skips():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the refund policy?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER


# --- degenerate inputs ---------------------------------------------------

def test_empty_messages_skips():
    assert parse_messages([]).skip_reason == SKIP_NO_MESSAGES
    assert parse_messages(None).skip_reason == SKIP_NO_MESSAGES


def test_no_user_message_skips():
    messages = [{"role": "assistant", "content": MARKER + "\nfact"}]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_USER_MESSAGE


def test_marker_present_but_no_evidence_skips():
    messages = [
        {"role": "assistant", "content": MARKER + "\n   \n\n"},
        {"role": "user", "content": "q?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_EMPTY_EVIDENCE
    assert result.marker_message_index == 0


def test_marker_with_no_trailing_content_skips():
    messages = [
        {"role": "assistant", "content": MARKER},
        {"role": "user", "content": "q?"},
    ]
    result = parse_messages(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_EMPTY_EVIDENCE


# --- parse_final_user_message (marker-free extraction for relevancy) ------

def test_final_user_message_no_marker_needed():
    # Answer Relevancy needs only the question - no evidence marker required.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result = parse_final_user_message(messages)
    assert result.compliant is True
    assert result.input == "What is the capital of France?"
    assert result.skip_reason is None


def test_final_user_message_picks_last_user_turn():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": "the real question"},
    ]
    result = parse_final_user_message(messages)
    assert result.compliant is True
    assert result.input == "the real question"


def test_final_user_message_ignores_marker_content():
    # Even when an evidence marker is present, relevancy just wants the question.
    messages = [
        {"role": "assistant", "content": MARKER + "\nsome evidence line"},
        {"role": "user", "content": "why?"},
    ]
    result = parse_final_user_message(messages)
    assert result.compliant is True
    assert result.input == "why?"


def test_final_user_message_empty_messages_skips():
    result = parse_final_user_message([])
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MESSAGES


def test_final_user_message_no_user_skips():
    messages = [{"role": "system", "content": "only a system prompt"}]
    result = parse_final_user_message(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_USER_MESSAGE


def test_final_user_message_non_string_content_skips():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "parts array"}]},
    ]
    result = parse_final_user_message(messages)
    assert result.compliant is False
    assert result.skip_reason == SKIP_USER_CONTENT_NOT_STRING
