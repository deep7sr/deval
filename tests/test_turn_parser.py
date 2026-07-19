"""Unit tests for parse_conversation (the multi-turn contract parser).

Pure string/dict logic - no LLM, no network. Mirrors the style of
tests/test_parser.py for the single-turn parser.
"""

from guardrail.parser import (
    SKIP_NO_MARKER_IN_CONVERSATION,
    SKIP_NO_MESSAGES,
    SKIP_NO_USER_MESSAGE,
    SKIP_USER_CONTENT_NOT_STRING,
    parse_conversation,
)

MARKER = "--- Retrieved Evidence ---"


def _evidence(*lines):
    return {"role": "assistant", "content": MARKER + "\n" + "\n".join(lines)}


# --- skips ----------------------------------------------------------------


def test_none_messages_skips():
    result = parse_conversation(None)
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MESSAGES


def test_empty_messages_skips():
    result = parse_conversation([])
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MESSAGES


def test_no_user_message_skips():
    result = parse_conversation(
        [
            {"role": "system", "content": "You are helpful."},
            _evidence("Paris is the capital of France."),
        ]
    )
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_USER_MESSAGE


def test_no_marker_anywhere_skips():
    result = parse_conversation(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER_IN_CONVERSATION


def test_marker_with_no_evidence_lines_is_ignored():
    result = parse_conversation(
        [
            {"role": "assistant", "content": MARKER + "\n   \n"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    assert result.compliant is False
    assert result.skip_reason == SKIP_NO_MARKER_IN_CONVERSATION


def test_final_user_content_not_string_skips():
    result = parse_conversation(
        [
            _evidence("Paris is the capital of France."),
            {"role": "user", "content": [{"type": "text", "text": "capital?"}]},
        ]
    )
    assert result.compliant is False
    assert result.skip_reason == SKIP_USER_CONTENT_NOT_STRING


def test_conversation_ending_with_assistant_answer_skips():
    result = parse_conversation(
        [
            _evidence("Paris is the capital of France."),
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
    )
    assert result.compliant is False
    assert result.skip_reason == SKIP_USER_CONTENT_NOT_STRING


# --- compliant single exchange ---------------------------------------------


def test_single_exchange_marker_then_question():
    result = parse_conversation(
        [
            {"role": "system", "content": "You are helpful."},
            _evidence("Paris is the capital of France.", "Population ~2.1M."),
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    assert result.compliant is True
    assert result.turns == [
        {
            "role": "user",
            "content": "What is the capital of France?",
            "retrieval_context": None,
        }
    ]
    assert result.pending_context == [
        "Paris is the capital of France.",
        "Population ~2.1M.",
    ]
    assert result.input == "What is the capital of France?"


# --- compliant multi-turn ---------------------------------------------------


def test_multi_turn_attaches_context_to_following_answer():
    result = parse_conversation(
        [
            _evidence("The Eiffel Tower is in Paris."),
            {"role": "user", "content": "Where is the Eiffel Tower?"},
            {"role": "assistant", "content": "It is in Paris."},
            _evidence("The Eiffel Tower is 330 metres tall."),
            {"role": "user", "content": "How tall is it?"},
        ]
    )
    assert result.compliant is True
    assert result.turns == [
        {
            "role": "user",
            "content": "Where is the Eiffel Tower?",
            "retrieval_context": None,
        },
        {
            "role": "assistant",
            "content": "It is in Paris.",
            "retrieval_context": ["The Eiffel Tower is in Paris."],
        },
        {"role": "user", "content": "How tall is it?", "retrieval_context": None},
    ]
    assert result.pending_context == ["The Eiffel Tower is 330 metres tall."]
    assert result.input == "How tall is it?"


def test_answer_without_preceding_marker_has_no_context():
    result = parse_conversation(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            _evidence("Paris is the capital of France."),
            {"role": "user", "content": "Capital of France?"},
        ]
    )
    assert result.compliant is True
    assert result.turns[1] == {
        "role": "assistant",
        "content": "Hello!",
        "retrieval_context": None,
    }
    assert result.pending_context == ["Paris is the capital of France."]


def test_two_markers_without_answer_between_later_wins():
    result = parse_conversation(
        [
            _evidence("Old, superseded retrieval."),
            _evidence("Fresh retrieval line."),
            {"role": "user", "content": "Question?"},
        ]
    )
    assert result.compliant is True
    assert result.pending_context == ["Fresh retrieval line."]


def test_context_is_consumed_by_the_first_following_answer():
    result = parse_conversation(
        [
            _evidence("Fact A."),
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
    )
    assert result.compliant is True
    assert result.turns[1]["retrieval_context"] == ["Fact A."]
    # The second answer must NOT inherit the already-consumed context.
    assert result.turns[3]["retrieval_context"] is None
    assert result.pending_context is None


def test_non_string_history_content_is_skipped_as_turn():
    result = parse_conversation(
        [
            {"role": "assistant", "content": [{"type": "text", "text": "parts"}]},
            _evidence("Paris is the capital of France."),
            {"role": "user", "content": "Capital of France?"},
        ]
    )
    assert result.compliant is True
    assert len(result.turns) == 1
