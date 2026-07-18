"""Unit tests for config helpers shared across the hooks."""

from guardrail import config


def test_fallback_messages_are_recognised():
    assert config.is_guardrail_fallback(config.FALLBACK_MESSAGE) is True
    assert (
        config.is_guardrail_fallback(config.ANSWER_RELEVANCY_FALLBACK_MESSAGE)
        is True
    )
    assert (
        config.is_guardrail_fallback(config.CONTEXTUAL_RELEVANCY_FALLBACK_MESSAGE)
        is True
    )


def test_ordinary_content_is_not_a_fallback():
    assert config.is_guardrail_fallback("Paris is the capital of France.") is False
    assert config.is_guardrail_fallback("") is False


def test_modes_are_normalised_lowercase():
    # Modes are trimmed + lowercased at load so "Block" / " BLOCK " work.
    assert config.MODE == config.MODE.strip().lower()
    assert config.ANSWER_RELEVANCY_MODE == config.ANSWER_RELEVANCY_MODE.strip().lower()
    assert (
        config.CONTEXTUAL_RELEVANCY_MODE
        == config.CONTEXTUAL_RELEVANCY_MODE.strip().lower()
    )
