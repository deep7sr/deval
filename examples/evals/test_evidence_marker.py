"""Unit tests for the deterministic evidence-marker parser (Path B core).

Pure Python, no MLflow/Groq/network needed — runnable anywhere:

    python examples/evals/test_evidence_marker.py     # tiny built-in runner
    pytest examples/evals/test_evidence_marker.py     # or via pytest

Most of the faithfulness correctness risk lives in this parser (many contract
edge cases), so it is tested in isolation from the judge.
"""

from evidence_marker import EVIDENCE_MARKER, parse_evidence

MARKER = EVIDENCE_MARKER


def _ev(*lines: str) -> str:
    return MARKER + "\n" + "\n".join(lines)


def test_single_turn_marker_present():
    messages = [
        {"role": "system", "content": "Answer only from the evidence."},
        {"role": "assistant", "content": _ev("Refunds within 30 days.", "Processed in 5 business days.")},
        {"role": "user", "content": "What is the refund policy?"},
    ]
    parsed = parse_evidence(messages)
    assert parsed is not None
    assert parsed.question == "What is the refund policy?"
    assert parsed.evidence == ["Refunds within 30 days.", "Processed in 5 business days."]
    assert parsed.evidence_message_index == 1


def test_no_marker_returns_none():
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "It is Paris."},
        {"role": "user", "content": "Are you sure?"},
    ]
    assert parse_evidence(messages) is None


def test_marker_typo_or_bad_spacing_is_not_a_marker():
    for bad in ["--- Retreived Evidence ---", "---Retrieved Evidence---", " --- Retrieved Evidence ---"]:
        messages = [
            {"role": "assistant", "content": bad + "\nSome text."},
            {"role": "user", "content": "Question?"},
        ]
        assert parse_evidence(messages) is None, bad


def test_multi_turn_nearest_preceding_marker_wins():
    messages = [
        {"role": "user", "content": "Return policy?"},
        {"role": "assistant", "content": _ev("Full refund within 30 days.")},
        {"role": "assistant", "content": "You get a full refund within 30 days."},
        {"role": "user", "content": "Warranty on premium plans?"},
        {"role": "assistant", "content": _ev("Premium plans get a 2-year extended warranty.")},
        {"role": "user", "content": "Does that cover accidental damage?"},
    ]
    parsed = parse_evidence(messages)
    assert parsed is not None
    # The nearest (warranty) block must win; the stale refund block is ignored.
    assert parsed.evidence == ["Premium plans get a 2-year extended warranty."]
    assert parsed.evidence_message_index == 4
    assert parsed.question == "Does that cover accidental damage?"


def test_marker_after_final_user_is_ignored():
    # A marker message that comes AFTER the final user question is not preceding
    # evidence and must not be used.
    messages = [
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": _ev("Trailing, not preceding.")},
    ]
    # final user is index 0; nothing precedes it -> skip.
    assert parse_evidence(messages) is None


def test_content_parts_array_is_non_compliant():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": _ev("As a list.")}]},
        {"role": "user", "content": "Question?"},
    ]
    assert parse_evidence(messages) is None


def test_marker_with_no_evidence_lines_is_skipped():
    messages = [
        {"role": "assistant", "content": MARKER + "\n   \n"},
        {"role": "user", "content": "Question?"},
    ]
    assert parse_evidence(messages) is None


def test_user_role_evidence_is_accepted():
    messages = [
        {"role": "user", "content": _ev("Evidence delivered as a user message.")},
        {"role": "user", "content": "Question?"},
    ]
    parsed = parse_evidence(messages)
    assert parsed is not None
    assert parsed.evidence == ["Evidence delivered as a user message."]


def test_blank_lines_between_evidence_are_dropped():
    messages = [
        {"role": "assistant", "content": _ev("Fact A.", "", "  ", "Fact B.")},
        {"role": "user", "content": "Question?"},
    ]
    parsed = parse_evidence(messages)
    assert parsed is not None
    assert parsed.evidence == ["Fact A.", "Fact B."]


def test_empty_or_malformed_inputs():
    assert parse_evidence([]) is None
    assert parse_evidence(None) is None  # type: ignore[arg-type]
    assert parse_evidence([{"role": "system", "content": "hi"}]) is None
    assert parse_evidence([{"role": "user", "content": ""}]) is None


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run() else 0)
