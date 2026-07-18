"""Unit tests for the grounding evaluator's verdict-building logic.

These do NOT call any model - they feed a fake, already-measured metric object
into GroundingEvaluator._build_verdict to verify score/pass/unsupported-claim
extraction deterministically. The live end-to-end behaviour (actually calling
the Groq judge) is exercised separately by scripts/step4_smoke.py.
"""

from types import SimpleNamespace

from guardrail.grounding import GroundingEvaluator, GroundingVerdict


def _verdict(verdict, reason=None):
    return SimpleNamespace(verdict=verdict, reason=reason)


def _fake_metric(score, reason, claims, verdicts):
    return SimpleNamespace(
        score=score, reason=reason, claims=claims, verdicts=verdicts
    )


def test_all_supported_passes_with_no_unsupported_claims():
    evaluator = GroundingEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=1.0,
        reason="Everything is grounded.",
        claims=["The refund window is 30 days."],
        verdicts=[_verdict("yes")],
    )
    result = evaluator._build_verdict(metric)
    assert isinstance(result, GroundingVerdict)
    assert result.score == 1.0
    assert result.passed is True
    assert result.unsupported_claims == []


def test_unsupported_claims_are_extracted_with_reasons():
    evaluator = GroundingEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.5,
        reason="One claim is not supported.",
        claims=["Refund window is 30 days.", "Customers also get a free phone."],
        verdicts=[
            _verdict("yes"),
            _verdict("no", "The context never mentions a free phone."),
        ],
    )
    result = evaluator._build_verdict(metric)
    assert result.score == 0.5
    assert result.passed is False  # 0.5 < 0.7
    assert result.unsupported_claims == [
        "Customers also get a free phone. (The context never mentions a free phone.)"
    ]


def test_verdict_matching_is_case_insensitive():
    evaluator = GroundingEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.0,
        reason="Contradicted.",
        claims=["The sky is green."],
        verdicts=[_verdict("NO", "Context says the sky is blue.")],
    )
    result = evaluator._build_verdict(metric)
    assert result.passed is False
    assert len(result.unsupported_claims) == 1


def test_unsupported_claim_without_reason_still_captured():
    evaluator = GroundingEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.0,
        reason="",
        claims=["Unsupported thing."],
        verdicts=[_verdict("no", None)],
    )
    result = evaluator._build_verdict(metric)
    assert result.unsupported_claims == ["Unsupported thing."]


def test_score_exactly_at_threshold_passes():
    evaluator = GroundingEvaluator(threshold=0.7)
    metric = _fake_metric(score=0.7, reason="", claims=[], verdicts=[])
    result = evaluator._build_verdict(metric)
    assert result.passed is True


def test_none_score_treated_as_zero():
    evaluator = GroundingEvaluator(threshold=0.7)
    metric = _fake_metric(score=None, reason="", claims=[], verdicts=[])
    result = evaluator._build_verdict(metric)
    assert result.score == 0.0
    assert result.passed is False


def test_claims_verdicts_length_mismatch_uses_verdict_reasons():
    # If the judge returns a different number of verdicts than claims, the
    # index pairing is unreliable - the verdict's own reason is used instead of
    # (mis)labelling a claim.
    evaluator = GroundingEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.5,
        reason="",
        claims=["claim one", "claim two", "claim three"],
        verdicts=[_verdict("yes"), _verdict("no", "Contradicts the evidence.")],
    )
    result = evaluator._build_verdict(metric)
    assert result.unsupported_claims == ["Contradicts the evidence."]
