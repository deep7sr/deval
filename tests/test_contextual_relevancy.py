"""Unit tests for the contextual-relevancy evaluator's verdict-building logic.

These do NOT call any model - they feed a fake, already-measured metric object
into ContextualRelevancyEvaluator._build_verdict to verify score/pass/
irrelevant-context extraction deterministically. The metric stores its verdicts
as verdicts_list: a LIST of per-context-chunk groups, each with a .verdicts list
of objects carrying statement/verdict/reason together - a different shape from
the other two metrics, so it gets its own coverage.
"""

from types import SimpleNamespace

from guardrail.contextual_relevancy import (
    ContextualRelevancyEvaluator,
    ContextualRelevancyVerdict,
)


def _verdict(statement, verdict, reason=None):
    return SimpleNamespace(statement=statement, verdict=verdict, reason=reason)


def _group(*verdicts):
    # One verdict group per retrieval_context chunk.
    return SimpleNamespace(verdicts=list(verdicts))


def _fake_metric(score, reason, verdicts_list):
    return SimpleNamespace(score=score, reason=reason, verdicts_list=verdicts_list)


def test_all_relevant_passes_with_no_irrelevant_context():
    evaluator = ContextualRelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=1.0,
        reason="All retrieved context is relevant.",
        verdicts_list=[_group(_verdict("Paris is the capital of France.", "yes"))],
    )
    result = evaluator._build_verdict(metric)
    assert isinstance(result, ContextualRelevancyVerdict)
    assert result.score == 1.0
    assert result.passed is True
    assert result.irrelevant_context == []


def test_irrelevant_context_extracted_across_multiple_chunks():
    evaluator = ContextualRelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.5,
        reason="Half the context is off-topic.",
        verdicts_list=[
            _group(_verdict("Paris is the capital.", "yes")),
            _group(
                _verdict(
                    "France is known for cheese.",
                    "no",
                    "Unrelated to the capital question.",
                )
            ),
        ],
    )
    result = evaluator._build_verdict(metric)
    assert result.score == 0.5
    assert result.passed is False
    assert result.irrelevant_context == [
        "France is known for cheese. (Unrelated to the capital question.)"
    ]


def test_verdict_matching_is_case_insensitive_and_not_yes_counts():
    # Anything that isn't "yes" (no / idk / etc.) is irrelevant context.
    evaluator = ContextualRelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.0,
        reason="Off-topic.",
        verdicts_list=[
            _group(
                _verdict("The weather is nice.", "NO", "Irrelevant."),
                _verdict("Maybe about travel.", "idk", "Cannot tell."),
            )
        ],
    )
    result = evaluator._build_verdict(metric)
    assert result.passed is False
    assert len(result.irrelevant_context) == 2


def test_irrelevant_context_without_reason_still_captured():
    evaluator = ContextualRelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.0,
        reason="",
        verdicts_list=[_group(_verdict("Off-topic chunk.", "no", None))],
    )
    result = evaluator._build_verdict(metric)
    assert result.irrelevant_context == ["Off-topic chunk."]


def test_empty_verdicts_list_treated_as_no_irrelevant():
    evaluator = ContextualRelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(score=0.8, reason="", verdicts_list=[])
    result = evaluator._build_verdict(metric)
    assert result.irrelevant_context == []
    assert result.passed is True


def test_score_exactly_at_threshold_passes():
    evaluator = ContextualRelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(score=0.7, reason="", verdicts_list=[])
    result = evaluator._build_verdict(metric)
    assert result.passed is True


def test_none_score_treated_as_zero():
    evaluator = ContextualRelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(score=None, reason="", verdicts_list=[])
    result = evaluator._build_verdict(metric)
    assert result.score == 0.0
    assert result.passed is False
