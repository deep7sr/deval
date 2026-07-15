"""Unit tests for the answer-relevancy evaluator's verdict-building logic.

These do NOT call any model - they feed a fake, already-measured metric object
into RelevancyEvaluator._build_verdict to verify score/pass/irrelevant-statement
extraction deterministically. The live end-to-end behaviour (actually calling
the judge) is exercised separately by scripts/step_relevancy_smoke.py.
"""

from types import SimpleNamespace

from guardrail.relevancy import RelevancyEvaluator, RelevancyVerdict


def _verdict(verdict, reason=None):
    return SimpleNamespace(verdict=verdict, reason=reason)


def _fake_metric(score, reason, statements, verdicts):
    return SimpleNamespace(
        score=score, reason=reason, statements=statements, verdicts=verdicts
    )


def test_all_relevant_passes_with_no_irrelevant_statements():
    evaluator = RelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=1.0,
        reason="Every statement addresses the question.",
        statements=["The capital of France is Paris."],
        verdicts=[_verdict("yes")],
    )
    result = evaluator._build_verdict(metric)
    assert isinstance(result, RelevancyVerdict)
    assert result.score == 1.0
    assert result.passed is True
    assert result.irrelevant_statements == []


def test_irrelevant_statements_are_extracted_with_reasons():
    evaluator = RelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.5,
        reason="One statement is off-topic.",
        statements=["The capital is Paris.", "France is famous for cheese."],
        verdicts=[
            _verdict("yes"),
            _verdict("no", "This does not address which city is the capital."),
        ],
    )
    result = evaluator._build_verdict(metric)
    assert result.score == 0.5
    assert result.passed is False  # 0.5 < 0.7
    assert result.irrelevant_statements == [
        "France is famous for cheese. "
        "(This does not address which city is the capital.)"
    ]


def test_verdict_matching_is_case_insensitive():
    evaluator = RelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.0,
        reason="Off-topic.",
        statements=["Let me tell you about the weather."],
        verdicts=[_verdict("NO", "Unrelated to the question asked.")],
    )
    result = evaluator._build_verdict(metric)
    assert result.passed is False
    assert len(result.irrelevant_statements) == 1


def test_idk_verdict_is_not_counted_as_irrelevant():
    # "idk" counts as relevant in the metric's own score and is not actionable
    # off-topic content, so it must NOT appear in irrelevant_statements.
    evaluator = RelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=1.0,
        reason="",
        statements=["Ambiguous statement."],
        verdicts=[_verdict("idk", "Cannot determine relevance.")],
    )
    result = evaluator._build_verdict(metric)
    assert result.irrelevant_statements == []


def test_irrelevant_statement_without_reason_still_captured():
    evaluator = RelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.0,
        reason="",
        statements=["Off-topic thing."],
        verdicts=[_verdict("no", None)],
    )
    result = evaluator._build_verdict(metric)
    assert result.irrelevant_statements == ["Off-topic thing."]


def test_score_exactly_at_threshold_passes():
    evaluator = RelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(score=0.7, reason="", statements=[], verdicts=[])
    result = evaluator._build_verdict(metric)
    assert result.passed is True


def test_none_score_treated_as_zero():
    evaluator = RelevancyEvaluator(threshold=0.7)
    metric = _fake_metric(score=None, reason="", statements=[], verdicts=[])
    result = evaluator._build_verdict(metric)
    assert result.score == 0.0
    assert result.passed is False
