"""Unit tests for the turn-faithfulness evaluator's verdict-building logic.

These do NOT call any model - they feed a fake, already-measured metric object
into TurnFaithfulnessEvaluator._build_verdict to verify score/pass/
unfaithful-claim extraction deterministically, plus the turns->test-case
conversion. The live end-to-end behaviour (actually calling the judge) is
exercised separately by scripts/step_turn_faithfulness_smoke.py.
"""

from types import SimpleNamespace

from guardrail.turn_faithfulness import (
    _RecordingTurnFaithfulnessMetric,
    TurnFaithfulnessEvaluator,
    TurnFaithfulnessVerdict,
)
from guardrail.turn_faithfulness_hook import TurnFaithfulnessGuardrail


def _verdict(verdict, reason=None):
    return SimpleNamespace(verdict=verdict, reason=reason)


def _interaction(claims, verdicts):
    return SimpleNamespace(claims=claims, verdicts=verdicts)


def _fake_metric(score, reason, interaction_scores):
    return SimpleNamespace(
        score=score, reason=reason, interaction_scores=interaction_scores
    )


def test_all_faithful_passes_with_no_unfaithful_claims():
    evaluator = TurnFaithfulnessEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=1.0,
        reason="All claims are grounded in the retrieval context.",
        interaction_scores=[
            _interaction(["The refund window is 30 days."], [_verdict("yes")])
        ],
    )
    result = evaluator._build_verdict(metric)
    assert isinstance(result, TurnFaithfulnessVerdict)
    assert result.score == 1.0
    assert result.passed is True
    assert result.unfaithful_claims == []
    assert result.windows_evaluated == 1


def test_unfaithful_claims_are_extracted_with_reasons():
    evaluator = TurnFaithfulnessEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.5,
        reason="One claim contradicts the context.",
        interaction_scores=[
            _interaction(
                ["Refund window is 30 days.", "The tower is made of wood."],
                [
                    _verdict("yes"),
                    _verdict("no", "The context says it is made of iron."),
                ],
            )
        ],
    )
    result = evaluator._build_verdict(metric)
    assert result.score == 0.5
    assert result.passed is False  # 0.5 < 0.7
    assert result.unfaithful_claims == [
        "The tower is made of wood. (The context says it is made of iron.)"
    ]


def test_claims_deduplicated_across_overlapping_windows():
    evaluator = TurnFaithfulnessEvaluator(threshold=0.7)
    repeated = _interaction(
        ["The tower is made of wood."],
        [_verdict("no", "Contradicts the context.")],
    )
    metric = _fake_metric(
        score=0.0, reason="Contradicted.", interaction_scores=[repeated, repeated]
    )
    result = evaluator._build_verdict(metric)
    assert result.windows_evaluated == 2
    assert result.unfaithful_claims == [
        "The tower is made of wood. (Contradicts the context.)"
    ]


def test_verdict_matching_is_case_insensitive():
    evaluator = TurnFaithfulnessEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=0.0,
        reason="Contradicted.",
        interaction_scores=[
            _interaction(
                ["The sky is green."],
                [_verdict("NO", "Context says the sky is blue.")],
            )
        ],
    )
    result = evaluator._build_verdict(metric)
    assert result.passed is False
    assert len(result.unfaithful_claims) == 1


def test_idk_verdicts_are_not_flagged():
    evaluator = TurnFaithfulnessEvaluator(threshold=0.7)
    metric = _fake_metric(
        score=1.0,
        reason="",
        interaction_scores=[
            _interaction(["Unverifiable aside."], [_verdict("idk")])
        ],
    )
    result = evaluator._build_verdict(metric)
    assert result.unfaithful_claims == []


def test_score_exactly_at_threshold_passes():
    evaluator = TurnFaithfulnessEvaluator(threshold=0.7)
    metric = _fake_metric(score=0.7, reason="", interaction_scores=[])
    result = evaluator._build_verdict(metric)
    assert result.passed is True


def test_none_score_treated_as_zero():
    evaluator = TurnFaithfulnessEvaluator(threshold=0.7)
    metric = _fake_metric(score=None, reason="", interaction_scores=[])
    result = evaluator._build_verdict(metric)
    assert result.score == 0.0
    assert result.passed is False


def test_coerce_verdicts_turns_dicts_into_schema_objects():
    # deepeval 4.1.0 hands back raw dicts when the judge is a custom
    # string-returning model; the metric's own scoring needs objects.
    coerced = _RecordingTurnFaithfulnessMetric._coerce_verdicts(
        [
            {"verdict": "yes"},
            {"verdict": " NO ", "reason": "contradicts"},
            {"verdict": "maybe?"},  # unrecognised -> degrades to idk
        ]
    )
    assert [v.verdict for v in coerced] == ["yes", "no", "idk"]
    assert coerced[1].reason == "contradicts"


def test_coerce_verdicts_passes_through_objects_and_empty():
    obj = SimpleNamespace(verdict="no", reason="r")
    assert _RecordingTurnFaithfulnessMetric._coerce_verdicts([obj]) == [obj]
    assert _RecordingTurnFaithfulnessMetric._coerce_verdicts(None) == []
    assert _RecordingTurnFaithfulnessMetric._coerce_verdicts([]) == []


def test_hook_full_turns_appends_answer_with_pending_context():
    parse = SimpleNamespace(
        turns=[
            {"role": "user", "content": "Q?", "retrieval_context": None},
        ],
        pending_context=["Fact A."],
    )
    turns = TurnFaithfulnessGuardrail._full_turns(parse, "The answer.")
    assert turns[-1] == {
        "role": "assistant",
        "content": "The answer.",
        "retrieval_context": ["Fact A."],
    }
    # The original parse result must not be mutated.
    assert len(parse.turns) == 1


def test_build_test_case_maps_turns_and_context():
    test_case = TurnFaithfulnessEvaluator._build_test_case(
        [
            {"role": "user", "content": "Where is the tower?", "retrieval_context": None},
            {
                "role": "assistant",
                "content": "In Paris.",
                "retrieval_context": ["The Eiffel Tower is in Paris."],
            },
        ]
    )
    assert [t.role for t in test_case.turns] == ["user", "assistant"]
    assert test_case.turns[0].retrieval_context is None
    assert test_case.turns[1].retrieval_context == ["The Eiffel Tower is in Paris."]
    assert test_case.turns[1].content == "In Paris."
