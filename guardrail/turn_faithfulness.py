"""DeepEval invocation wrapper (turn-faithfulness evaluator).

The conversational counterpart of ``grounding.py``. Takes only the parser's
clean output (ordered turns, each optionally carrying retrieval context) plus
the model's new answer already appended as the final assistant turn, runs
DeepEval's TurnFaithfulnessMetric through the shared LiteLLM judge, and returns
a compact verdict: score, pass/fail, the human-readable reason, and - for the
remediation retry loop - the specific unfaithful claims.

TurnFaithfulnessMetric evaluates a sliding window per unit interaction: for
each window it extracts truths from the window's retrieval context, extracts
claims from the assistant turns, checks each claim against the truths, and the
final score is the mean of the per-window scores. The stock metric does NOT
retain the per-window claims/verdicts after ``measure()`` (only score, reason
and a verbose log string), so a thin recording subclass captures the
InteractionFaithfulnessScore objects as they are produced - that is where the
actionable unfaithful-claim detail comes from.

This module knows nothing about message arrays or the marker contract; its
world begins once the turns already exist as clean dicts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepeval.metrics import TurnFaithfulnessMetric
from deepeval.metrics.turn_faithfulness.schema import FaithfulnessVerdict
from deepeval.test_case import ConversationalTestCase, Turn

from . import config
from .groq_judge import GroqJudge


@dataclass
class TurnFaithfulnessVerdict:
    """Result of a single conversation-level faithfulness evaluation."""

    score: float
    passed: bool
    reason: str
    # Claims from the assistant turns that the judge marked "no" (contradicting
    # the retrieval context), each paired with the judge's reason. Fed back
    # into the retry prompt so the model gets targeted correction. De-duplicated
    # across overlapping sliding windows.
    unfaithful_claims: List[str] = field(default_factory=list)
    # How many sliding windows the metric evaluated (observability).
    windows_evaluated: int = 0


class _RecordingTurnFaithfulnessMetric(TurnFaithfulnessMetric):
    """TurnFaithfulnessMetric that keeps the per-window interaction scores.

    The parent computes an InteractionFaithfulnessScore (claims, truths,
    verdicts, score, reason) per sliding window but only aggregates them into
    ``score``/``reason``/``verbose_logs``. Recording them here is what lets the
    guardrail report WHICH claims were unfaithful.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction_scores: List[Any] = []

    def _get_prompt(self, method, **kwargs):
        # deepeval resolves prompt templates by class name; keep borrowing the
        # parent's templates (PromptMixin documents template_class for this).
        kwargs.setdefault("template_class", TurnFaithfulnessMetric.__name__)
        return super()._get_prompt(method, **kwargs)

    @staticmethod
    def _coerce_verdicts(verdicts):
        """Normalise judge verdicts to FaithfulnessVerdict objects.

        With a custom string-returning judge (our LiteLLMJudge), deepeval
        4.1.0's generate_with_schema_and_extract returns the raw parsed JSON -
        a list of dicts - and the metric's own scoring then crashes on
        ``verdict.verdict``. Coerce dicts into the schema objects the metric
        expects; an unrecognised verdict value degrades to "idk" (uncounted
        by default) rather than failing the whole evaluation.
        """
        coerced = []
        for v in verdicts or []:
            if isinstance(v, dict):
                raw = str(v.get("verdict", "")).strip().lower()
                if raw not in ("yes", "no", "idk"):
                    raw = "idk"
                coerced.append(
                    FaithfulnessVerdict(verdict=raw, reason=v.get("reason"))
                )
            else:
                coerced.append(v)
        return coerced

    def _generate_verdicts(self, claims, truths, multimodal):
        return self._coerce_verdicts(
            super()._generate_verdicts(claims, truths, multimodal)
        )

    async def _a_generate_verdicts(self, claims, truths, multimodal):
        return self._coerce_verdicts(
            await super()._a_generate_verdicts(claims, truths, multimodal)
        )

    def _get_faithfulness_scores(self, turns_window, multimodal):
        scores = super()._get_faithfulness_scores(turns_window, multimodal)
        self.interaction_scores.extend(scores)
        return scores

    async def _a_get_faithfulness_scores(self, turns_window, multimodal):
        scores = await super()._a_get_faithfulness_scores(turns_window, multimodal)
        self.interaction_scores.extend(scores)
        return scores


class TurnFaithfulnessEvaluator:
    """Wraps TurnFaithfulnessMetric. The judge (and its litellm client) is
    created once and reused; a fresh metric object is used per evaluation so no
    per-request measurement state is ever shared across concurrent requests.
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        threshold: Optional[float] = None,
        window_size: Optional[int] = None,
    ):
        self._judge = GroqJudge(judge_model or config.JUDGE_MODEL)
        self._threshold = (
            threshold if threshold is not None else config.TURN_FAITHFULNESS_THRESHOLD
        )
        self._window_size = (
            window_size
            if window_size is not None
            else config.TURN_FAITHFULNESS_WINDOW_SIZE
        )

    def _make_metric(self, async_mode: bool) -> _RecordingTurnFaithfulnessMetric:
        return _RecordingTurnFaithfulnessMetric(
            threshold=self._threshold,
            model=self._judge,
            include_reason=True,
            async_mode=async_mode,
            window_size=self._window_size,
        )

    @staticmethod
    def _build_test_case(turns: List[Dict[str, Any]]) -> ConversationalTestCase:
        return ConversationalTestCase(
            turns=[
                Turn(
                    role=turn["role"],
                    content=turn["content"],
                    retrieval_context=turn.get("retrieval_context"),
                )
                for turn in turns
            ]
        )

    def evaluate(self, turns: List[Dict[str, Any]]) -> TurnFaithfulnessVerdict:
        metric = self._make_metric(async_mode=False)
        metric.measure(self._build_test_case(turns))
        return self._build_verdict(metric)

    async def a_evaluate(self, turns: List[Dict[str, Any]]) -> TurnFaithfulnessVerdict:
        metric = self._make_metric(async_mode=True)
        await metric.a_measure(self._build_test_case(turns))
        return self._build_verdict(metric)

    def _build_verdict(self, metric) -> TurnFaithfulnessVerdict:
        unfaithful: List[str] = []
        seen = set()
        interaction_scores = getattr(metric, "interaction_scores", None) or []
        for interaction in interaction_scores:
            claims = getattr(interaction, "claims", None) or []
            verdicts = getattr(interaction, "verdicts", None) or []
            # claims[i] corresponds to verdicts[i]; a "no" verdict means the
            # claim contradicts the truths extracted from the retrieval
            # context. Overlapping sliding windows re-judge the same claims,
            # so de-duplicate on the claim text itself.
            for claim, verdict in zip(claims, verdicts):
                if str(getattr(verdict, "verdict", "")).strip().lower() != "no":
                    continue
                if str(claim) in seen:
                    continue
                seen.add(str(claim))
                reason = getattr(verdict, "reason", None)
                unfaithful.append(f"{claim} ({reason})" if reason else str(claim))

        score = metric.score if metric.score is not None else 0.0
        return TurnFaithfulnessVerdict(
            score=score,
            passed=score >= self._threshold,
            reason=metric.reason or "",
            unfaithful_claims=unfaithful,
            windows_evaluated=len(interaction_scores),
        )
