"""DeepEval invocation wrapper (grounding evaluator).

Takes only the parser's clean output plus the model's actual_output, runs
DeepEval's FaithfulnessMetric through the Groq judge, and returns a compact
verdict: score, pass/fail, the human-readable reason, and - crucially for the
remediation retry loop - the specific unsupported claims.

This module knows nothing about message arrays or the marker contract; its
world begins once input / actual_output / retrieval_context already exist as
clean fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from . import config
from .groq_judge import GroqJudge


@dataclass
class GroundingVerdict:
    """Result of a single grounding evaluation."""

    score: float
    passed: bool
    reason: str
    # Claims from actual_output that the judge marked "no" (unsupported /
    # contradicted), each paired with the judge's reason. Fed back into the
    # retry prompt so the model gets targeted correction, not "try harder".
    unsupported_claims: List[str] = field(default_factory=list)


class GroundingEvaluator:
    """Wraps FaithfulnessMetric. The judge (and its litellm client) is created
    once and reused; a fresh metric object is used per evaluation so no
    per-request measurement state is ever shared across concurrent requests.
    """

    def __init__(self, judge_model: Optional[str] = None, threshold: Optional[float] = None):
        self._judge = GroqJudge(judge_model or config.JUDGE_MODEL)
        self._threshold = (
            threshold if threshold is not None else config.FAITHFULNESS_THRESHOLD
        )

    def _make_metric(self, async_mode: bool) -> FaithfulnessMetric:
        return FaithfulnessMetric(
            threshold=self._threshold,
            model=self._judge,
            include_reason=True,
            async_mode=async_mode,
        )

    def evaluate(
        self, input: str, actual_output: str, retrieval_context: List[str]
    ) -> GroundingVerdict:
        metric = self._make_metric(async_mode=False)
        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
            retrieval_context=list(retrieval_context),
        )
        metric.measure(test_case)
        return self._build_verdict(metric)

    async def a_evaluate(
        self, input: str, actual_output: str, retrieval_context: List[str]
    ) -> GroundingVerdict:
        metric = self._make_metric(async_mode=True)
        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
            retrieval_context=list(retrieval_context),
        )
        await metric.a_measure(test_case)
        return self._build_verdict(metric)

    def _build_verdict(self, metric: FaithfulnessMetric) -> GroundingVerdict:
        unsupported: List[str] = []
        claims = getattr(metric, "claims", None) or []
        verdicts = getattr(metric, "verdicts", None) or []
        # claims[i] corresponds to verdicts[i]; a "no" verdict means the claim
        # is not supported by (or contradicts) the retrieval context. If the
        # judge returned a different number of verdicts than claims, the
        # pairing is unreliable - fall back to the verdicts' own reasons so a
        # wrong claim is never labelled as the unsupported one.
        if len(claims) != len(verdicts):
            for verdict in verdicts:
                if str(getattr(verdict, "verdict", "")).strip().lower() == "no":
                    reason = getattr(verdict, "reason", None)
                    unsupported.append(reason or "unsupported claim")
        else:
            for claim, verdict in zip(claims, verdicts):
                if str(getattr(verdict, "verdict", "")).strip().lower() == "no":
                    reason = getattr(verdict, "reason", None)
                    unsupported.append(f"{claim} ({reason})" if reason else str(claim))

        score = metric.score if metric.score is not None else 0.0
        return GroundingVerdict(
            score=score,
            passed=score >= self._threshold,
            reason=metric.reason or "",
            unsupported_claims=unsupported,
        )
