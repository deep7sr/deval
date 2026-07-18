"""DeepEval invocation wrapper (answer-relevancy evaluator).

The metric-specific counterpart to ``grounding.py``. Takes only the parser's
clean output (the question) plus the model's actual_output, runs DeepEval's
AnswerRelevancyMetric through the shared LiteLLM judge, and returns a compact
verdict: score, pass/fail, the human-readable reason, and - crucially for the
remediation retry loop - the specific off-topic / irrelevant statements.

Answer Relevancy is reference-free and needs ONLY ``input`` + ``actual_output``
(no retrieval_context, no evidence marker). Its semantics differ from
faithfulness: it penalises statements in the answer that do not address the
question, rather than statements that contradict retrieved evidence.

This module knows nothing about message arrays or the marker contract; its
world begins once input / actual_output already exist as clean fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from . import config
from .groq_judge import LiteLLMJudge


@dataclass
class RelevancyVerdict:
    """Result of a single answer-relevancy evaluation."""

    score: float
    passed: bool
    reason: str
    # Statements extracted from actual_output that the judge marked "no" (not
    # relevant to the question), each paired with the judge's reason. Fed back
    # into the retry prompt so the model gets targeted correction, not "try
    # harder". "idk" verdicts are NOT collected - they count as relevant in the
    # metric's own score and are not actionable off-topic content.
    irrelevant_statements: List[str] = field(default_factory=list)


class RelevancyEvaluator:
    """Wraps AnswerRelevancyMetric. The judge (and its litellm client) is
    created once and reused; a fresh metric object is used per evaluation so no
    per-request measurement state is ever shared across concurrent requests.
    """

    def __init__(
        self, judge_model: Optional[str] = None, threshold: Optional[float] = None
    ):
        self._judge = LiteLLMJudge(judge_model or config.JUDGE_MODEL)
        self._threshold = (
            threshold if threshold is not None else config.ANSWER_RELEVANCY_THRESHOLD
        )

    def _make_metric(self, async_mode: bool) -> AnswerRelevancyMetric:
        return AnswerRelevancyMetric(
            threshold=self._threshold,
            model=self._judge,
            include_reason=True,
            async_mode=async_mode,
        )

    def evaluate(self, input: str, actual_output: str) -> RelevancyVerdict:
        metric = self._make_metric(async_mode=False)
        test_case = LLMTestCase(input=input, actual_output=actual_output)
        metric.measure(test_case)
        return self._build_verdict(metric)

    async def a_evaluate(self, input: str, actual_output: str) -> RelevancyVerdict:
        metric = self._make_metric(async_mode=True)
        test_case = LLMTestCase(input=input, actual_output=actual_output)
        await metric.a_measure(test_case)
        return self._build_verdict(metric)

    def _build_verdict(self, metric: AnswerRelevancyMetric) -> RelevancyVerdict:
        irrelevant: List[str] = []
        statements = getattr(metric, "statements", None) or []
        verdicts = getattr(metric, "verdicts", None) or []
        # statements[i] corresponds to verdicts[i]; a "no" verdict means the
        # statement does not address the question (off-topic / irrelevant). If
        # the judge returned a different number of verdicts than statements,
        # the pairing is unreliable - fall back to the verdicts' own reasons so
        # a wrong statement is never labelled as the irrelevant one.
        if len(statements) != len(verdicts):
            for verdict in verdicts:
                if str(getattr(verdict, "verdict", "")).strip().lower() == "no":
                    reason = getattr(verdict, "reason", None)
                    irrelevant.append(reason or "irrelevant statement")
        else:
            for statement, verdict in zip(statements, verdicts):
                if str(getattr(verdict, "verdict", "")).strip().lower() == "no":
                    reason = getattr(verdict, "reason", None)
                    irrelevant.append(
                        f"{statement} ({reason})" if reason else str(statement)
                    )

        score = metric.score if metric.score is not None else 0.0
        return RelevancyVerdict(
            score=score,
            passed=score >= self._threshold,
            reason=metric.reason or "",
            irrelevant_statements=irrelevant,
        )
