"""DeepEval invocation wrapper (contextual-relevancy evaluator).

The third metric-specific evaluator, alongside grounding.py (faithfulness) and
relevancy.py (answer relevancy). It runs DeepEval's ContextualRelevancyMetric
through the shared LiteLLM judge and returns a compact verdict.

Contextual Relevancy grades the RETRIEVER, not the answer: given the question
and the retrieved context, it measures how much of that context is actually
relevant to the question. It needs ``input`` + ``retrieval_context`` only - it
does NOT look at ``actual_output``. Because it evaluates retrieval (which
happened upstream in the caller's RAG pipeline), a bad score cannot be fixed by
re-prompting the model - there is no remediation, only block / observe.

This module knows nothing about message arrays or the marker contract; its
world begins once input / retrieval_context already exist as clean fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from . import config
from .groq_judge import LiteLLMJudge


@dataclass
class ContextualRelevancyVerdict:
    """Result of a single contextual-relevancy evaluation."""

    score: float
    passed: bool
    reason: str
    # Statements from the retrieved context that the judge marked not relevant
    # to the question (verdict != "yes"), each paired with the judge's reason.
    # This is the actionable detail: which retrieved chunks were off-topic.
    # Logged (block/observe) so retrieval quality can be diagnosed; there is no
    # remediation loop to feed it into.
    irrelevant_context: List[str] = field(default_factory=list)


class ContextualRelevancyEvaluator:
    """Wraps ContextualRelevancyMetric. The judge (and its litellm client) is
    created once and reused; a fresh metric object is used per evaluation so no
    per-request measurement state is ever shared across concurrent requests.
    """

    def __init__(
        self, judge_model: Optional[str] = None, threshold: Optional[float] = None
    ):
        self._judge = LiteLLMJudge(judge_model or config.JUDGE_MODEL)
        self._threshold = (
            threshold
            if threshold is not None
            else config.CONTEXTUAL_RELEVANCY_THRESHOLD
        )

    def _make_metric(self, async_mode: bool) -> ContextualRelevancyMetric:
        return ContextualRelevancyMetric(
            threshold=self._threshold,
            model=self._judge,
            include_reason=True,
            async_mode=async_mode,
        )

    def evaluate(
        self, input: str, retrieval_context: List[str]
    ) -> ContextualRelevancyVerdict:
        metric = self._make_metric(async_mode=False)
        test_case = LLMTestCase(
            input=input,
            # ContextualRelevancyMetric does not use actual_output, but
            # LLMTestCase requires it to be a string; a placeholder is fine and
            # never influences the score.
            actual_output="",
            retrieval_context=list(retrieval_context),
        )
        metric.measure(test_case)
        return self._build_verdict(metric)

    async def a_evaluate(
        self, input: str, retrieval_context: List[str]
    ) -> ContextualRelevancyVerdict:
        metric = self._make_metric(async_mode=True)
        test_case = LLMTestCase(
            input=input,
            actual_output="",
            retrieval_context=list(retrieval_context),
        )
        await metric.a_measure(test_case)
        return self._build_verdict(metric)

    def _build_verdict(
        self, metric: ContextualRelevancyMetric
    ) -> ContextualRelevancyVerdict:
        irrelevant: List[str] = []
        # ContextualRelevancyMetric stores verdicts_list: a list of per-context
        # verdict groups (one group per retrieval_context chunk). Each group has
        # a .verdicts list of ContextualRelevancyVerdict, and each of THOSE
        # carries statement / verdict / reason together (unlike faithfulness /
        # answer relevancy, which used parallel lists).
        verdicts_list = getattr(metric, "verdicts_list", None) or []
        for group in verdicts_list:
            for verdict in getattr(group, "verdicts", None) or []:
                if str(getattr(verdict, "verdict", "")).strip().lower() != "yes":
                    statement = getattr(verdict, "statement", "")
                    reason = getattr(verdict, "reason", None)
                    irrelevant.append(
                        f"{statement} ({reason})" if reason else str(statement)
                    )

        score = metric.score if metric.score is not None else 0.0
        return ContextualRelevancyVerdict(
            score=score,
            passed=score >= self._threshold,
            reason=metric.reason or "",
            irrelevant_context=irrelevant,
        )
