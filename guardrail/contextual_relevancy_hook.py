"""LiteLLM post-call guardrail hook for Contextual Relevancy (thin orchestrator).

The third guardrail, alongside the faithfulness and answer-relevancy hooks. It
runs on LiteLLM's ``async_post_call_success_hook``:

    parse request  ->  no evidence marker?   ->  allow, skip
                       has input + context   ->  score contextual relevancy
                                                 passed  -> allow
                                                 failed  -> block (or observe)

Contextual Relevancy grades the RETRIEVER (is the retrieved context relevant to
the question?), not the LLM's answer. Because retrieval happened upstream in the
caller's RAG pipeline, a bad score cannot be fixed by re-prompting the model -
so there is deliberately NO remediation loop here. The two modes are:

  block   : replace the answer with a fallback message (default).
  observe : log the verdict only; never alter the response.

Like faithfulness, it needs ``retrieval_context`` and therefore REQUIRES the
evidence marker (reuses parse_messages), silently skipping requests without one.

Its identity is completely separate from the other guardrails (own class, own
guardrail_name, own logger, own config namespace) so a team can enable any
combination of the three independently.

The hook is fail-open: any unexpected error logs and returns the original
response rather than breaking the user's request.
"""

import logging
import sys

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail

from . import config
from .contextual_relevancy import ContextualRelevancyEvaluator
from .parser import parse_messages

# Own logger with its own stdout handler so guardrail decisions always show up
# in `docker logs`, regardless of how litellm/uvicorn configure the root logger.
verbose_logger = logging.getLogger("guardrail.contextual_relevancy")
if not verbose_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s GUARDRAIL %(levelname)s %(message)s")
    )
    verbose_logger.addHandler(_handler)
    verbose_logger.setLevel(config.LOG_LEVEL.upper())
    verbose_logger.propagate = False


_ALLOWED_MODES = ("block", "observe")


class ContextualRelevancyGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        # Strip our own non-CustomGuardrail params before calling super so
        # litellm doesn't choke on unknown kwargs from the config block.
        self.optional_params = kwargs
        super().__init__(**kwargs)
        self.evaluator = ContextualRelevancyEvaluator()
        self.mode = config.CONTEXTUAL_RELEVANCY_MODE
        if self.mode not in _ALLOWED_MODES:
            verbose_logger.warning(
                "unknown GUARDRAIL_CONTEXTUAL_RELEVANCY_MODE %r; falling back "
                "to 'block' (allowed: %s)", self.mode, ", ".join(_ALLOWED_MODES),
            )
            self.mode = "block"

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            return await self._run(data, response)
        except Exception as exc:  # fail-open
            verbose_logger.exception(
                "contextual-relevancy guardrail error; passing response through: %s",
                exc,
            )
            return response

    # ------------------------------------------------------------------
    async def _run(self, data, response):
        if not isinstance(response, litellm.ModelResponse):
            return response

        # A previously-run guardrail already replaced this response with its
        # fallback; the delivered content is already a safe refusal, so don't
        # spend judge calls or overwrite it a second time.
        if config.is_guardrail_fallback(self._get_output(response) or ""):
            verbose_logger.info(
                "guardrail skip: response is another guardrail's fallback"
            )
            return response

        # Needs retrieval_context -> requires the evidence marker, exactly like
        # the faithfulness guardrail. No marker -> silent skip.
        parse = parse_messages(data.get("messages"))
        if not parse.compliant:
            verbose_logger.info(
                "guardrail skip: reason=%s (guardrail did not run)", parse.skip_reason
            )
            return response

        verdict = await self.evaluator.a_evaluate(
            parse.input, parse.retrieval_context
        )
        verbose_logger.info(
            "guardrail verdict: score=%.3f passed=%s irrelevant_context=%d marker_idx=%s",
            verdict.score,
            verdict.passed,
            len(verdict.irrelevant_context),
            parse.marker_message_index,
        )

        if verdict.passed:
            return response

        # Failed: the retrieved context is (largely) irrelevant to the question.
        if self.mode == "observe":
            # Retrieval-quality signal only - never alter the response.
            verbose_logger.warning(
                "guardrail OBSERVE: irrelevant retrieval (score=%.3f, %d off-topic "
                "context statements); response passed through unchanged",
                verdict.score,
                len(verdict.irrelevant_context),
            )
            return response

        # block (default): re-prompting can't fix retrieval, so replace the
        # answer with a safe fallback rather than answer on off-topic context.
        verbose_logger.warning(
            "guardrail BLOCK: replacing answer built on irrelevant context "
            "(score=%.3f)",
            verdict.score,
        )
        self._set_output(response, config.CONTEXTUAL_RELEVANCY_FALLBACK_MESSAGE)
        return response

    # ------------------------------------------------------------------
    @staticmethod
    def _get_output(response):
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return None

    @staticmethod
    def _set_output(response, text):
        try:
            message = response.choices[0].message
            message.content = text
            # The answer we're delivering is no longer the model's original
            # generation, so any chain-of-thought it emitted (reasoning /
            # reasoning_content) now describes a discarded answer. Clear it so
            # the response can't ship a reasoning trace that contradicts the
            # content we just wrote.
            for attr in ("reasoning", "reasoning_content"):
                if getattr(message, attr, None) is not None:
                    try:
                        setattr(message, attr, None)
                    except (AttributeError, TypeError):
                        pass
        except (AttributeError, IndexError, TypeError):
            verbose_logger.exception("failed to set guardrail output on response")
