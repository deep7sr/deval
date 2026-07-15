"""LiteLLM post-call guardrail hook for Answer Relevancy (thin orchestrator).

The relevancy counterpart to ``hook.py``. Wires the decoupled pieces together on
LiteLLM's ``async_post_call_success_hook``:

    parse request  ->  no final user message?  ->  allow, skip
                       has question            ->  score answer relevancy
                                                   passed  -> allow
                                                   failed  -> block / remediate

Unlike the faithfulness guardrail, this needs NO evidence marker and NO
retrieval context: it runs on any Q&A request that has a final user message and
a text response, and is scoped per key. Its identity is completely separate from
the faithfulness guardrail (own class, own guardrail_name, own logger, own
config namespace) so a team can enable faithfulness, relevancy, both, or
neither.

Regeneration during remediation goes through the proxy's Router (so the model
alias and its configured API key resolve correctly) and therefore does NOT
re-enter this hook - no recursion. As defence in depth, retry calls are tagged
with metadata and the hook short-circuits if it ever sees that tag.

The hook is fail-open: any unexpected error logs and returns the original
response rather than breaking the user's request.
"""

import logging
import sys

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail

from . import config
from .parser import parse_final_user_message
from .relevancy import RelevancyEvaluator
from .remediation import build_relevancy_correction_prompt, remediate

# Own logger with its own stdout handler so guardrail decisions always show up
# in `docker logs`, regardless of how litellm/uvicorn configure the root logger.
verbose_logger = logging.getLogger("guardrail.answer_relevancy")
if not verbose_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s GUARDRAIL %(levelname)s %(message)s")
    )
    verbose_logger.addHandler(_handler)
    verbose_logger.setLevel(config.LOG_LEVEL.upper())
    verbose_logger.propagate = False

# Distinct from the faithfulness hook's flag so the two guardrails never
# mistake each other's retry regenerations for their own.
_RETRY_FLAG = "guardrail_relevancy_retry"


class AnswerRelevancyGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        # Strip our own non-CustomGuardrail params before calling super so
        # litellm doesn't choke on unknown kwargs from the config block.
        self.optional_params = kwargs
        super().__init__(**kwargs)
        self.evaluator = RelevancyEvaluator()
        self.mode = config.ANSWER_RELEVANCY_MODE

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            return await self._run(data, response)
        except Exception as exc:  # fail-open
            verbose_logger.exception(
                "answer-relevancy guardrail error; passing response through: %s", exc
            )
            return response

    # ------------------------------------------------------------------
    async def _run(self, data, response):
        if not isinstance(response, litellm.ModelResponse):
            return response

        # Defence in depth: never act on our own retry regenerations.
        metadata = data.get("metadata") or {}
        if metadata.get(_RETRY_FLAG):
            return response

        parse = parse_final_user_message(data.get("messages"))
        if not parse.compliant:
            verbose_logger.info(
                "guardrail skip: reason=%s (guardrail did not run)", parse.skip_reason
            )
            return response

        actual_output = self._get_output(response)
        if not actual_output:
            verbose_logger.info("guardrail skip: no text output in response")
            return response

        verdict = await self.evaluator.a_evaluate(parse.input, actual_output)
        verbose_logger.info(
            "guardrail verdict: score=%.3f passed=%s irrelevant=%d",
            verdict.score,
            verdict.passed,
            len(verdict.irrelevant_statements),
        )

        if verdict.passed:
            return response

        if self.mode == "block":
            verbose_logger.warning(
                "guardrail BLOCK: replacing off-topic response (score=%.3f)",
                verdict.score,
            )
            self._set_output(response, config.ANSWER_RELEVANCY_FALLBACK_MESSAGE)
            return response

        # remediate (default)
        result = await self._remediate(data, parse, actual_output, verdict)
        self._set_output(response, result.final_output)
        verbose_logger.info(
            "guardrail REMEDIATE: outcome=%s attempts=%s final_score=%s",
            result.outcome,
            result.attempts,
            getattr(result.final_verdict, "score", None),
        )
        return response

    # ------------------------------------------------------------------
    async def _remediate(self, data, parse, actual_output, initial_verdict):
        model = data.get("model")
        messages = data.get("messages") or []

        async def regenerate(correction_prompt, previous_output):
            retry_messages = list(messages) + [
                {"role": "assistant", "content": previous_output},
                {"role": "user", "content": correction_prompt},
            ]
            resp = await self._acompletion(model, retry_messages)
            return self._get_output(resp) or ""

        # Relevancy needs no retrieval context; the shared loop's third arg is
        # accepted and ignored so the remediate() signature stays uniform.
        async def evaluate(question, output, _retrieval_context):
            return await self.evaluator.a_evaluate(question, output)

        return await remediate(
            question=parse.input,
            retrieval_context=[],
            original_output=actual_output,
            initial_verdict=initial_verdict,
            evaluate=evaluate,
            regenerate=regenerate,
            max_retries=config.MAX_RETRIES,
            time_budget=config.RETRY_TIME_BUDGET_SECONDS,
            fallback_message=config.ANSWER_RELEVANCY_FALLBACK_MESSAGE,
            build_prompt=lambda v: build_relevancy_correction_prompt(
                v.irrelevant_statements
            ),
            on_event=lambda event, **kw: verbose_logger.info(
                "remediation event=%s %s", event, kw
            ),
        )

    async def _acompletion(self, model, messages):
        """Regenerate via the proxy Router (resolves alias + api key, and does
        not re-enter guardrail hooks). Falls back to the litellm SDK when no
        router is present (e.g. standalone tests)."""
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=config.RETRY_TEMPERATURE,
            metadata={_RETRY_FLAG: True},
        )
        router = self._get_router()
        if router is not None:
            return await router.acompletion(**kwargs)
        return await litellm.acompletion(**kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_router():
        try:
            from litellm.proxy.proxy_server import llm_router

            return llm_router
        except Exception:
            return None

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
