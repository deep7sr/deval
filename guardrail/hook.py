"""LiteLLM post-call guardrail hook (the thin orchestrator).

Wires the three decoupled pieces together on LiteLLM's
``async_post_call_success_hook``:

    parse request  ->  non-compliant?  ->  allow, skip (rule 7)
                       compliant       ->  score grounding
                                           passed  -> allow
                                           failed  -> block / remediate

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
from .grounding import GroundingEvaluator
from .parser import parse_messages
from .remediation import in_remediation, remediate

# Own logger with its own stdout handler so guardrail decisions always show up
# in `docker logs`, regardless of how litellm/uvicorn configure the root logger.
verbose_logger = logging.getLogger("guardrail.hallucination")
if not verbose_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s GUARDRAIL %(levelname)s %(message)s")
    )
    verbose_logger.addHandler(_handler)
    verbose_logger.setLevel(config.LOG_LEVEL.upper())
    verbose_logger.propagate = False

# Tag attached to retry regenerations for log/spend attribution. NOT used as a
# skip signal: request metadata is client-supplied, so trusting it would let a
# caller bypass the guardrail by forging the flag in the request body. The
# actual recursion guard is remediation.in_remediation() (a ContextVar).
_RETRY_FLAG = "guardrail_retry"

_ALLOWED_MODES = ("block", "remediate")


class HallucinationGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        # Strip our own non-CustomGuardrail params before calling super so
        # litellm doesn't choke on unknown kwargs from the config block.
        self.optional_params = kwargs
        super().__init__(**kwargs)
        self.evaluator = GroundingEvaluator()
        self.mode = config.MODE
        if self.mode not in _ALLOWED_MODES:
            verbose_logger.warning(
                "unknown GUARDRAIL_MODE %r; falling back to 'remediate' "
                "(allowed: %s)", self.mode, ", ".join(_ALLOWED_MODES),
            )
            self.mode = "remediate"

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            return await self._run(data, response)
        except Exception as exc:  # fail-open
            verbose_logger.exception(
                "hallucination guardrail error; passing response through: %s", exc
            )
            return response

    # ------------------------------------------------------------------
    async def _run(self, data, response):
        if not isinstance(response, litellm.ModelResponse):
            return response

        # Defence in depth: never act on our own retry regenerations. Checked
        # via a process-local ContextVar - request metadata is deliberately NOT
        # trusted here, because a client could forge a metadata flag in the
        # request body to bypass the guardrail.
        if in_remediation():
            return response

        parse = parse_messages(data.get("messages"))
        if not parse.compliant:
            verbose_logger.info(
                "guardrail skip: reason=%s (guardrail did not run)", parse.skip_reason
            )
            return response

        actual_output = self._get_output(response)
        if not actual_output:
            verbose_logger.info("guardrail skip: no text output in response")
            return response

        # A previously-run guardrail already replaced this response with its
        # fallback; scoring a refusal message is meaningless and would cause
        # pointless remediation / double replacement.
        if config.is_guardrail_fallback(actual_output):
            verbose_logger.info(
                "guardrail skip: response is another guardrail's fallback"
            )
            return response

        verdict = await self.evaluator.a_evaluate(
            parse.input, actual_output, parse.retrieval_context
        )
        verbose_logger.info(
            "guardrail verdict: score=%.3f passed=%s marker_idx=%s",
            verdict.score,
            verdict.passed,
            parse.marker_message_index,
        )

        if verdict.passed:
            return response

        if self.mode == "block":
            verbose_logger.warning(
                "guardrail BLOCK: replacing ungrounded response (score=%.3f)",
                verdict.score,
            )
            self._set_output(response, config.FALLBACK_MESSAGE)
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

        async def evaluate(question, output, retrieval_context):
            return await self.evaluator.a_evaluate(question, output, retrieval_context)

        return await remediate(
            question=parse.input,
            retrieval_context=parse.retrieval_context,
            original_output=actual_output,
            initial_verdict=initial_verdict,
            evaluate=evaluate,
            regenerate=regenerate,
            max_retries=config.MAX_RETRIES,
            time_budget=config.RETRY_TIME_BUDGET_SECONDS,
            fallback_message=config.FALLBACK_MESSAGE,
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
