"""LiteLLM post-call guardrail hook for Turn Faithfulness (thin orchestrator).

The conversational counterpart of ``hook.py``. Wires the decoupled pieces
together on LiteLLM's ``async_post_call_success_hook``:

    parse request  ->  non-compliant?  ->  allow, skip
                       compliant       ->  score turn faithfulness over the
                                           WHOLE conversation (history turns +
                                           the new answer as the final
                                           assistant turn)
                                           passed  -> allow
                                           failed  -> block / remediate

Unlike the single-turn faithfulness guardrail (which grades only the final
answer against the evidence block preceding the final question), this one
grades every assistant answer in the conversation against the retrieval
context of its own exchange, using DeepEval's sliding-window
TurnFaithfulnessMetric. Its identity is completely separate (own class, own
guardrail_name, own logger, own config namespace) so a team can enable either
faithfulness guardrail - or both - independently.

Remediation regenerates only the FINAL answer (earlier turns were already
delivered and cannot be un-said); each retry re-scores the full conversation
with the regenerated answer in place.

Regeneration goes through the proxy's Router (so the model alias and its
configured API key resolve correctly) and therefore does NOT re-enter this
hook - no recursion. As defence in depth, retry calls are tagged with metadata
and the hook short-circuits if it ever sees that tag.

The hook is fail-open: any unexpected error logs and returns the original
response rather than breaking the user's request.
"""

import logging
import sys

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail

from . import config
from .parser import parse_conversation
from .remediation import build_correction_prompt, remediate
from .turn_faithfulness import TurnFaithfulnessEvaluator

# Own logger with its own stdout handler so guardrail decisions always show up
# in `docker logs`, regardless of how litellm/uvicorn configure the root logger.
verbose_logger = logging.getLogger("guardrail.turn_faithfulness")
if not verbose_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s GUARDRAIL %(levelname)s %(message)s")
    )
    verbose_logger.addHandler(_handler)
    verbose_logger.setLevel(config.LOG_LEVEL.upper())
    verbose_logger.propagate = False

# Distinct from the other hooks' flags so the guardrails never mistake each
# other's retry regenerations for their own.
_RETRY_FLAG = "guardrail_turn_faithfulness_retry"


class TurnFaithfulnessGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        # Strip our own non-CustomGuardrail params before calling super so
        # litellm doesn't choke on unknown kwargs from the config block.
        self.optional_params = kwargs
        super().__init__(**kwargs)
        self.evaluator = TurnFaithfulnessEvaluator()
        self.mode = config.TURN_FAITHFULNESS_MODE

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            return await self._run(data, response)
        except Exception as exc:  # fail-open
            verbose_logger.exception(
                "turn-faithfulness guardrail error; passing response through: %s", exc
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

        parse = parse_conversation(data.get("messages"))
        if not parse.compliant:
            verbose_logger.info(
                "guardrail skip: reason=%s (guardrail did not run)", parse.skip_reason
            )
            return response

        actual_output = self._get_output(response)
        if not actual_output:
            verbose_logger.info("guardrail skip: no text output in response")
            return response

        verdict = await self.evaluator.a_evaluate(
            self._full_turns(parse, actual_output)
        )
        verbose_logger.info(
            "guardrail verdict: score=%.3f passed=%s windows=%d unfaithful=%d",
            verdict.score,
            verdict.passed,
            verdict.windows_evaluated,
            len(verdict.unfaithful_claims),
        )

        if verdict.passed:
            return response

        if self.mode == "block":
            verbose_logger.warning(
                "guardrail BLOCK: replacing unfaithful response (score=%.3f)",
                verdict.score,
            )
            self._set_output(response, config.TURN_FAITHFULNESS_FALLBACK_MESSAGE)
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
    @staticmethod
    def _full_turns(parse, output):
        """History turns + the model's answer as the final assistant turn,
        carrying the evidence retrieved for the final question."""
        return list(parse.turns) + [
            {
                "role": "assistant",
                "content": output,
                "retrieval_context": parse.pending_context,
            }
        ]

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

        # Only the regenerated FINAL answer changes between retries; the
        # question/context args of the shared loop are accepted and ignored so
        # the remediate() signature stays uniform.
        async def evaluate(_question, output, _retrieval_context):
            return await self.evaluator.a_evaluate(self._full_turns(parse, output))

        return await remediate(
            question=parse.input,
            retrieval_context=parse.pending_context or [],
            original_output=actual_output,
            initial_verdict=initial_verdict,
            evaluate=evaluate,
            regenerate=regenerate,
            max_retries=config.MAX_RETRIES,
            time_budget=config.RETRY_TIME_BUDGET_SECONDS,
            fallback_message=config.TURN_FAITHFULNESS_FALLBACK_MESSAGE,
            build_prompt=lambda v: build_correction_prompt(v.unfaithful_claims),
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
