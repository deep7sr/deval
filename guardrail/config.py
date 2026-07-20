"""Central configuration for the LiteLLM + DeepEval hallucination guardrail.

Everything tunable lives here as a named constant so nothing is buried as an
inline literal in the parser, the judge wrapper, or the hook. Values can be
overridden via environment variables so the same code runs unchanged in the
local Docker test setup and in the infra team's production proxy.
"""

import os


def _get_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# --- The contract marker -------------------------------------------------
# Evidence lives in an assistant-role message whose content STARTS WITH this
# exact string (exact spelling and spacing - note "Retrieved", not the
# typo'd "Retreived" from early reference material). This is matched with a
# plain, case-sensitive startswith() - no normalisation, no trimming.
EVIDENCE_MARKER = _get_str("GUARDRAIL_EVIDENCE_MARKER", "--- Retrieved Evidence ---")

# --- Networking ----------------------------------------------------------
# Whether the guardrail's own litellm SDK calls (judge + retries) verify TLS.
# Defaults to False to match the corporate proxy's ssl_verify: false; infra
# can flip this to True once a proper CA bundle is in place.
SSL_VERIFY = _get_bool("GUARDRAIL_SSL_VERIFY", False)

# --- Faithfulness scoring ------------------------------------------------
# A response passes when its faithfulness score is >= this threshold.
FAITHFULNESS_THRESHOLD = _get_float("GUARDRAIL_FAITHFULNESS_THRESHOLD", 0.7)

# The DeepEval judge model, addressed through the LiteLLM SDK. Kept separate
# from whatever model actually served the chat completion. Any model string
# LiteLLM understands works (hosted Groq/OpenAI/etc., or a self-hosted
# open-source model via Ollama / vLLM / TGI).
JUDGE_MODEL = _get_str("GUARDRAIL_JUDGE_MODEL", "groq/llama-3.3-70b-versatile")

# For a self-hosted / OpenAI-compatible judge endpoint (Ollama, vLLM, TGI, ...).
# Leave unset for hosted providers that read their key from the environment
# (e.g. Groq via GROQ_API_KEY). Example: http://ollama:11434
JUDGE_API_BASE = os.environ.get("GUARDRAIL_JUDGE_API_BASE")
JUDGE_API_KEY = os.environ.get("GUARDRAIL_JUDGE_API_KEY")

# Whether to request JSON-mode structured output from the judge. Strong models
# support it; some small/local models don't - set false if the judge errors on
# response_format (DeepEval still parses JSON out of a plain-text reply).
JUDGE_JSON_MODE = _get_bool("GUARDRAIL_JUDGE_JSON_MODE", True)

# --- Answer Relevancy scoring --------------------------------------------
# Own namespace so the answer-relevancy guardrail is configured, enabled, and
# tuned completely independently of the faithfulness one (no mixups). The
# generic settings below (judge model, SSL, retries, retry temperature) are
# deliberately SHARED between guardrails; only the metric-specific knobs are
# namespaced.
#
# A response passes when its answer-relevancy score is >= this threshold.
ANSWER_RELEVANCY_THRESHOLD = _get_float("GUARDRAIL_ANSWER_RELEVANCY_THRESHOLD", 0.7)

# block     : replace an off-topic response with the fallback message, no retry.
# remediate : run the self-correction retry loop, then fall back if still bad.
ANSWER_RELEVANCY_MODE = _get_str("GUARDRAIL_ANSWER_RELEVANCY_MODE", "remediate")

# Message returned to the user when all retries still fail the relevancy check.
ANSWER_RELEVANCY_FALLBACK_MESSAGE = _get_str(
    "GUARDRAIL_ANSWER_RELEVANCY_FALLBACK_MESSAGE",
    "I wasn't able to give you a focused answer to that question. Could you "
    "rephrase it or add a little more detail so I can respond directly?",
)

# --- Contextual Relevancy scoring ----------------------------------------
# Own namespace, independent of the other guardrails. Contextual Relevancy
# grades the RETRIEVER (is the retrieved context relevant to the question?),
# not the LLM's answer - so re-prompting the model can't fix a bad score. There
# is deliberately NO remediate mode here.
#
# A response passes when the contextual-relevancy score is >= this threshold.
CONTEXTUAL_RELEVANCY_THRESHOLD = _get_float(
    "GUARDRAIL_CONTEXTUAL_RELEVANCY_THRESHOLD", 0.7
)

# block   : replace the answer with the fallback message when the retrieved
#           context is judged irrelevant to the question (default).
# observe : run the metric and log the verdict, but never alter the response
#           (retrieval-quality signal only, for dashboards / shadow mode).
CONTEXTUAL_RELEVANCY_MODE = _get_str("GUARDRAIL_CONTEXTUAL_RELEVANCY_MODE", "block")

# Message returned to the user when the retrieved context is irrelevant and the
# guardrail is in block mode.
CONTEXTUAL_RELEVANCY_FALLBACK_MESSAGE = _get_str(
    "GUARDRAIL_CONTEXTUAL_RELEVANCY_FALLBACK_MESSAGE",
    "I couldn't find information relevant to your question in the material "
    "available to me, so I'd rather not answer than risk giving you something "
    "based on unrelated context.",
)

# --- Turn Faithfulness scoring ---------------------------------------------
# Own namespace, independent of the other guardrails. Turn Faithfulness is the
# CONVERSATIONAL counterpart of the single-turn faithfulness guardrail: it
# scores the assistant's claims across the WHOLE conversation against the
# retrieval context attached to each exchange (DeepEval TurnFaithfulnessMetric,
# sliding-window evaluation), instead of grading only the final answer.
#
# A response passes when the conversation-level score is >= this threshold.
TURN_FAITHFULNESS_THRESHOLD = _get_float("GUARDRAIL_TURN_FAITHFULNESS_THRESHOLD", 0.7)

# block     : replace an unfaithful response with the fallback message, no retry.
# remediate : run the self-correction retry loop, then fall back if still bad.
# NOTE: remediation can only regenerate the FINAL answer; unfaithful claims in
# already-delivered earlier turns cannot be fixed retroactively, so a long
# unfaithful history may keep the conversation score below threshold even after
# a clean regeneration.
TURN_FAITHFULNESS_MODE = _get_str("GUARDRAIL_TURN_FAITHFULNESS_MODE", "remediate")

# Sliding-window size (in unit interactions) used by TurnFaithfulnessMetric.
# Each unit interaction in the conversation produces one evaluation window of
# up to this many interactions, so longer conversations mean more judge calls.
TURN_FAITHFULNESS_WINDOW_SIZE = _get_int("GUARDRAIL_TURN_FAITHFULNESS_WINDOW_SIZE", 10)

# Whether an "idk" verdict (the judge can't confirm the claim is true OR
# false against the truths) counts AGAINST faithfulness. Default False matches
# DeepEval's own default: idk is treated as faithful (only "no" is penalized).
# Flip to True if the judge is being lenient on claims that should have been a
# clear contradiction but got marked ambiguous instead.
TURN_FAITHFULNESS_PENALIZE_AMBIGUOUS_CLAIMS = _get_bool(
    "GUARDRAIL_TURN_FAITHFULNESS_PENALIZE_AMBIGUOUS_CLAIMS", False
)

# Message returned to the user when all retries still fail the check.
TURN_FAITHFULNESS_FALLBACK_MESSAGE = _get_str(
    "GUARDRAIL_TURN_FAITHFULNESS_FALLBACK_MESSAGE",
    "I couldn't produce an answer that stays consistent with the information "
    "retrieved during this conversation, so I'd rather not answer than risk "
    "giving you something inaccurate.",
)

# --- Logging -------------------------------------------------------------
# Level for the guardrail's own logger. INFO surfaces every parse/skip/verdict/
# remediation decision in the container logs so behaviour is observable.
LOG_LEVEL = _get_str("GUARDRAIL_LOG_LEVEL", "INFO")

# --- Operating mode ------------------------------------------------------
# block     : replace an ungrounded response with FALLBACK_MESSAGE, no retry.
# remediate : run the self-correction retry loop, then fall back if still bad.
MODE = _get_str("GUARDRAIL_MODE", "remediate")

# --- Remediation (self-correction retry loop) ----------------------------
# When a response is judged ungrounded, re-prompt the SAME model that produced
# it with the specific unsupported claims, up to this many times, before
# falling back to a safe message.
MAX_RETRIES = _get_int("GUARDRAIL_MAX_RETRIES", 3)

# Total wall-clock budget (seconds) for the whole remediation loop, so a slow
# case can't hang the user's request indefinitely.
RETRY_TIME_BUDGET_SECONDS = _get_float("GUARDRAIL_RETRY_TIME_BUDGET_SECONDS", 30.0)

# Temperature for corrective retry regenerations. A small positive value nudges
# the model off an identical repeat of the same ungrounded answer.
RETRY_TEMPERATURE = _get_float("GUARDRAIL_RETRY_TEMPERATURE", 0.3)

# Message returned to the user when all retries still fail the grounding check.
FALLBACK_MESSAGE = _get_str(
    "GUARDRAIL_FALLBACK_MESSAGE",
    "I couldn't produce an answer grounded in the available information, "
    "so I'd rather not answer than risk giving you something inaccurate.",
)
