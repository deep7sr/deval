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
# from whatever model actually served the chat completion.
JUDGE_MODEL = _get_str("GUARDRAIL_JUDGE_MODEL", "groq/llama-3.3-70b-versatile")

# --- Logging -------------------------------------------------------------
# Level for the guardrail's own logger. INFO surfaces every parse/skip/verdict/
# remediation decision in the container logs so behaviour is observable.
LOG_LEVEL = _get_str("GUARDRAIL_LOG_LEVEL", "INFO")

# --- Operating mode ------------------------------------------------------
# monitor   : score + log only, never alter the response (safe rollout mode).
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
