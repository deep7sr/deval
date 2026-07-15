"""DeepEval judge model backed by the LiteLLM SDK.

The judge is addressed entirely through the LiteLLM SDK, so it is
provider-agnostic: any model string LiteLLM understands works, selected via
GUARDRAIL_JUDGE_MODEL. Switching from Groq to a self-hosted open-source model
(Ollama / vLLM / TGI / any OpenAI-compatible server) is a config change - set
the model string, and (for a self-hosted endpoint) GUARDRAIL_JUDGE_API_BASE.

Examples for GUARDRAIL_JUDGE_MODEL:
  groq/llama-3.3-70b-versatile        (hosted Groq; key via GROQ_API_KEY)
  ollama_chat/gemma3:4b               (+ GUARDRAIL_JUDGE_API_BASE=http://ollama:11434)
  openai/gemma-3-4b-it                (vLLM/TGI OpenAI-compatible; + api_base + api_key)
  hosted_vllm/google/gemma-3-4b-it    (+ api_base)

A custom DeepEval model only has to return a string; DeepEval extracts any
structured JSON itself, so this stays thin. The judge is deliberately separate
from the model that served the chat completion, to avoid self-grading.
"""

import litellm
from deepeval.models import DeepEvalBaseLLM

from . import config

# Match the proxy's TLS behaviour for the guardrail's own outbound calls.
litellm.ssl_verify = config.SSL_VERIFY


class LiteLLMJudge(DeepEvalBaseLLM):
    """A DeepEvalBaseLLM that routes generation through the LiteLLM SDK to any
    configured judge model (hosted or self-hosted)."""

    def __init__(self, model: str = None):
        self.model = model or config.JUDGE_MODEL

    def load_model(self):
        return self.model

    def _extra_kwargs(self, schema) -> dict:
        kwargs = {}
        # Structured-output mode. Some small/local models don't support the
        # json_object response_format; disable via GUARDRAIL_JUDGE_JSON_MODE=false.
        if schema is not None and config.JUDGE_JSON_MODE:
            kwargs["response_format"] = {"type": "json_object"}
        # Self-hosted / OpenAI-compatible endpoint (Ollama, vLLM, TGI, ...).
        if config.JUDGE_API_BASE:
            kwargs["api_base"] = config.JUDGE_API_BASE
        if config.JUDGE_API_KEY:
            kwargs["api_key"] = config.JUDGE_API_KEY
        return kwargs

    def generate(self, prompt: str, schema=None) -> str:
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self._extra_kwargs(schema),
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        resp = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self._extra_kwargs(schema),
        )
        return resp.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model


# Backwards-compatible alias (the class was previously named GroqJudge).
GroqJudge = LiteLLMJudge
