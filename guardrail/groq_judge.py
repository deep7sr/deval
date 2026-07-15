"""DeepEval judge model backed by the LiteLLM SDK.

DeepEval has no native Groq class, so we wrap the exact litellm -> Groq path
that we already proved works end-to-end (including the corporate-SSL fix). A
custom DeepEval model only has to return a string; DeepEval extracts any
structured JSON itself, so this stays thin.

The judge model is deliberately separate from whatever model served the chat
completion, and different from the generation model, to avoid a model grading
its own output.
"""

import litellm
from deepeval.models import DeepEvalBaseLLM

from . import config

# Match the proxy's TLS behaviour for the guardrail's own outbound calls.
litellm.ssl_verify = config.SSL_VERIFY


class GroqJudge(DeepEvalBaseLLM):
    """A DeepEvalBaseLLM that routes generation through litellm to Groq."""

    def __init__(self, model: str = None):
        self.model = model or config.JUDGE_MODEL

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        kwargs = {}
        if schema is not None:
            # Groq supports JSON mode; DeepEval's prompts already mention JSON,
            # which Groq requires when json_object response_format is set.
            kwargs["response_format"] = {"type": "json_object"}
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model
