# Why context/query segregation requires an explicit signal — sources

**Claim being defended:** You cannot *reliably and deterministically* separate
retrieved context from the user query in a raw chat request **without an explicit
signal from the sender** (either out-of-band request `metadata`, or an in-band
marker convention). The only signal-free alternative is an LLM that *infers* the
split — which is probabilistic, not deterministic, and unfit for a hard guardrail gate.

---

## 1. The chat request schema has no field for retrieval context

Retrieved evidence is placed free-form inside a message's `content`, structurally
indistinguishable from instructions or the user's own words. There is no field to
parse.

- **OpenAI OpenAPI spec (`ChatCompletionRequestMessage`)** — allowed `role` values are
  only `system, developer, user, assistant, tool`. There is **no** property named
  `context`, `retrieval_context`, or `documents` on a message.
  https://github.com/openai/openai-openapi/blob/master/openapi.yaml
- API reference (human-readable): https://platform.openai.com/docs/api-reference/chat/create

## 2. DeepEval requires context as an explicit, caller-supplied field

DeepEval never parses a raw `messages` array. Every metric reads only the named
fields you construct on an `LLMTestCase`.

- **`LLMTestCase` source** — `input`, `actual_output`, `context`, `retrieval_context`
  are explicit constructor parameters:
  https://github.com/confident-ai/deepeval/blob/main/deepeval/test_case/llm_test_case.py
- **`FaithfulnessMetric` requires `retrieval_context`** (`_required_params`):
  ```python
  _required_params = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.RETRIEVAL_CONTEXT,
  ]
  ```
  https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/faithfulness/faithfulness.py
- Docs: https://deepeval.com/docs/metrics-faithfulness

## 3. The explicit signal can travel via LiteLLM request metadata

A client can attach custom `metadata` to a completion request; it reaches the
guardrail hook as `data["metadata"]`.

- https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail
- https://docs.litellm.ai/docs/proxy/call_hooks

---

## Chain of logic

1. Chat schema has **no context field** → context is indistinguishable inside `content`.
2. DeepEval's `FaithfulnessMetric` **requires `retrieval_context` explicitly** → it
   cannot derive it from raw messages.
3. Therefore the sender must signal the boundary — via **metadata** (out-of-band) or a
   **marker** (in-band). Signal-free = LLM guessing = probabilistic, not deterministic.

**Precise wording:** the defensible claim is *"an explicit signal is required."*
Metadata is one valid channel; an in-band marker is another. Both are explicit;
neither is "no explicit mention."
