# deval

LiteLLM proxy setup notes and evaluation/observability integrations.

- [`mlflow-litellm-integration.md`](./mlflow-litellm-integration.md) — MLflow for
  evals + observability, with a self-hosted open-source judge model behind the
  LiteLLM proxy. Runnable examples in [`examples/`](./examples/).
- [`litellm-deepeval-guardrail-context.md`](./litellm-deepeval-guardrail-context.md) —
  live faithfulness/hallucination guardrail design (DeepEval, `post_call` hook).
