# MLflow + LiteLLM Integration — Evals & Observability

Setup guide for adding **MLflow** (open-source MLOps/GenAI platform) to an existing
**LiteLLM Enterprise proxy** for two purposes:

1. **Observability (tracing)** — capture every request that flows through the proxy.
2. **Evaluation (LLM-as-a-judge)** — score app quality using a **self-hosted,
   open-source judge model** so no data leaves the network and there is no
   frontier-model cost.

This document is derived from the official LiteLLM MLflow docs
(<https://docs.litellm.ai/docs/observability/mlflow>), the LiteLLM eval-suite
tutorial (<https://docs.litellm.ai/docs/tutorials/eval_suites>), and the MLflow
GenAI docs (tracing: <https://mlflow.org/docs/latest/genai/tracing/integrations/listing/litellm-proxy/>,
eval/judges: <https://mlflow.org/docs/latest/genai/eval-monitor/>).

---

## 0. The core idea for our setup

Everything already goes through **one** LiteLLM proxy. We keep it that way and
register the **judge model as just another model in the same proxy**:

```
                         ┌───────────────────────────────────────┐
   app teams  ─────────► │            LiteLLM Proxy              │ ──► Claude / GPT / … (frontier, app traffic)
   (OpenAI SDK)          │        (Enterprise, OpenAI-compat)    │ ──► self-hosted OSS judge (vLLM/Ollama/TGI)
                         └───────────────┬───────────────────────┘
                                         │ success_callback: ["mlflow"]   (tracing)
                                         ▼
                              ┌────────────────────┐
                              │  MLflow Tracking    │  ◄── offline eval jobs
                              │  server + UI        │      (mlflow.genai.evaluate)
                              │  (self-hosted)      │      judge calls go back
                              └────────────────────┘      through the SAME proxy
```

Why this shape:

- **Data stays internal.** The judge model is a local OSS model served by vLLM /
  Ollama / TGI and exposed *through the proxy*. MLflow's eval calls hit the proxy
  (OpenAI-compatible), which routes to the local judge. Nothing goes to a frontier
  vendor.
- **No new cost.** Judge inference runs on our own hardware.
- **One control plane.** Virtual keys, rate limits, spend tracking, and logging
  already exist on the Enterprise proxy — the judge inherits all of it.
- **Observability and eval connect.** Traces captured from production can be pulled
  straight into an offline eval run (`mlflow.search_traces(...)`), so we evaluate
  *real* traffic, not just synthetic prompts.

---

## Part A — Observability (tracing) on the proxy

### A.1 Install MLflow in the proxy image

MLflow ships as a LiteLLM logging callback. It must be installed **inside the
LiteLLM proxy container** (not just on a laptop).

```dockerfile
# Dockerfile (extends the official/enterprise LiteLLM image)
FROM ghcr.io/berriai/litellm:main-stable
RUN pip install --no-cache-dir "mlflow>=3.1.4"
```

> Per the LiteLLM docs, the proxy needs `mlflow>=3.1.4`. (The SDK path uses the
> `litellm[mlflow]` extra, but on the proxy you install `mlflow` directly.)

### A.2 Enable the callback in `config.yaml`

```yaml
litellm_settings:
  success_callback: ["mlflow"]
  failure_callback: ["mlflow"]   # capture failed calls too
```

This is **server-side tracing**: every request through the proxy — regardless of
which team/client made it — is logged to one MLflow experiment.

### A.3 Point the proxy at our self-hosted MLflow

We are **not** using Databricks, so ignore the `DATABRICKS_*` / `MLFLOW_REGISTRY_URI`
variables from the LiteLLM doc. For a self-hosted MLflow tracking server, set:

```shell
MLFLOW_TRACKING_URI=http://mlflow:5000       # our tracking server
MLFLOW_EXPERIMENT_NAME=litellm-proxy-prod     # groups all proxy traces
# (MLFLOW_EXPERIMENT_ID may be used instead of the name if you prefer)
```

These must be present **inside the container** (via `environment:` / an env file),
not just on the host shell.

### A.4 Run a self-hosted MLflow tracking server

MLflow is free and needs no signup/API key. Minimal production-ish server:

```shell
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow \
  --artifacts-destination /mlflow/artifacts
```

Open the UI at `http://<host>:5000` → **Traces** tab to see logged calls. See
`examples/docker-compose.yml` in this repo for a full LiteLLM + MLflow + Postgres
stack.

### A.5 Tag requests for filterable traces

Teams can attach tags per request; they show up on the trace for search/filter:

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "..."}],
    extra_body={
        "litellm_metadata": {
            "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"]
        }
    },
)
```

### A.6 (Optional) Fan out to an OpenTelemetry collector

MLflow traces are OTel-compatible. To also ship them to Jaeger/Datadog/etc.:

```shell
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://otel-collector:4317/v1/traces
OTEL_SERVICE_NAME=litellm-proxy
```

---

## Part B — Evaluation with an open-source judge

MLflow 3.x's evaluation API is `mlflow.genai.evaluate(...)`, driven by **scorers**.
Two kinds:

- **Built-in judges** (`mlflow.genai.scorers`): `Correctness`, `Guidelines`,
  `RelevanceToQuery`, `Safety`, `RetrievalGroundedness`, etc.
- **Custom judges** (`mlflow.genai.judges.make_judge`): natural-language grading
  criteria you define yourself.

Every LLM judge takes a `model` parameter. **That's where we plug in our
self-hosted model through the proxy.**

### B.1 Register the open-source judge model in the LiteLLM proxy

Serve an OSS model locally (examples: `Qwen2.5-32B-Instruct`, `Llama-3.3-70B`,
`Mistral-Small`) with vLLM/Ollama/TGI, then add it to the proxy as an
OpenAI-compatible model:

```yaml
# config.yaml  (add to model_list)
model_list:
  - model_name: judge-qwen           # the name MLflow will call
    litellm_params:
      model: openai/Qwen2.5-32B-Instruct
      api_base: http://vllm:8000/v1  # the local OSS server
      api_key: "none"                # or a real key if the server needs one
```

Now `judge-qwen` is callable through the proxy just like any frontier model.

### B.2 Point MLflow's judge at the proxy

MLflow judge model URIs use `provider:/model`. Because the proxy is
OpenAI-compatible, use the `openai:/` provider and redirect its base URL to the
proxy via environment variables:

```shell
export OPENAI_API_BASE=http://litellm:4000/v1   # our proxy, NOT api.openai.com
export OPENAI_API_KEY=sk-litellm-virtual-key     # a LiteLLM virtual key
```

```python
from mlflow.genai.judges import make_judge

# model string = openai:/<model_name as registered in the proxy>
faithfulness = make_judge(
    name="faithfulness",
    instructions=(
        "You are grading whether the response is fully supported by the "
        "retrieved evidence.\n\n"
        "Evidence:\n{{ inputs }}\n\nResponse:\n{{ outputs }}\n\n"
        "Return 'pass' only if every claim in the response is supported by the "
        "evidence; otherwise 'fail'. Explain briefly."
    ),
    model="openai:/judge-qwen",
)
```

Judge inference now runs on our own hardware, routed and metered by the same proxy.

> Notes on alternatives:
> - MLflow also supports a `litellm` provider (`pip install litellm`) and custom
>   `proxy_url` / headers on judges. The `openai:/` + `OPENAI_API_BASE` route above
>   is the simplest and needs nothing beyond the proxy we already run.
> - MLflow's AI Gateway (`gateway:/<endpoint>`) is another option, but we already
>   have LiteLLM as our gateway, so we don't add a second one.

### B.3 Run an evaluation

```python
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("litellm-eval")

# 1) A dataset: each row has inputs (+ optional expectations)
data = [
    {
        "inputs": {"question": "What is our refund window?"},
        "expectations": {"expected_response": "30 days"},
    },
    # ...
]

# 2) The app under test — call it through the proxy
from openai import OpenAI
client = OpenAI(base_url="http://litellm:4000/v1", api_key="sk-litellm-virtual-key")

def predict_fn(question: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o",  # whatever model the app actually uses
        messages=[{"role": "user", "content": question}],
    )
    return r.choices[0].message.content

# 3) Scorers — all judged by our OSS model via the proxy
results = mlflow.genai.evaluate(
    data=data,
    predict_fn=predict_fn,
    scorers=[
        Correctness(model="openai:/judge-qwen"),
        RelevanceToQuery(model="openai:/judge-qwen"),
        Guidelines(
            name="tone",
            guidelines="The answer must be professional and in English.",
            model="openai:/judge-qwen",
        ),
        faithfulness,  # the custom judge from B.2
    ],
)
print(results.metrics)
```

Results (per-row scores + aggregate metrics) land in the MLflow UI under the
experiment, with the judge's rationale attached to each row.

### B.4 Evaluate real production traffic (recommended)

Because Part A logs every proxy call as a trace, we can evaluate **actual logged
traffic** instead of synthetic prompts:

```python
traces = mlflow.search_traces(
    experiment_names=["litellm-proxy-prod"],
    filter_string="tags.taskName = 'run_page_classification'",
    max_results=200,
)
mlflow.genai.evaluate(
    data=traces,                 # traces carry inputs + outputs already
    scorers=[Safety(model="openai:/judge-qwen"),
             RelevanceToQuery(model="openai:/judge-qwen")],
)
```

This closes the loop: **observe in prod → sample traces → score offline with the
OSS judge → track quality over time in MLflow.**

---

## How this relates to the existing DeepEval guardrail work

The prior design (`litellm-deepeval-guardrail-context.md`) is a **live, blocking**
faithfulness guardrail (`post_call` hook, single-turn `FaithfulnessMetric`). This
MLflow work is complementary and **offline/observability-focused**:

| | DeepEval guardrail | MLflow eval + tracing |
|---|---|---|
| When | Inline, per request (blocks) | Offline / retrospective (scores) |
| Purpose | Stop hallucinations reaching users | Track quality, compare models/prompts, monitor drift |
| Judge | DeepEval `FaithfulnessMetric` | MLflow judges (OSS model via proxy) |
| Data | Live request/response | Logged traces or eval datasets |

They can share the **same self-hosted judge model** and the **same context/marker
extraction logic**. MLflow can also wrap DeepEval/RAGAS as scorers, so the
`FaithfulnessMetric` can be reused as an offline scorer for validating the guardrail
against a labeled set — exactly the "offline/retrospective quality metric" that was
flagged as future work in that doc.

---

## Decisions locked in

1. **Self-hosted MLflow tracking server** (Postgres backend) — not Databricks. No
   `DATABRICKS_*` vars.
2. **Server-side tracing** via `success_callback`/`failure_callback: ["mlflow"]` so
   all teams are covered with zero client changes.
3. **Judge = OSS model behind the same LiteLLM proxy**, reached from MLflow via
   `openai:/<name>` + `OPENAI_API_BASE` → proxy. Keeps data internal, zero frontier
   cost, centralized keys.
4. **`mlflow>=3.1.4`** baked into the proxy image.

## Open questions / next steps

- Which OSS model for the judge, and how served (vLLM vs Ollama vs TGI)? Judge
  quality matters — a too-small model gives noisy verdicts. Validate against a
  small labeled set before trusting scores.
- Sizing: judge inference competes with app traffic for GPU. Consider a dedicated
  serving instance (still registered in the proxy).
- Where the MLflow server + Postgres run, backups, and who has UI access.
- Sampling policy for offline eval over prod traces (100% is expensive; sample by
  tag/route).
- Retention for traces/artifacts in the MLflow backing store.

See `examples/` for a runnable `docker-compose.yml`, `litellm_config.yaml`, and an
eval script.
