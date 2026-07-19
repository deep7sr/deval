# MLflow + LiteLLM Integration — Evals & Observability

Setup guide for adding **MLflow** (open-source MLOps/GenAI platform) to an existing
**LiteLLM Enterprise proxy** for two purposes:

1. **Observability (tracing)** — capture every request that flows through the proxy.
2. **Evaluation (LLM-as-a-judge)** — score app quality using an **open-source judge
   model** so no data leaves the network and there is no frontier-model cost.

### Two phases

- **Now (dev / demo):** run everything against a **local dev LiteLLM proxy**, with
  the judge model served by **Groq** (a free/cheap, OpenAI-compatible provider that
  hosts open-weight models — gpt-oss, Qwen, Llama). This lets us build and demo the
  full flow with just a `GROQ_API_KEY`, no GPU.
- **Later (production):** the infra team keeps the exact same setup and only
  **repoints the `judge` model** in the proxy from Groq to a **self-hosted OSS
  model** (vLLM/Ollama/TGI). Nothing in the MLflow eval code changes. This is the
  central point of the demo — see **Part C**.

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
   (OpenAI SDK)          │        (Enterprise, OpenAI-compat)    │ ──► judge model:
                         └───────────────┬───────────────────────┘        dev  = Groq (open-weight)
                                         │                                 prod = self-hosted OSS
                                         │                                        (vLLM/Ollama/TGI)
                                         │ success_callback: ["mlflow"]   (tracing)
                                         ▼
                              ┌────────────────────┐
                              │  MLflow Tracking    │  ◄── offline eval jobs
                              │  server + UI        │      (mlflow.genai.evaluate)
                              │  (self-hosted)      │      judge calls go back
                              └────────────────────┘      through the SAME proxy
```

Why this shape:

- **Data stays internal (in prod).** The judge is a local OSS model served by vLLM /
  Ollama / TGI and exposed *through the proxy*. MLflow's eval calls hit the proxy
  (OpenAI-compatible), which routes to the local judge. Nothing goes to a frontier
  vendor. *(In dev the judge is Groq — same wiring, swapped later; see Part C.)*
- **No new cost.** In prod, judge inference runs on our own hardware; in dev, Groq's
  free/cheap open-weight tier.
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

### B.1 Register the judge model in the LiteLLM proxy

Register the judge under a **provider-neutral name** (`judge`) so nothing downstream
cares what's behind it. In dev that's a Groq open-weight model; in prod it's a
self-hosted OSS model. Both are OpenAI-compatible:

```yaml
# config.yaml (model_list) — DEV / DEMO
  - model_name: judge                       # the stable name MLflow calls
    litellm_params:
      model: groq/openai/gpt-oss-120b       # open-weight model on Groq
      api_key: os.environ/GROQ_API_KEY

# PRODUCTION — same model_name, only litellm_params change:
#  - model_name: judge
#    litellm_params:
#      model: openai/Qwen2.5-32B-Instruct   # served locally
#      api_base: http://vllm:8000/v1        # self-hosted vLLM/Ollama/TGI
#      api_key: "none"
```

Now `judge` is callable through the proxy just like any other model. See
**Part C** for the full dev→prod story.

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
    model="openai:/judge",
)
```

The judge call now routes through the proxy — to Groq in dev, to our own hardware in
prod — without the eval code ever knowing the difference.

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
        Correctness(model="openai:/judge"),
        RelevanceToQuery(model="openai:/judge"),
        Guidelines(
            name="tone",
            guidelines="The answer must be professional and in English.",
            model="openai:/judge",
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
    scorers=[Safety(model="openai:/judge"),
             RelevanceToQuery(model="openai:/judge")],
)
```

This closes the loop: **observe in prod → sample traces → score offline with the
OSS judge → track quality over time in MLflow.**

---

## Part C — Dev/demo with Groq, then swap to production OSS

The entire point of using Groq now is that **the migration to production is a
one-line change in one file**. Everything else — the MLflow server, the tracing
callbacks, the eval script, the scorers, the `openai:/judge` URI — stays identical.

### C.1 Why Groq for dev

- **OpenAI-compatible** — LiteLLM talks to it exactly like it will talk to a local
  vLLM/TGI server, so the integration we build and demo is the real one.
- **Open-weight models** — Groq serves `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
  `qwen/qwen3-32b`, Llama, etc. These are the *same class* of model we'll self-host,
  so judge behavior in the demo is representative.
- **No GPU, one key** — the whole demo runs on a single `GROQ_API_KEY`.
- **Free/cheap tier** — fine for dev volumes.

> ⚠️ Groq rotates model IDs. `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`
> were **deprecated in June 2026**; use `openai/gpt-oss-120b` / `qwen/qwen3-32b`
> instead, and check <https://console.groq.com/docs/models> for the current list.

### C.2 The only thing that changes: the `judge` entry

```yaml
# DEV (what we demo)                    # PROD (what infra deploys)
- model_name: judge                     - model_name: judge
  litellm_params:                         litellm_params:
    model: groq/openai/gpt-oss-120b        model: openai/Qwen2.5-32B-Instruct
    api_key: os.environ/GROQ_API_KEY       api_base: http://vllm:8000/v1
                                           api_key: "none"
```

`model_name: judge` is unchanged, so MLflow keeps calling `openai:/judge`. **No eval
code, no MLflow config, no client change.** That is the migration.

### C.3 What the infra team does to go to production

1. Stand up the OSS judge server (vLLM/Ollama/TGI) serving e.g. Qwen2.5-32B.
2. In the proxy `config.yaml`, replace the `judge` block's `litellm_params` with the
   PROD variant above (local `api_base`, drop the Groq key).
3. Point `MLFLOW_TRACKING_URI` at the production MLflow server, set the prod
   experiment name.
4. Re-run the eval script unchanged → verify scores still land in MLflow.

That's it. Optionally keep Groq configured as a **fallback** for the judge in
LiteLLM's router, so eval jobs don't fail if the local judge is down.

---

## Demo walkthrough (for the infra team)

A ~10-minute script to run live. Assumes `examples/` from this repo.

**0. One-time setup**
```bash
cd examples
cp .env.example .env          # paste your GROQ_API_KEY
docker compose up -d          # LiteLLM :4000, MLflow :5000, Postgres
```

**1. Show tracing/observability is on** — make a normal proxy call:
```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" -H "Content-Type: application/json" \
  -d '{"model":"app-model","messages":[{"role":"user","content":"hi"}],
       "litellm_metadata":{"tags":["demo:trace"]}}'
```
Open **http://localhost:5000 → Traces**. Point out: every proxy request is captured
automatically (server-side callback), searchable by the `demo:trace` tag — zero
changes required from app teams.

**2. Run an evaluation** judged by the open-weight model on Groq:
```bash
export OPENAI_API_BASE=http://localhost:4000/v1
export OPENAI_API_KEY=sk-1234
export MLFLOW_TRACKING_URI=http://localhost:5000
python evals/run_eval.py
```
Open **MLflow → Experiments → litellm-eval**. Walk through per-row scores
(Correctness, RelevanceToQuery, tone Guidelines, custom faithfulness) and the
**judge's written rationale** on each row.

**3. The money slide — dev → prod is one line.** Show `litellm_config.yaml`: the
`judge` block is Groq today; the commented PROD block below it points at a local
vLLM. Emphasize: MLflow always calls `openai:/judge`, so swapping the judge to a
self-hosted OSS model is a config edit + proxy reload — **the eval code you just
ran does not change.**

**4. Close the loop (optional).** Show `mlflow.search_traces(...)` feeding real
logged traffic (from step 1) into `mlflow.genai.evaluate`, so in prod we grade
actual traffic, not just test prompts.

**Talking points to land**
- One gateway for app traffic *and* the judge → centralized keys, spend, rate limits.
- Data never leaves the network in prod (judge is self-hosted); no frontier cost.
- Observability and eval share the same traces → measure real quality over time.
- The demo *is* the production architecture, minus the judge endpoint.

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

1. **Two-phase rollout:** dev/demo on a local proxy with a **Groq** open-weight
   judge; production swaps the `judge` entry to a **self-hosted OSS** model. Only
   that one config block changes.
2. **Judge registered under a neutral name (`judge`)** so MLflow always calls
   `openai:/judge` + `OPENAI_API_BASE` → proxy, in both dev and prod.
3. **Self-hosted MLflow tracking server** (Postgres backend) — not Databricks. No
   `DATABRICKS_*` vars.
4. **Server-side tracing** via `success_callback`/`failure_callback: ["mlflow"]` so
   all teams are covered with zero client changes.
5. **`mlflow>=3.1.4`** baked into the proxy image.

## Open questions / next steps

- **Dev now:** confirm which Groq model to demo with (`openai/gpt-oss-120b` vs
  `qwen/qwen3-32b`) and get a `GROQ_API_KEY`.
- **Prod (infra team):** which OSS model for the judge, and how served (vLLM vs
  Ollama vs TGI)? Judge quality matters — a too-small model gives noisy verdicts.
  Validate against a small labeled set before trusting scores.
- Sizing: judge inference competes with app traffic for GPU. Consider a dedicated
  serving instance (still registered in the proxy).
- Where the MLflow server + Postgres run, backups, and who has UI access.
- Sampling policy for offline eval over prod traces (100% is expensive; sample by
  tag/route).
- Retention for traces/artifacts in the MLflow backing store.

See `examples/` for a runnable `docker-compose.yml`, `litellm_config.yaml`, and an
eval script.
