# LiteLLM Hallucination Guardrail (DeepEval Faithfulness)

A custom LiteLLM guardrail that detects **ungrounded (hallucinated) responses**
and prevents them from reaching the user. It scores each response against the
retrieved evidence supplied in the request using DeepEval's `FaithfulnessMetric`,
and — when a response is not grounded — retries the same model with targeted
correction before falling back to a safe message.

- **Runs as:** a LiteLLM custom `CustomGuardrail` on the `post_call` hook.
- **Scope (v1):** non-streaming (`stream: false`) chat completions.
- **Judge:** any model reachable via the LiteLLM SDK (validated with Groq).

---

## How it works

```
request ──► [1] Parser ──► non-compliant? ──► allow, skip (guardrail does not run)
                    │
                    └► compliant ──► [2] Grounding judge (DeepEval FaithfulnessMetric)
                                          │
                                          ├─ passed ──► allow (return original response)
                                          └─ failed ──► [3] by MODE:
                                                          block     → replace with fallback
                                                          remediate → retry ≤3× → fallback
```

Three decoupled components (see `guardrail/`):

| Component | File | Responsibility |
|---|---|---|
| **Parser** | `guardrail/parser.py` | Deterministic contract logic: extract `input` + `retrieval_context` from the raw `messages`, or signal "skip". No LLM, no network. |
| **Grounding judge** | `guardrail/grounding.py`, `guardrail/groq_judge.py` | Runs `FaithfulnessMetric` via the judge model; returns score, pass/fail, and the specific unsupported claims. |
| **Retry loop** | `guardrail/remediation.py` | Re-prompts the same model with the unsupported claims, stops early on success, bounds attempts + wall-clock time, falls back if still ungrounded. |
| **Hook (orchestrator)** | `guardrail/hook.py` | The `CustomGuardrail` that wires the above onto `async_post_call_success_hook`. Fail-open on unexpected errors. |

---

## The evidence contract (what calling teams must do)

The guardrail **only runs when the request follows this contract**. Non-compliant
requests are passed through untouched (silently skipped).

1. Retrieved evidence goes in a message with `"role": "assistant"`.
2. That message's `content` must **start with the exact marker**:
   `--- Retrieved Evidence ---`
3. Everything after the marker is evidence — **one fact per line**.
4. The question is the final `"role": "user"` message.
5. `content` must be a **plain string** (not a content-parts array) for both the
   evidence message and the user message.
6. **Multi-turn:** if several marker messages exist, only the one **immediately
   preceding the final user message** is used; older ones are ignored.
7. **No valid marker → the guardrail does not run** for that request.

### Example
```json
{
  "model": "groq-test-model",
  "messages": [
    { "role": "system", "content": "You are a factual assistant. Answer using only the retrieved evidence." },
    { "role": "assistant", "content": "--- Retrieved Evidence ---\nAll customers are eligible for a 30 day full refund at no extra cost.\nRefunds are processed within 5 business days." },
    { "role": "user", "content": "What is the refund policy?" }
  ]
}
```

---

## Repository layout

```
guardrail/            The guardrail package (this is what gets deployed)
  config.py           All tunables, overridable via env vars
  parser.py           Contract parser
  grounding.py        DeepEval FaithfulnessMetric wrapper
  groq_judge.py       Judge model (litellm-backed DeepEvalBaseLLM)
  remediation.py      Self-correction retry loop
  hook.py             LiteLLM CustomGuardrail (post_call)
tests/                Unit tests (28, no network) — run with: python -m pytest tests/ -q
deploy/               Self-contained test stack (Dockerfile, compose, config.yaml)
scripts/              Live smoke scripts
```

---

## Try it locally (self-contained test stack)

A one-container stack on **port 4001** (won't collide with a proxy on 4000).
See `deploy/README.md` for details.

```bash
cd deploy
cp .env.example .env          # set your real GROQ_API_KEY
docker compose up --build
```

Then send a request with the marker (see the contract example above) to
`http://localhost:4001/v1/chat/completions` with header
`Authorization: Bearer sk-123`. Guardrail decisions are logged as `GUARDRAIL …`
lines in `docker compose logs`.

### Browser demo (for non-technical audiences)

The stack also serves a point-and-click demo at **http://localhost:8090**.
Click a preset case (grounded / hallucinated / partly-unsupported / no-evidence)
and it generates the answer through the proxy, then shows the real faithfulness
score, the verdict, the flagged claims, and what a guarded user would receive —
no terminal needed.

---

## Deploying to the production LiteLLM — step by step

The guardrail is **additive**: it drops into an existing proxy without changing
`model_list` or existing `litellm_settings`. Three things must be true in the
production deployment.

### Step 1 — Install DeepEval in the proxy image

DeepEval must be importable by the proxy process. Add to the production
Dockerfile (adjust the venv path to match your image; the official image uses
`/app/.venv`):

```dockerfile
RUN /app/.venv/bin/python -m pip install --no-cache-dir "deepeval==4.1.0"
```

> **Corporate CA:** if the build environment intercepts TLS, either bake the
> corporate CA into the image (`COPY corp-ca.crt … && update-ca-certificates`)
> or use the internal PyPI mirror (`--index-url`). The `--trusted-host` shortcut
> used in `deploy/Dockerfile` is fine for testing but not recommended for prod.
> See `deploy/README.md`.

### Step 2 — Make the `guardrail/` package importable

The proxy's guardrail loader resolves `guardrail.hook.HallucinationGuardrail`
**relative to the config file's directory**, and the package's relative imports
require its parent dir on `PYTHONPATH`. Two supported options:

- **Bake into the image (recommended for prod):**
  ```dockerfile
  COPY guardrail /app/guardrail
  ENV PYTHONPATH=/app
  ```
  (place `config.yaml` at `/app/config.yaml` so the loader finds `/app/guardrail/hook.py`)

- **Volume-mount (used by the test stack, good for iteration):**
  mount the `guardrail/` directory to `/app/guardrail` and set `PYTHONPATH=/app`.

### Step 3 — Add the guardrail to the production `config.yaml`

Append this block to the existing config (nothing else needs to change):

```yaml
guardrails:
  - guardrail_name: "hallucination-guardrail"
    litellm_params:
      guardrail: guardrail.hook.HallucinationGuardrail
      mode: "post_call"
      default_on: true       # runs on all traffic (open-source). See Step 5.
```

**Enablement model — read this before choosing `default_on`:**

| Model | Open-source | Enterprise |
|---|---|---|
| Global (`default_on: true`) | ✅ | ✅ |
| Per-key / per-team opt-in | ❌ needs `LITELLM_LICENSE` | ✅ |

Attaching guardrails to a virtual key (`/key/generate` with
`"guardrails": [...]`) is an **Enterprise** feature — on open-source it returns
HTTP 403. So on open-source the choice is **global** (`default_on: true`) or
**per-request** (the calling app includes `"guardrails":
["hallucination-guardrail"]` in each request body).

### Step 4 — Provide the judge API key to the container

The judge model needs its key in the **container** environment (not just the
host shell). For the validated Groq judge:

```
GROQ_API_KEY=gsk_…
```

### Step 5 — (Enterprise only) Enable per-key / per-team

> **Requires a LiteLLM Enterprise license.** On open-source, skip this — the
> guardrail runs globally via `default_on: true` (Step 3). Attaching guardrails
> to a key on open-source returns HTTP 403.

With Enterprise, the guardrail can be **always-on for any key it is attached
to**, and off for every other key. A team requests coverage; infra enables it on
that team's key — no config edit or restart per team.

**Attach to a new key:**
```bash
curl -X POST 'http://<proxy>/key/generate' \
  -H 'Authorization: Bearer <MASTER_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"guardrails": ["hallucination-guardrail"], "metadata": {"team": "team-name"}}'
```

**Attach to an existing key:**
```bash
curl -X POST 'http://<proxy>/key/update' \
  -H 'Authorization: Bearer <MASTER_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"key": "sk-...the-team-key...", "guardrails": ["hallucination-guardrail"]}'
```

Requests made with that key now always run the guardrail; the team cannot turn
it off. (The same `guardrails` list can also be set on a **team** via
`/team/update`, or through the Admin UI.)

### Step 6 — Verify

Restart/redeploy and confirm the proxy boots with no import errors. Then send a
marker request **using a key that has the guardrail attached** and look for
`GUARDRAIL … verdict: … passed=…` in the logs. A request with a key that does
NOT have it attached should pass through with no guardrail log line.

---

## Configuration reference (env vars)

All read by `guardrail/config.py`; all optional (defaults shown).

| Env var | Default | Meaning |
|---|---|---|
| `GUARDRAIL_MODE` | `remediate` | `block` \| `remediate` |
| `GUARDRAIL_JUDGE_MODEL` | `groq/llama-3.3-70b-versatile` | Judge model (any litellm SDK id) |
| `GUARDRAIL_JUDGE_API_BASE` | *(unset)* | Endpoint for a self-hosted judge (Ollama/vLLM/TGI) |
| `GUARDRAIL_JUDGE_API_KEY` | *(unset)* | Key for the judge endpoint (dummy for local servers) |
| `GUARDRAIL_JUDGE_JSON_MODE` | `true` | Request JSON-mode output; set false for models that don't support it |
| `GUARDRAIL_FAITHFULNESS_THRESHOLD` | `0.7` | Pass if score ≥ threshold |
| `GUARDRAIL_MAX_RETRIES` | `3` | Corrective retries before fallback |
| `GUARDRAIL_RETRY_TIME_BUDGET_SECONDS` | `30` | Wall-clock cap on the retry loop |
| `GUARDRAIL_RETRY_TEMPERATURE` | `0.3` | Temperature for retry regenerations |
| `GUARDRAIL_EVIDENCE_MARKER` | `--- Retrieved Evidence ---` | Contract marker string |
| `GUARDRAIL_FALLBACK_MESSAGE` | *(safe message)* | Returned when all retries fail |
| `GUARDRAIL_SSL_VERIFY` | `false` | TLS verify for the guardrail's own calls |
| `GUARDRAIL_LOG_LEVEL` | `INFO` | Guardrail logger level |

---

## Swapping the judge model (e.g. to a self-hosted open-source model)

The judge is addressed entirely through the LiteLLM SDK, so changing it is
**configuration, not code** — set env vars, no code edits:

- **Ollama** (e.g. Gemma): serve `ollama pull gemma3:4b`, then
  ```
  GUARDRAIL_JUDGE_MODEL=ollama_chat/gemma3:4b
  GUARDRAIL_JUDGE_API_BASE=http://<ollama-host>:11434
  ```
- **vLLM / TGI / any OpenAI-compatible server**:
  ```
  GUARDRAIL_JUDGE_MODEL=openai/google/gemma-3-4b-it
  GUARDRAIL_JUDGE_API_BASE=http://<server>:8000/v1
  GUARDRAIL_JUDGE_API_KEY=<dummy-or-real>
  ```
- If the model errors on JSON response_format, set `GUARDRAIL_JUDGE_JSON_MODE=false`.

> **Accuracy caveat:** a small model (e.g. 4B) is a much weaker judge than a
> 70B. Faithfulness judging (claim extraction + verification + JSON output) is
> demanding, and small models produce noisier verdicts and less reliable JSON.
> **Validate the chosen judge on a labelled set and tune the threshold before
> trusting it to block** — this is the single most important production check.

## Operating modes

- **`block`** — replace an ungrounded response with the fallback message (no retry).
- **`remediate`** *(default)* — retry the same model with the specific unsupported
  claims (≤ `MAX_RETRIES`, within the time budget), then fall back if still ungrounded.

---

## Known limitations (by design, documented)

- **Non-streaming only.** LiteLLM's `post_call` guardrails cannot block streamed
  responses; requests with `stream: true` are not covered.
- **Added latency + cost.** Faithfulness is 2+ judge LLM calls; remediation adds a
  regeneration + re-score per retry. Budgeted via `RETRY_TIME_BUDGET_SECONDS`.
- **Contract trust.** Anything that can write an `assistant` message could forge
  the marker. The contract is a routing mechanism, not an authentication one.
- **Staleness.** "Nearest preceding marker" is a proxy for "correct context"; an
  app that doesn't re-retrieve for a new sub-topic may be judged against stale
  evidence (possible false flags, not a parsing bug).

---

## Tests

```bash
python -m pytest tests/ -q      # 28 unit tests, no network required
```
