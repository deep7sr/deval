# MLflow + LiteLLM Integration — Progress & Handoff

> **Purpose of this doc:** a complete, self-contained snapshot so a new chat session can
> continue seamlessly. If you're a fresh session reading this: the human has a working
> MLflow + LiteLLM eval/observability setup on a corporate VM. Everything below is
> already built and verified unless marked ⏳/TODO. **The immediate next task is in
> §9: build "Path A" LLM-extract faithfulness.** Do NOT re-derive or rebuild what's
> in §3–§4; continue from §9.

---

## 1. Goal & context

Integrate **MLflow** (self-hosted, open-source) with an existing **LiteLLM Enterprise proxy**
for **(a) observability/tracing** and **(b) LLM-as-a-judge evaluations**. All AI traffic
already flows through the LiteLLM gateway. The **judge model must stay in-network / open-weight**
(no frontier cost, no data egress). Currently on **Groq** for dev/demo (open-weight models);
production will swap the judge to a self-hosted OSS model (vLLM/Ollama/TGI).

**Immediate objective:** a demo for the infra team showing tracing + a variety of eval metrics +
faithfulness, then hand it to infra to productionize.

---

## 2. Environment (corporate VM)

- **Host:** `RNDAZINLLML001`, **8 GB RAM shared across users, NO swap** → keep everything lean; hard OOM if memory spikes.
- **Home dir:** `/home/rajpd@ustrnd.com` (note the literal `@` in the path — avoid it in SQLite URIs by `cd`-ing into the data dir and using relative paths).
- **Corporate SSL interception** is active → TLS is MITM'd with a corporate CA. Consequences:
  - `pip` inside Docker builds fails cert verification unless the corp CA is baked in.
  - LiteLLM configs use `ssl_verify: false`.
- **LiteLLM runs in Docker.** Multiple stacks already running (do NOT disturb):
  - Guardrail work (separate project, untouched): `deploy-litellm-guardrail-1`, `deploy-demo-1`, `deploy-postgres-1`, `llm-eval-guardrail-demo-*`.
  - **Our MLflow demo container:** `litellm-mlflow-demo` (see §4).
- **Groq account model availability** (verify with `curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`):
  - ✅ `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`
  - ❌ `qwen/qwen3-32b` (not on this account — caused a `model_not_found`)

---

## 3. What's built & VERIFIED (do not rebuild)

| Capability | Status | How |
|---|---|---|
| LiteLLM → MLflow **tracing** | ✅ | `success_callback`/`failure_callback: ["mlflow"]`; captures success **and** failure traces |
| MLflow server (lean, RAM-safe) | ✅ | host venv, `--workers 1`, sqlite backend |
| Reliable **judge model** | ✅ | `llama-3.3-70b-versatile` (instruct) via the proxy |
| **Native scorers** in batch | ✅ | RelevanceToQuery, Safety, PIIDetection, Fluency, Guidelines, ResponseLength, RegexMatch (`eval_demo.py`) |
| **Eval on real captured traces** | ✅ | `search_traces()` → `mlflow.genai.evaluate(data=traces, ...)` (`eval_on_traces.py`) |
| **Auto-run** ("run on all future traces") | ✅ mechanism | requires `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true` (currently ON) |
| Faithfulness (Path A LLM-extract) | ✅ built | custom `make_judge` that extracts the marker from `{{ inputs }}` (`faithfulness_path_a.py`) — see §9 |
| Faithfulness (Path B deterministic) | ✅ built | unit-tested marker parser + `@scorer` judging clean evidence (`evidence_marker.py` + `faithfulness_path_b.py`) — see §9 |
| Faithfulness — VM validation run | ⏳ TODO | run `--validate` for A **and** B on the VM against `faithfulness_labeled_set.jsonl`; record agreement/precision (§9) |

---

## 4. Files & runtime state on the VM

### MLflow server (host venv)
- **Venv:** `~/mlflow-env` (mlflow **3.14.0**, plus `openai`, `litellm`, `deepeval` installed).
- **Data:** `~/mlflow/mlflow.db` (sqlite backend), `~/mlflow/artifacts/`.
- **Start script:** `~/mlflow/start-mlflow.sh` — runs:
  ```
  mlflow server --host 0.0.0.0 --port 5000 --workers 1 \
    --allowed-hosts "*" --cors-allowed-origins "*" \
    --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts
  ```
  (bound `0.0.0.0` + open allowed-hosts/CORS so the Docker container can reach it; dev-only.)
- **Job execution:** currently `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true` (enabled to test auto-run;
  costs ~1.4 GB RAM via huey workers). **Set back to `false` and restart to reclaim RAM when not demoing auto-run.**
- **UI:** `http://127.0.0.1:5000` (via SSH tunnel `ssh -L 5000:127.0.0.1:5000 ...`).

### LiteLLM demo container
- **Image:** `litellm-mlflow-demo:local`, built from `~/litellm-proxy/Dockerfile`:
  ```dockerfile
  FROM ghcr.io/berriai/litellm:v1.92.0
  USER root
  COPY corp-ca-bundle.crt /etc/pki/corp-ca-bundle.crt      # corp CA for pip TLS
  RUN /app/.venv/bin/python -m ensurepip --upgrade
  RUN /app/.venv/bin/python -m pip install --no-cache-dir --cert /etc/pki/corp-ca-bundle.crt "mlflow>=3.1.4"
  ```
  (`~/litellm-proxy/corp-ca-bundle.crt` = copy of host CA bundle from `python3 -c "import ssl;print(ssl.get_default_verify_paths().openssl_cafile)"`.)
- **Run command:**
  ```
  docker run -d --name litellm-mlflow-demo \
    --add-host host.docker.internal:host-gateway \
    -p 4010:4000 \
    -e GROQ_API_KEY="$GROQ_API_KEY" \
    -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" \
    -e MLFLOW_EXPERIMENT_NAME="litellm-proxy-demo" \
    -v ~/litellm-proxy/config.mlflow-demo.yaml:/app/config.yaml:ro \
    litellm-mlflow-demo:local --config /app/config.yaml --port 4000
  ```
  - GROQ_API_KEY was pulled from a running guardrail container: `docker exec deploy-litellm-guardrail-1 printenv GROQ_API_KEY`.
  - Container reaches host MLflow via `host.docker.internal` (Docker gateway `172.21.0.1`).
- **Config:** `~/litellm-proxy/config.mlflow-demo.yaml`
  ```yaml
  model_list:
    - model_name: groq-test-model      # app-under-test
      litellm_params: { model: groq/openai/gpt-oss-120b, api_key: os.environ/GROQ_API_KEY }
    - model_name: judge-model          # reliable instruct judge
      litellm_params: { model: groq/llama-3.3-70b-versatile, api_key: os.environ/GROQ_API_KEY }
  litellm_settings:
    ssl_verify: false
    success_callback: ["mlflow"]
    failure_callback: ["mlflow"]
  general_settings:
    master_key: sk-123
  ```
- **Proxy URL:** `http://localhost:4010/v1`, master key `sk-123`.

### Eval scripts (on the VM, `~/litellm-proxy/`)
- **`eval_demo.py`** — variety pack of native scorers on a crafted 5-row dataset (quality/safety/PII/policy/length). Reliable in batch.
- **`eval_on_traces.py`** — sends fresh traffic through the proxy, pulls those traces via `search_traces`, runs `mlflow.genai.evaluate(data=traces, ...)`. **This is the production pattern (eval on captured traces).**

### MLflow experiments
- **`litellm-proxy-demo`** (exp id **2**): all proxy traffic (app **and** judge calls — see pollution note §6).
- **`litellm-eval`** (exp id **3**): synthetic/batch eval runs.

### Judge routing for code-based scorers (the env vars used when running evals)
```
export OPENAI_API_KEY=sk-123
export OPENAI_API_BASE=http://127.0.0.1:4010/v1     # the LiteLLM proxy, NOT api.openai.com
export OPENAI_BASE_URL=http://127.0.0.1:4010/v1
export DEEPEVAL_TELEMETRY_OPT_OUT=YES               # silence PostHog SSL noise
```
MLflow judge model string = **`openai:/judge-model`** → resolves through the proxy → Groq llama-3.3-70b.

### UI Judges tab (native no-code judges)
- `groq-endpoint` = an **MLflow AI Gateway endpoint pointing directly at Groq** (bypasses LiteLLM proxy → no pollution). Currently `llama-3.1-8b-instant` — **too weak; switch to a 70B endpoint**.
- `judge-test` judge currently uses **Retrieval Groundedness** → **errors** (`SCORER_ERROR: No retrieval context found... requires a span with type RETRIEVER`). Switch to `RelevanceToQuery` or a custom judge.

---

## 5. Decisions LOCKED IN

1. **MLflow self-hosted** (sqlite dev → **PostgreSQL prod**, artifacts → object store). Not Databricks; no `DATABRICKS_*` vars.
2. **Native MLflow scorers are the eval backbone**, NOT DeepEval (see §6 batch bug).
3. **Judge = `llama-3.3-70b-versatile`** (instruct). Reasoning models are unreliable judges.
4. **Faithfulness = a custom judge** (native `RetrievalGroundedness` is unusable on proxy-only traces).
5. **Faithfulness contract (opt-in):** developers who want faithfulness put retrieved context in a
   **`role: assistant` (or `user`) message whose content starts with the exact marker
   `--- Retrieved Evidence ---`**, followed by the evidence (one chunk per line); the actual question
   is the final `user` message. Other metrics (relevance/safety/PII) stay **contract-free/transparent**.
6. **Faithfulness runs OFFLINE + SAMPLED** (not on 100% of live traffic — too costly per call).
7. **Build faithfulness Path A first** (LLM-extract, no code), test it; if it works, build **Path B**
   (deterministic code parser) for production precision. *(This is the next task — §9.)*

---

## 6. HARD-WON LESSONS (do not relearn these)

1. **Judge reliability is everything.** `gpt-oss-120b` (a *reasoning* model) produced inconsistent
   JSON → contradictory verdicts (rationale said "score 1.00" but the boolean was "No"). Switching to
   `llama-3.3-70b-versatile` (*instruct*) fixed it. **Use instruct judges; validate against a small labeled set.**
2. **DeepEval scorers via `mlflow.genai.evaluate` have a BATCH BUG.** Stateful DeepEval metric objects
   are reused across rows → verdicts cross-contaminate (a correct row like "Paris" flips to Fail in a
   multi-row batch, correct alone). Works single-row/per-example. **→ Use MLflow-native scorers for
   batch; only use DeepEval per-example.** (Native `RelevanceToQuery` = DeepEval AnswerRelevancy equivalent, and it works.)
3. **Judge-through-proxy = trace pollution.** Every scorer call is itself traced into the app experiment
   (huge judge prompts, e.g. 3,580 tokens each; 5 scorers × N rows). Auto-run amplifies it and can create
   a **feedback loop** (judge trace → evaluated → more traces). **→ For prod, route the judge OFF the traced
   path** (MLflow AI Gateway direct to Groq, or a separate un-traced key/experiment). This is a concrete input
   to the judge-routing decision (§7).
4. **`RetrievalGroundedness` (native faithfulness) needs a `RETRIEVER` span**, which proxy-only traces
   don't have (retrieval happens app-side, *upstream* of the proxy; by the time the request hits LiteLLM the
   docs are just text inside the prompt). **→ faithfulness on gateway traces needs whole-input context or the marker contract.**
5. **MLflow 3.14 operational gotchas:**
   - Server-side **job-execution** (huey workers) eats ~1.4 GB → disable (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`) on low RAM; enable only for auto-run.
   - **Security middleware is localhost-only by default** → need `--host 0.0.0.0 --allowed-hosts "*" --cors-allowed-origins "*"` for the container to reach it.
   - `search_traces` uses **`locations=`** (not the deprecated `experiment_ids=`).
   - Use **`--workers 1`** — multi-worker default caused an OOM worker-respawn death loop on this VM.
6. **`make_judge` / MLflow judges only allow RESERVED template vars:** `{{ inputs }}`, `{{ outputs }}`,
   `{{ expectations }}`, `{{ trace }}`, `{{ conversation }}`. **You cannot inject a pre-parsed
   `{{ retrieval_context }}`.** Context must arrive via `{{ inputs }}` (the raw messages, which contain the
   marker). This is why Path A relies on the LLM to extract the marker from `{{ inputs }}`.
7. **Corporate SSL:** bake the corp CA into the container image for `pip` (`--cert /etc/pki/corp-ca-bundle.crt`); host `pip` already trusts it.

---

## 7. OPEN decisions (for the tech lead)

1. **Judge routing: through the proxy vs. outside it.** Through = centralized governance/keys/spend, but
   **pollutes app traces + inflates spend + feedback-loop risk on auto-run**. Outside (e.g. MLflow AI Gateway
   direct to Groq / self-hosted) = clean, but bypasses the Enterprise gateway's governance. *(Concrete pollution
   evidence already gathered — 100s of judge traces in `litellm-proxy-demo`.)*
2. **Faithfulness contract: mandatory vs. opt-in.** Decided opt-in for now (teams that want faithfulness add
   the marker; everyone still gets contract-free relevance/safety/PII).
3. **Automation:** MLflow **scheduled scorers** (native, needs job-execution backend + RAM) vs. an **external
   scheduler** (cron/Airflow/K8s CronJob running the eval script) — the latter is recommended given RAM limits & control.
4. **Prod judge model & serving:** which OSS instruct model (e.g. Llama-3.3-70B-Instruct / Qwen2.5-72B-Instruct),
   served on vLLM/Ollama/TGI, registered in the proxy (or as an AI Gateway endpoint).
5. **Prod storage/ops:** Postgres backend, object-store artifacts, trace **retention** policy (traces accumulate fast), sampling rate.

---

## 8. How things flow (for reference)

- **Tracing:** app → LiteLLM proxy → (on completion) the `mlflow` success/failure callback (running inside
  the container) uses the mlflow SDK to send the trace to `MLFLOW_TRACKING_URI` → stored under
  `MLFLOW_EXPERIMENT_NAME`. Automatic, async, zero app code change.
- **Storage:** MLflow **backend store** = SQL DB (sqlite now, Postgres in prod) holds experiments/traces/
  spans/assessments; **artifact store** = files/object storage for large blobs.
- **Evals:** `mlflow.genai.evaluate(data=<traces or dataset>, scorers=[...])`. Data can be a hand-built
  dataset OR real captured traces via `mlflow.search_traces(locations=[exp_id], ...)`. Automatable via
  external scheduler or native scheduled scorers.

---

## 9. ✅ Faithfulness — Path A **and** Path B BUILT. Remaining: run the VM validation.

Both paths are implemented and committed under `examples/evals/`. The deterministic
half (Path B's parser) is unit-tested and green locally; the judge-dependent runs
(Path A end-to-end, Path B's pass/fail verdicts) still need to be executed **on the
VM** where the proxy + MLflow + Groq are reachable — this container has no MLflow/Groq.

### Files (all in `examples/evals/`)
| File | What it is |
|---|---|
| `evidence_marker.py` | **Deterministic contract parser** — exact marker, nearest-preceding rule, plain-string-only content, skip-if-absent. Pure Python, no deps. Path B core. |
| `test_evidence_marker.py` | Unit tests for the parser (10 cases: typo/spacing, multi-turn, content-parts array, empty evidence, user-role evidence, …). **10/10 pass locally.** |
| `faithfulness_path_a.py` | **Path A** — one `make_judge` custom judge that extracts the marker from `{{ inputs }}` itself. `--validate` scores the labeled set; `--sample-traces` scores real traffic. |
| `faithfulness_path_b.py` | **Path B** — `@scorer` that parses deterministically then calls the instruct judge over *clean* evidence (skips silently when no marker). Plugs into `mlflow.genai.evaluate` / scheduled scorers. |
| `faithfulness_labeled_set.jsonl` | **16 hand-labeled examples** (pass/fail/skip) shared by A & B: faithful, hallucinated-number/entity/channel/negation, faithful-refusal, multi-turn-nearest, multi-turn-against-stale, no-marker. All 16 are consistent with the deterministic parser (verified locally). |

### Design notes (what was decided while building)
- **Path A** uses one `make_judge` judge; it can only see `{{ inputs }}` + `{{ outputs }}`
  (reserved template vars only — handoff §6.6), so the marker rides in via `{{ inputs }}`
  and the LLM extracts it. Emits categorical `pass`/`fail`/`skip`.
- **Path B** does NOT use `make_judge` (can't inject a pre-parsed `{{ retrieval_context }}`).
  It calls the judge directly through the proxy (`temperature=0`, JSON response) over the
  parser's clean evidence, and wraps the result as an MLflow `Feedback`/`@scorer`. Returns
  `None` to **skip** contract-free traffic (never a false fail). The parser matches the
  guardrail contract exactly and is the one place the correctness risk lives — hence the
  dedicated unit tests.
- **Judge model:** `openai:/judge-model` → `llama-3.3-70b-versatile` (instruct) via the proxy
  for code runs; a 70B AI Gateway endpoint for UI/auto-run. Overridable via
  `FAITHFULNESS_JUDGE` / `FAITHFULNESS_JUDGE_MODEL`.
- **Execution:** offline + **sampled** — `--sample-traces` over-fetches then keeps only
  marker-bearing traces (via `search_traces(locations=[...])`) so the judge isn't run on
  traffic that would just skip.

### ⏭️ What's left (do this on the VM)
1. Export the judge env (§10), then run **both** validators and record the numbers:
   ```bash
   ~/mlflow-env/bin/python examples/evals/faithfulness_path_a.py            # 5 §9 cases
   ~/mlflow-env/bin/python examples/evals/faithfulness_path_a.py --validate # labeled-set agreement
   ~/mlflow-env/bin/python examples/evals/faithfulness_path_b.py            # labeled-set agreement
   python examples/evals/test_evidence_marker.py                            # parser (no venv needed)
   ```
2. **Compare A vs B on the two multi-turn rows** (`multi_turn_uses_nearest`,
   `multi_turn_hallucinated_against_nearest` / Path A case `4b`). Path A is fuzzy here —
   it depends on the LLM honoring "nearest-preceding marker"; Path B enforces it in code.
   Record whether Path A gets them right. This is the concrete evidence for "is the code
   parser worth it."
3. If Path A agreement is high and false-fails are low → Path A is demoable; **prefer Path B
   for production** (deterministic, reproducible, cheaper — skip is free, no judge call).
4. Feed results into the tech-lead decisions (§7): judge routing, opt-in vs mandatory contract.

**Constraints respected in the build:** offline+sampled; judge = instruct 70B; skip is free
(no judge call on contract-free traffic → RAM/spend friendly); nothing here routes heavy judge
traffic through the traced proxy by default beyond the existing dev setup (pollution — §6.3).

---

## 10. Key commands (reproducibility)

```bash
# --- MLflow server (lean). For auto-run testing, MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true (uses ~1.4GB). ---
bash ~/mlflow/start-mlflow.sh          # or the inline command in §4
curl -s http://127.0.0.1:5000/health   # -> OK
free -h                                # watch available RAM (8GB, no swap)

# --- Judge routing env (for code-based evals) ---
export OPENAI_API_KEY=sk-123
export OPENAI_API_BASE=http://127.0.0.1:4010/v1
export OPENAI_BASE_URL=http://127.0.0.1:4010/v1
export DEEPEVAL_TELEMETRY_OPT_OUT=YES

# --- Run the demos ---
~/mlflow-env/bin/python ~/litellm-proxy/eval_demo.py         # native variety pack
~/mlflow-env/bin/python ~/litellm-proxy/eval_on_traces.py    # eval on real captured traces

# --- Faithfulness (§9): Path A (LLM-extract) & Path B (deterministic parse) ---
python examples/evals/test_evidence_marker.py                       # parser unit tests (no venv/network)
~/mlflow-env/bin/python examples/evals/faithfulness_path_a.py            # 5 §9 test cases
~/mlflow-env/bin/python examples/evals/faithfulness_path_a.py --validate # Path A agreement on labeled set
~/mlflow-env/bin/python examples/evals/faithfulness_path_b.py            # Path B agreement on labeled set
~/mlflow-env/bin/python examples/evals/faithfulness_path_b.py --sample-traces  # Path B on sampled real traces

# --- Send traffic through the proxy (generates a trace) ---
curl -s http://localhost:4010/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","messages":[{"role":"user","content":"..."}],
       "litellm_metadata":{"tags":["demo:x"]}}'

# --- Check judge container / Groq models ---
docker logs litellm-mlflow-demo --tail 30
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"

# --- Reclaim RAM after auto-run demo ---
# edit MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false, then restart mlflow (pkill -f mlflow-env; bash ~/mlflow/start-mlflow.sh)
```

---

## 11. Quick "am I set up?" checklist for a new session

- [ ] MLflow server up: `curl -s http://127.0.0.1:5000/health` → `OK`
- [ ] `litellm-mlflow-demo` container running: `docker ps | grep litellm-mlflow-demo` (port 4010)
- [ ] A test call to `:4010` returns a completion (key valid)
- [ ] Judge env vars exported (§10)
- [ ] Decide: job execution ON (auto-run, +1.4GB RAM) or OFF (reclaim RAM)
- [ ] **§9 faithfulness is BUILT (A + B).** Remaining: run the two validators on the VM and record
      agreement/precision + the A-vs-B multi-turn comparison (see §9 "What's left").

---

*Not yet done: the **VM validation run** for faithfulness (code for Path A + Path B is committed and the
parser is unit-tested; the judge-dependent runs still need the VM), external-scheduler automation script,
the production handoff/runbook, and the tech-lead decisions in §7. The general integration guide + Groq
examples live in `mlflow-litellm-integration.md` and `examples/` in this repo; the earlier guardrail design
is in `litellm-deepeval-guardrail-context.md`.*
