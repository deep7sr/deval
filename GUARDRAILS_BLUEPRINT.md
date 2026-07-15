# DeepEval RAG Guardrails for LiteLLM — Blueprint & Context

**Purpose of this document.** We built one guardrail (DeepEval **Faithfulness**)
end-to-end on a LiteLLM proxy. We now want to build the other DeepEval RAG-metric
guardrails using the **same proven pattern**, starting with **Answer Relevancy**.
This file is the self-contained context to hand to a fresh session: it records
how the faithfulness guardrail is designed, the reusable pattern, exactly how the
metrics differ, and the rules to avoid mixups between guardrails.

Read this top to bottom before writing any code for a new guardrail.

---

## 0. TL;DR for building the next guardrail

1. A guardrail = **parser → evaluator (one DeepEval metric) → hook (+ optional
   remediation)**, wired into LiteLLM as a `CustomGuardrail` on `post_call`.
2. **Only reference-free metrics can be live guardrails.** Answer Relevancy,
   Contextual Relevancy, and Faithfulness qualify. Contextual **Precision** and
   **Recall** need a ground-truth `expected_output`, which does not exist at
   inference time → they can only be **offline** evaluations, not live guardrails.
3. Each metric needs **different LLMTestCase fields** — see the table in §4. That
   determines what the parser must extract.
4. **No mixups:** every guardrail gets its own class, its own `guardrail_name`,
   its own namespaced env-config, and is registered + enabled independently.
   Shared plumbing (judge, parser helpers, remediation) is fine; identity is not.
5. Follow the same **methodology** that worked (§8): verify against source first,
   build incrementally, unit-test the deterministic parts, test live.

---

## 1. What already exists (the Faithfulness guardrail)

A working guardrail that blocks ungrounded answers. Built and validated on a live
LiteLLM proxy (Groq judge), packaged for Docker, and handed to infra.

**Repo branches:**
- `claude/litellm-deepeval-guardrail-review-c7dzhd` — full working branch
  (guardrail code, tests, `deploy/` sandbox stack, browser demo, this doc).
- `guardrail-infra-handoff` — minimal production drop (just `guardrail/` +
  a concise install README).

**Package layout (`guardrail/`):**

| File | Responsibility |
|---|---|
| `config.py` | All settings as env-overridable constants (marker, threshold, judge model, retries, SSL, log level, mode). |
| `parser.py` | Deterministic "contract" logic. Turns the raw `messages` array into the clean fields DeepEval needs, or a "skip" signal. No LLM, no network. Heavily unit-tested. |
| `groq_judge.py` | `LiteLLMJudge(DeepEvalBaseLLM)` — the judge model, addressed through the **LiteLLM SDK**, so it's provider-agnostic (Groq, self-hosted Ollama/vLLM, etc.). `GroqJudge` alias kept. |
| `grounding.py` | `GroundingEvaluator` — wraps `FaithfulnessMetric`, returns a compact verdict (score, pass/fail, reason, unsupported claims). This is the **metric-specific** module. |
| `remediation.py` | The self-correction retry loop (pure, injectable, unit-tested). |
| `hook.py` | `HallucinationGuardrail(CustomGuardrail)` — the orchestrator on `async_post_call_success_hook`. Fail-open. |

---

## 2. The reusable architecture

Four decoupled pieces. For a new metric, **three are reused almost verbatim**
and **one is metric-specific**:

```
LiteLLM post_call hook (orchestrator)      hook.py         mostly reusable (subclass)
   │
   ├─ Parser (extract test-case fields)    parser.py       reuse; adjust which fields it produces
   ├─ Evaluator (run the DeepEval metric)  grounding.py    METRIC-SPECIFIC — the new work
   └─ Remediation (retry loop)             remediation.py  reuse; new correction prompt only
Judge model (LiteLLM SDK)                  groq_judge.py   reuse as-is
Config (env vars)                          config.py       add namespaced settings
```

**Design principles carried over (keep these):**
- **Decoupled:** parser knows nothing about DeepEval/LiteLLM; evaluator knows
  nothing about message arrays; hook is a thin orchestrator.
- **Judge via LiteLLM SDK**, provider-agnostic, one env var to swap
  (`GUARDRAIL_JUDGE_MODEL` + optional `GUARDRAIL_JUDGE_API_BASE/_API_KEY`).
- **Judge reused across requests; a fresh metric object per request** (metrics
  hold per-measurement state — never share one across concurrent requests).
- **Fail-open:** any guardrail error logs and returns the original response.
- **Config via env** (namespaced constants, nothing hardcoded inline).
- **Structured logging** of every decision (its own stdout logger).
- **Deterministic parts are unit-tested** (parser, remediation) with no network.

---

## 3. LiteLLM integration facts (verified from source — reuse these)

- Base class: `from litellm.integrations.custom_guardrail import CustomGuardrail`.
- Hook: `async def async_post_call_success_hook(self, data, user_api_key_dict, response)`.
  - `data["messages"]` = the request messages; `data["model"]` = the model id
    the caller used (reuse this to route retries to the **same** model).
  - `response.choices[0].message.content` = the answer (`actual_output`).
  - Returning the (modified) `response` replaces it; raising blocks it.
- Registration in `config.yaml`:
  ```yaml
  guardrails:
    - guardrail_name: "<name>"
      litellm_params:
        guardrail: <module.path>.<ClassName>   # e.g. guardrail.hook.HallucinationGuardrail
        mode: "post_call"
        default_on: false
  ```
- **Loader gotcha (important):** LiteLLM's `get_instance_fn` **file-loads** the
  class relative to the config file's directory (`<config_dir>/<module>/<file>.py`).
  For a package with relative imports to work, the package must sit **next to
  config.yaml** AND its parent dir must be on `PYTHONPATH`. (This is why the
  Docker image does `COPY guardrail /app/guardrail` + `ENV PYTHONPATH=/app` with
  config at `/app/config.yaml`.)
- **Per-key / per-team enablement is a LiteLLM *Enterprise* feature**
  (`/key/generate` or `/key/update` with `"guardrails": ["<name>"]`). On
  open-source it returns HTTP 403; the alternatives are global `default_on: true`
  or per-request `guardrails` in the body. Our production infra has Enterprise,
  so guardrails are enabled per key.
- **Scope:** `post_call` cannot block **streaming** responses — v1 covers
  `stream: false` only.

---

## 4. DeepEval metric field requirements (THE key table — verified from source)

Each metric requires different `LLMTestCase` fields. This determines (a) what the
parser must produce and (b) **whether the metric can be a live guardrail at all.**

| Metric | Required fields | Reference-free? | Live guardrail? |
|---|---|---|---|
| **Faithfulness** (built) | `input`, `actual_output`, `retrieval_context` | ✅ | ✅ |
| **Answer Relevancy** (next) | `input`, `actual_output` | ✅ | ✅ |
| **Contextual Relevancy** | `input`, `retrieval_context` | ✅ | ✅ |
| **Contextual Precision** | `input`, `retrieval_context`, **`expected_output`** | ❌ | ❌ offline only |
| **Contextual Recall** | `input`, `retrieval_context`, **`expected_output`** | ❌ | ❌ offline only |

Where the fields come from at inference time:
- `input` = content of the final `role: user` message.
- `actual_output` = the LLM's completion (`response.choices[0].message.content`).
- `retrieval_context` = the evidence lines parsed from the `--- Retrieved
  Evidence ---` marker message.
- `expected_output` = a **ground-truth reference answer**. **Not available in a
  live request.** → Precision/Recall cannot run as live guardrails; they belong
  in an offline/batch evaluation over a labelled dataset (e.g. `deepeval test
  run`, or scored over logged traffic). **Do not attempt to build them as live
  guardrails** — flag this to the user if asked.

> ⚠️ Always re-verify a metric's `_required_params` against the installed
> DeepEval version before building (we pin `deepeval==4.1.0`). The table above
> was read from DeepEval source.

---

## 5. What the parser produces per metric

The current `parser.py` returns `{input, retrieval_context}` (+ skip reasons).
For a new metric, decide which fields it needs and whether the **evidence marker
contract is still required**:

- **Faithfulness / Contextual Relevancy** — need `retrieval_context` → **require
  the marker** (skip if absent), exactly like today.
- **Answer Relevancy** — needs only `input` + `actual_output`, **no context**. So
  it does **not** need the evidence marker at all. Design decision for the new
  session (see §7).

Keep parser changes additive/back-compatible: `parse_messages()` already finds
the final user message (`input`) and the nearest preceding marker
(`retrieval_context`). A relevancy guardrail can reuse the "final user message"
extraction and ignore the marker parts.

---

## 6. Metric-specific evaluator + remediation (the pattern to copy)

**Evaluator** (`grounding.py` is the template). For a new metric, create a
parallel module (e.g. `relevancy.py`) that:
- instantiates the DeepEval metric with the shared `LiteLLMJudge` and its own
  threshold;
- builds an `LLMTestCase` with **only that metric's required fields**;
- runs `measure()` / `a_measure()` on a **fresh metric per call**;
- returns a small dataclass verdict: `score`, `passed`, `reason`, and the
  metric's actionable detail (for Faithfulness: unsupported claims from
  `verdicts` where `verdict == "no"`; for Answer Relevancy: the **irrelevant
  statements** — `metric.verdicts` are `AnswerRelevancyVerdict` per
  `metric.statements`; collect the ones judged not relevant).

**Remediation** is reused as-is; only the **correction prompt** changes:
- Faithfulness: "these claims aren't supported by the evidence; answer using only
  the evidence."
- Answer Relevancy: "your previous answer did not directly address the question;
  answer the user's question directly and concisely, without off-topic content."

(Whether a given metric should remediate or just block is a per-metric choice.
Relevancy remediation = "re-answer more on-topic" is sensible; keep `MODE`
block|remediate.)

---

## 7. Answer Relevancy — the next guardrail (specifics)

- **Measures:** does `actual_output` actually address `input` (the question),
  without irrelevant/off-topic content. Reference-free.
- **Fields:** `input`, `actual_output` only. **No `retrieval_context`.**
- **DeepEval:** `AnswerRelevancyMetric`. After `measure()`: `score`, `reason`,
  `statements` (extracted from the output), `verdicts`
  (`AnswerRelevancyVerdict` per statement — the actionable detail is the
  statements judged not relevant).
- **Contract decision to make first:** since it needs no evidence, does the
  guardrail run on *any* Q&A request, or should it still be scoped somehow?
  Options:
  - Run on any request that has a final user message + a response (simplest;
    relevancy applies to all Q&A). Enable/scope via **per-key** attachment.
  - Or require the marker anyway (to limit it to RAG traffic) — usually
    unnecessary for relevancy. Recommend the first, gated per key.
- **Remediation prompt:** re-answer addressing the question directly.
- **No-mixup requirements (apply to every new guardrail):**
  - class `AnswerRelevancyGuardrail(CustomGuardrail)` in its own hook module;
  - `guardrail_name: "answer-relevancy-guardrail"`;
  - own env-config namespace, e.g. `GUARDRAIL_ANSWER_RELEVANCY_THRESHOLD`,
    `GUARDRAIL_ANSWER_RELEVANCY_MODE`, `GUARDRAIL_ANSWER_RELEVANCY_FALLBACK_MESSAGE`
    (share generic ones like judge model / SSL / retries unless a metric needs
    its own);
  - registered as a **separate** entry in `config.yaml` `guardrails:`;
  - enabled **independently** per key (a team can have faithfulness, relevancy,
    both, or neither).

---

## 8. Methodology that worked (repeat it)

1. **Verify against source before coding.** We confirmed the LiteLLM hook
   signature, the `get_instance_fn` loader behaviour, and each DeepEval metric's
   required params + attributes from the actual repos — not from memory. Do the
   same for anything new.
2. **Build incrementally, one piece at a time, test each** before moving on:
   connectivity → judge → parser (unit tests) → evaluator (mock + live smoke) →
   hook + remediation (unit tests) → config/Docker → live end-to-end.
3. **Unit-test the deterministic parts** (parser, remediation, verdict
   extraction) with no network; smoke-test the judge live.
4. **Test live end-to-end** through the proxy, watching the guardrail logs.
5. **Keep the deliverable minimal** for infra; keep the full dev stack on the
   working branch.

---

## 9. Gotchas learned (carry forward)

- **Loader:** package must be next to `config.yaml` **and** its parent on
  `PYTHONPATH`, or you get `No module named '<pkg>'` / "attempted relative
  import" at startup.
- **pip in the image:** `ensurepip` doesn't create the `pip` executable — use
  `/app/.venv/bin/python -m pip …`.
- **Corporate SSL:** build-time pip needs the corporate CA or
  `--trusted-host pypi.org --trusted-host files.pythonhosted.org`; runtime
  outbound calls use `litellm.ssl_verify = False` (via `GUARDRAIL_SSL_VERIFY`).
- **Per-key guardrails = Enterprise** (403 on OSS). Needs a DB.
- **DeepEval custom judge:** a `DeepEvalBaseLLM` only has to return a string;
  DeepEval extracts JSON itself. Small/local judge models produce noisier
  verdicts and less reliable JSON — set `GUARDRAIL_JUDGE_JSON_MODE=false` if the
  model rejects `response_format`, and **validate any judge on a labelled set
  before trusting it to block**.
- **Metric semantics differ.** Faithfulness penalizes claims that *contradict*
  the context, **not** mere additions — so "extra correct info" still scores 1.0.
  Each new metric has its own semantics; learn them before writing test cases
  (e.g. Answer Relevancy penalizes off-topic/irrelevant statements).
- **Live models resist hallucinating on demand** — for demoing failure cases,
  construct inputs that force the behaviour (e.g. mixed correct/incorrect
  evidence), or use a forced threshold.
- **Metric objects are stateful** — one fresh metric per request; reuse only the
  judge.
- **Precision/Recall are not live guardrails** (need `expected_output`).

---

## 10. Concrete checklist to build a new live guardrail

- [ ] Verify the metric's `_required_params` + post-`measure()` attributes in the
      installed DeepEval.
- [ ] Confirm it's **reference-free** (no `expected_output`). If not, stop — it's
      offline-only.
- [ ] Decide whether it needs the evidence-marker contract (needs
      `retrieval_context`?).
- [ ] Parser: produce exactly the required fields (reuse existing extraction).
- [ ] Evaluator module: fresh metric per call, shared judge, own threshold,
      return `{score, passed, reason, <actionable detail>}`.
- [ ] Remediation: reuse loop, write a metric-appropriate correction prompt (or
      block-only).
- [ ] Hook subclass: own class name + `guardrail_name`, fail-open, same
      no-recursion retry routing (via router, `metadata` skip flag).
- [ ] Config: namespaced env vars; nothing hardcoded.
- [ ] Register separately in `config.yaml`; enable per key independently.
- [ ] Unit tests (parser/verdict/remediation) + live smoke through the proxy.
- [ ] Update the infra handoff (separate section/branch) for the new guardrail.

---

## 11. Pointers to the reference implementation

Everything above is realised in the faithfulness guardrail on branch
`claude/litellm-deepeval-guardrail-review-c7dzhd`:
- Pattern to copy: `guardrail/grounding.py` (evaluator) + `guardrail/hook.py`
  (orchestrator) + `guardrail/remediation.py` (retry loop).
- Reuse as-is: `guardrail/groq_judge.py` (judge), `guardrail/parser.py` (field
  extraction), `guardrail/config.py` (config pattern).
- Deployment + per-key enablement: `README.md` (working branch) and the
  `guardrail-infra-handoff` branch README.
- Test patterns: `tests/` (parser, grounding-verdict, remediation).
