# Session Handoff — DeepEval LiteLLM Guardrails

**Purpose.** Continuity doc so a fresh chat session can resume exactly where the
previous one left off. Read this **together with `GUARDRAILS_BLUEPRINT.md`**
(the design bible for building each guardrail). This file records current state,
branches, what was learned, the user's environment, and the open work + pending
decisions.

> How to resume: check out `claude/all-guardrails-demo` (the most complete
> branch), read `GUARDRAILS_BLUEPRINT.md` then this file. Continue in the same
> working style noted under "Working norms" below.

---

## 1. What this project is

Custom **LiteLLM `CustomGuardrail`** guardrails that score each chat response
with a **DeepEval RAG metric** (pinned `deepeval==4.1.0`) and then **allow /
remediate / block**. A separate **judge model** (via the LiteLLM SDK, so it's
provider-agnostic) does the scoring. Three guardrails are built and validated
live; a 4th (code detection) is designed but not built.

Package layout (`guardrail/`): `parser.py` (marker contract), `groq_judge.py`
(`LiteLLMJudge`), `remediation.py` (retry loop), `config.py` (env-namespaced
settings), plus one evaluator + one hook per guardrail.

---

## 2. The three guardrails (all built, live, reference-free)

| Guardrail | Evaluator / Hook | Metric | Grades | Needs marker? | Default mode | Actionable detail | Log field |
|---|---|---|---|---|---|---|---|
| **Faithfulness** | `grounding.py` / `hook.py` → `HallucinationGuardrail` (`hallucination-guardrail`) | `FaithfulnessMetric` | answer vs evidence | yes | remediate | unsupported/contradicting claims | `marker_idx=` |
| **Answer Relevancy** | `relevancy.py` / `relevancy_hook.py` → `AnswerRelevancyGuardrail` (`answer-relevancy-guardrail`) | `AnswerRelevancyMetric` | answer vs question | **no** (any Q&A) | remediate | irrelevant statements | `irrelevant=` |
| **Contextual Relevancy** | `contextual_relevancy.py` / `contextual_relevancy_hook.py` → `ContextualRelevancyGuardrail` (`contextual-relevancy-guardrail`) | `ContextualRelevancyMetric` | **retriever** vs question | yes | **block** (no retry) | irrelevant context | `irrelevant_context=` |
| **Turn Faithfulness** (4th, built on `claude/deepeval-turn-faithfulness-fkw26a`) | `turn_faithfulness.py` / `turn_faithfulness_hook.py` → `TurnFaithfulnessGuardrail` (`turn-faithfulness-guardrail`) | `TurnFaithfulnessMetric` (conversational) | every answer in the conversation vs its exchange's evidence | yes (≥1 marker anywhere) | remediate (final answer only) | unfaithful claims (deduped across windows) | `windows=` `unfaithful=` |

- Fields per metric: Faithfulness = input+actual_output+retrieval_context;
  Answer Relevancy = input+actual_output; Contextual Relevancy =
  input+retrieval_context (actual_output unused).
- Evidence marker contract: an `assistant` message whose content **starts with**
  `--- Retrieved Evidence ---`, one evidence line per line; question = final
  `user` message. No marker → guardrail silently skips.
- Hooks are fail-open, clear stale `reasoning`/`reasoning_content` on overwrite,
  and route remediation retries via the proxy Router (no recursion).
- **Precision / Recall are NOT possible as live guardrails** (need ground-truth
  `expected_output`) → offline only.

---

## 3. Branch map (IMPORTANT — one guardrail per branch, "no mixups")

The user is strict about this: each guardrail on its **own** branch off the
faithfulness base; combined only for demo/handoff.

| Branch | Tip | Contents |
|---|---|---|
| `claude/litellm-deepeval-guardrail-review-c7dzhd` | `7efe2d3` | Faithfulness base + `GUARDRAILS_BLUEPRINT.md` (the shared base everything is built on) |
| `claude/session-1v5577` | `1107b2a` | base + **Answer Relevancy** + reasoning-clear fix |
| `claude/contextual-relevancy-guardrail` | `e31b0b6` | base + **Contextual Relevancy** only |
| `claude/all-guardrails-demo` | `b6b24bb` | **ALL THREE** + `default_on:false` (per-request opt-in) + `demo/DEMO_GUIDE.md` + `demo/ARCHITECTURE.md` + demo-case fixes. **This is the demo branch.** |
| `claude/deepeval-turn-faithfulness-fkw26a` | (this branch) | demo branch + **Turn Faithfulness** (4th guardrail, conversational): `parse_conversation` in `parser.py`, `turn_faithfulness.py`, `turn_faithfulness_hook.py`, config namespace `GUARDRAIL_TURN_FAITHFULNESS_*`, registered in `deploy/config.yaml`, smoke script. **Gotcha found & fixed:** deepeval 4.1.0's `TurnFaithfulnessMetric` (a) resolves prompt templates by class name — a subclass must pass `template_class` — and (b) returns verdicts as raw dicts with a custom string-returning judge, crashing its own scoring; the recording subclass coerces them to schema objects. |

Naming convention going forward: `claude/<guardrail-name>-guardrail`. The
answer-relevancy branch keeps its `session-1v5577` name (already approved).
57 tests pass on the demo branch; 48 on session-1v5577; 37 on contextual.

---

## 4. Key learnings this session (carry forward — these bit us live)

1. **Faithfulness penalizes CONTRADICTIONS, not additions.** A claim only scores
   `no` if it *contradicts* the evidence; extra true-but-unlisted info scores
   `yes`/`idk` → still faithful → 1.0. Verified from DeepEval source. So "add an
   extra fact" **passes**. To fail/partial faithfulness, the answer must
   *contradict* an evidence fact.
2. **Contextual Relevancy is strict about question-specificity.** A narrow
   question ("What is the capital of France?") makes the judge rule that "Paris
   is the largest city" is *not relevant* (wrong attribute) → most things block.
   For a clean PASS demo, use a **broad** question ("Tell me about the city of
   Paris") and vary only the evidence.
3. **Live models resist hallucinating on demand** (gpt-oss-120b is safety-tuned;
   it inconsistently obeys "state false info"). Don't rely on coaxing the model.
   For a **reliable** faithfulness failure: give **deliberately-wrong evidence +
   a strong-prior question** (e.g. evidence "Eiffel Tower is in Rome", ask where
   it is → model says Paris → contradicts evidence → low score). For a **reliable
   PARTIAL** (~0.5): evidence with one correct + one blatantly-wrong fact (e.g.
   "in Paris" [grounded] + "made of wood" [model says iron → contradicts]).
4. **Groq strict JSON mode can 400** (`json_validate_failed`) when the judge
   emits slightly malformed JSON (unescaped quotes). Fix: set
   `GUARDRAIL_JUDGE_JSON_MODE=false` (DeepEval then parses JSON itself, more
   leniently). This is set in the VM's `deploy/.env`.
5. **The `reasoning` field** in responses is the *served model's* chain-of-thought
   (gpt-oss emits it), **NOT** the judge. Hooks null it out when they overwrite
   `content` so a stale trace can't contradict the delivered answer.
6. **Markdown/`\n` in responses** are the model's formatting + JSON encoding, not
   junk — a real chat UI renders them. For clean terminal demo output use
   `jq -r '.choices[0].message.content'`.

---

## 5. The user's VM / how to test (their real environment)

- Company VM, LiteLLM proxy run via **docker compose** at
  `~/litellm-proxy/deval/deploy/`. Service **`litellm-guardrail`**, host port
  **4001**, master key **`sk-123`**.
- Model `groq-test-model` = `groq/openai/gpt-oss-120b`; judge =
  `groq/llama-3.3-70b-versatile` (both Groq, `GROQ_API_KEY` in `deploy/.env`,
  which also has `GUARDRAIL_JUDGE_JSON_MODE=false`).
- `guardrail/` is volume-mounted + `PYTHONPATH=/app`; `config.yaml` is mounted.
  So **`docker compose restart litellm-guardrail`** picks up code/config changes;
  no rebuild unless deps change.
- Watch decisions: `docker compose logs -f --tail=0 litellm-guardrail | grep --line-buffered GUARDRAIL`.
- Guardrails are `default_on:false` on the demo branch → invoke per request with
  `"guardrails":["<name>"]` in the body (demo convenience).
- NOTE: `~/litellm-proxy/config.yaml` is a **separate experiment proxy** — ignore
  it for guardrail work.

---

## 6. Demo vs production enablement (settled)

- **Demo (OSS stack):** per-request `"guardrails":[...]` in the body; lets the
  user invoke one guardrail at a time.
- **Production (Enterprise LiteLLM):** infra attaches a guardrail to a **virtual
  key** (`/key/update` with `"guardrails":["<name>"]`). The user's request is then
  **transparent** — no special fields. Same guardrail code; `default_on:false` is
  the correct prod posture. Per-key attachment is an **Enterprise** feature
  (403 on OSS).

---

## 7. Open work + PENDING DECISIONS (what to do next)

### A. Guardrail actions: retry / block / warning (post-demo feedback — NEXT UP)
Infra asked to choose the **action** per key. Design agreed:
- Standardize a `guardrail_action` = **block | retry | warn** (contextual:
  **block | warn** only — retry can't fix retrieval), read from
  `litellm_params` (NOT the reserved `mode` key, which is the hook stage),
  falling back to the env default.
- Implement **warn** on faithfulness + answer-relevancy (contextual already has
  `observe`): run the check, log the verdict, **deliver the original answer**,
  optionally add an `x-guardrail-warning` header.
- Ship **named variants** so infra picks the action **from the UI by attaching a
  name** (LiteLLM per-key attaches by name; it does NOT natively pass custom
  per-key params). E.g. `faithfulness-block`, `faithfulness-retry`,
  `faithfulness-warn`, etc.
- Build on branch `claude/guardrail-actions` (part of production-handoff work).
- **PENDING CONFIRM from user before building:** (1) warn = log-only, or also add
  a response header/flag (recommended: add header)? (2) named-variant approach OK
  (vs expecting a single guardrail whose mode is toggled in UI — not natively
  supported)? Also: verify the exact LiteLLM version's UI + `pre_call`/guardrail
  param support from source before coding.

### B. 4th guardrail: Code detection (designed, not built)
- **Decision: build CUSTOM, not the LiteLLM built-in** (built-ins are
  pattern-based → false-positive on natural language; poor multi-language).
- LLM-judge **classifier** (reuse `LiteLLMJudge`, NOT DeepEval): returns
  `{contains_code, language, confidence, snippet}`. Optional cheap fenced-code
  fast-path.
- Runs on **BOTH input and output**: input via a **pre_call hook** (block before
  the LLM runs), output via post_call. Block-only with a clear message.
- Must NOT false-positive on natural language, incl. **non-English human
  languages** (strong reason for the judge over regex). Confidence threshold;
  validate on a labelled set before enforcing.
- Config namespace `GUARDRAIL_CODE_DETECTION_*`; own branch.
- **PENDING CONFIRM:** default `observe` vs `block`; scan input+output vs one
  side; confirm "any language" = don't false-positive on non-English prose.
- Verify LiteLLM `pre_call` hook signature + built-in catalog from source first.

### C. Production handoff (after infra go-ahead)
- Single combined package/branch: all guardrails, one `config.yaml` registering
  all (named variants from A), unified tests, and an **infra README** with the
  per-key `/key/update` enablement + the transparency contract. Blueprint §10.

### D. Not yet done (flagged)
- **Judge validation on a labelled set** before gating critical traffic
  (blueprint §9) — done ad-hoc only.
- Offline eval harness for Contextual Precision/Recall (offline-only metrics).

---

## 8. Working norms (how this collaboration runs)

- **One guardrail per branch**, off the faithfulness base; combine only for
  demo/handoff. Confirm branch name before pushing.
- **Confirm the plan/understanding before implementing** (the user reviews first).
- **Verify against DeepEval/LiteLLM source before coding** — never from memory
  (e.g. `_required_params`, post-`measure()` attrs, hook signatures).
- **Unit-test deterministic parts** (parser, verdict extraction, remediation) with
  no network; smoke-test the judge live through the proxy.
- Commit trailer used:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and a
  `Claude-Session:` line. Push with `-u origin <branch>`; force-with-lease only
  for cleanup of unmerged history.
- Model id is `claude-opus-4-8` — keep it out of commits/PRs/artifacts.
- Do NOT create PRs unless the user asks.

---

## 9. Quick pointers

- Demo runbook: `demo/DEMO_GUIDE.md` (pass/partial/fail cases per guardrail,
  talking points, Q&A prep). Diagrams: `demo/ARCHITECTURE.md` (Mermaid).
- Reference impl + gotchas: `GUARDRAILS_BLUEPRINT.md`.
- Live smoke scripts: `scripts/step_relevancy_smoke.py`,
  `scripts/step_contextual_relevancy_smoke.py`.
- Deploy stack: `deploy/` (Dockerfile, docker-compose.yml, config.yaml, .env.example).
