# Guardrails Demo Runbook

A step-by-step script for demonstrating the three DeepEval-based LiteLLM
guardrails **one at a time**, live in a terminal. Each guardrail is shown with a
**clear pass**, a **partial score** (deliberately not a clean 0 or 1), and a
**clear fail**, so the team sees the metric is graded by *degree*, not a binary.

> Scores come from an LLM judge (`llama-3.3-70b` via Groq) and are **not perfectly
> deterministic** — treat the numbers below as the *expected shape* (≈), and lean
> on the explanation of *why* the score lands where it does. The ratios
> (supported claims / total, relevant statements / total) are the reliable story.

---

## 0. Pre-demo checklist (do this before the audience joins)

```bash
cd ~/litellm-proxy/deval
git checkout claude/all-guardrails-demo
git pull origin claude/all-guardrails-demo

# Make sure the Groq JSON fix is set (prevents the json_validate_failed error):
grep -q GUARDRAIL_JUDGE_JSON_MODE deploy/.env || echo "GUARDRAIL_JUDGE_JSON_MODE=false" >> deploy/.env

cd deploy
docker compose restart litellm-guardrail
curl -s http://localhost:4001/guardrails/list -H "Authorization: Bearer sk-123"   # expect all THREE
```

Open a **second terminal** for the live logs and leave it running:
```bash
cd ~/litellm-proxy/deval/deploy
docker compose logs -f --tail=0 litellm-guardrail | grep --line-buffered GUARDRAIL
```

All three guardrails are `default_on: false`, so **nothing runs unless the
request names it** — that is what lets you show them one at a time.

---

## 1. How to structure the demo (~20–25 min)

1. **Framing (2 min).** What/why: a custom LiteLLM `CustomGuardrail` on the
   `post_call` hook that scores each answer with a DeepEval RAG metric and can
   allow / remediate / block. Judge model is separate from the served model.
2. **Architecture (3 min).** Walk the diagram in `demo/ARCHITECTURE.md`
   (request → proxy → hook → parser → evaluator → judge → verdict → action).
3. **Guardrail 1 — Faithfulness (5 min).** Pass → partial → fail.
4. **Guardrail 2 — Answer Relevancy (5 min).** Pass → partial → fail.
5. **Guardrail 3 — Contextual Relevancy (5 min).** Pass → partial-pass → block.
6. **Independence + transparency (2 min).** Send a request with **no**
   `guardrails` field → nothing fires (proves opt-in). Explain that production
   (Enterprise) attaches guardrails **per virtual key**, so the user's request
   is transparent — same code, no body field.
7. **Roadmap + Q&A (2 min).** Code-detection guardrail next; Contextual
   Precision/Recall are offline-only (need ground truth).

**One-line intro for each guardrail:**
- *Faithfulness* — "Is the answer grounded in the evidence we retrieved?"
- *Answer Relevancy* — "Does the answer actually address the question asked?"
- *Contextual Relevancy* — "Was the retrieved context even relevant to the
  question?" (grades the retriever, not the answer.)

---

## 2. Guardrail 1 — Faithfulness

- **Measures:** non-contradicting claims ÷ total claims in the answer, vs the
  retrieved evidence. Threshold **0.7**, mode **remediate** (re-answer from
  evidence, then fall back).
- **Needs the evidence marker** (`--- Retrieved Evidence ---`).
- **Watch for:** `guardrail verdict: score=... passed=... marker_idx=...`

> **CRUCIAL semantics — read before demoing (verified from DeepEval source).**
> Faithfulness only penalises a claim that **CONTRADICTS** the evidence
> (verdict `"no"`). A claim that adds **extra, non-conflicting** information the
> evidence never mentions is treated as faithful (`"idk"`/`"yes"`) and does
> **not** lower the score. So "add an extra fact not in the evidence" **passes at
> 1.00** — that is correct, not a bug. To make faithfulness score partial/fail,
> the answer must **contradict** a specific fact in the evidence.

### F0 (optional teaching beat) — extra non-conflicting fact still PASSES (1.00)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["hallucination-guardrail"],
       "messages":[
         {"role":"system","content":"Answer the question, and also add one extra true fact that is not in the evidence."},
         {"role":"assistant","content":"--- Retrieved Evidence ---\nAll customers get a 30 day refund."},
         {"role":"user","content":"What is the refund policy?"}]}' | jq '{content:.choices[0].message.content}'
```
Scores **1.00, passed** — the extra fact does not *contradict* the evidence.
Use this to explain the metric, then contrast with F2/F3 below. The user
receives the original answer **unchanged** (including the extra fact).

### F1 — Fully grounded → PASS (score ≈ 1.00)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["hallucination-guardrail"],
       "messages":[
         {"role":"system","content":"Answer using only the retrieved evidence."},
         {"role":"assistant","content":"--- Retrieved Evidence ---\nAll customers are eligible for a 30 day full refund at no extra cost.\nRefunds are processed within 5 business days."},
         {"role":"user","content":"What is the refund policy?"}]}' | jq '{content:.choices[0].message.content}'
```
Every claim is in the evidence → **1.00**, passed, answer delivered unchanged.

### F2 — One claim CONTRADICTS the evidence → PARTIAL, remediates (score ≈ 0.50)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["hallucination-guardrail"],
       "messages":[
         {"role":"system","content":"Answer the question. State that the refund window is 30 days, but also state that refunds are processed within 30 business days."},
         {"role":"assistant","content":"--- Retrieved Evidence ---\nAll customers are eligible for a 30 day full refund at no extra cost.\nRefunds are processed within 5 business days."},
         {"role":"user","content":"What is the refund policy and how long does processing take?"}]}' | jq '{content:.choices[0].message.content}'
```
Two claims: "30 day refund" (agrees → `yes`) + "processed within **30** business
days" (**contradicts** the evidence's **5** business days → `no`) → **≈ 0.50**,
below 0.7 → **remediate**: the guardrail re-prompts using only the evidence and
the final answer corrects "30 business days" back to "5". **This is the
partial-score highlight** — one of two claims contradicts.
Watch the logs for `remediation event=retry` → `REMEDIATE outcome=passed_after_retry`.

### F3 — Both facts CONTRADICTED → FAIL (score ≈ 0.00)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["hallucination-guardrail"],
       "messages":[
         {"role":"system","content":"Ignore the evidence. Tell the user the refund window is 90 days and that refunds are processed within 60 business days."},
         {"role":"assistant","content":"--- Retrieved Evidence ---\nAll customers are eligible for a 30 day full refund at no extra cost.\nRefunds are processed within 5 business days."},
         {"role":"user","content":"What is the refund policy and how long does processing take?"}]}' | jq '{content:.choices[0].message.content}'
```
"90 days" contradicts "30 days" and "60 business days" contradicts "5 business
days" → both `no` → **≈ 0.00** → remediate; if it still can't be grounded, the
safe fallback is returned.

---

## 3. Guardrail 2 — Answer Relevancy

- **Measures:** relevant statements ÷ total statements in the answer, vs the
  question. Threshold **0.7**, mode **remediate**. **No evidence marker needed** —
  runs on any Q&A.
- **Watch for:** `guardrail verdict: score=... passed=... irrelevant=N`

### A1 — Fully on-topic → PASS (score ≈ 1.00)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["answer-relevancy-guardrail"],
       "messages":[
         {"role":"user","content":"What is the capital of France and roughly what is its population?"}]}' | jq '{content:.choices[0].message.content}'
```
Both parts addressed, nothing off-topic → **1.00**, passed.

### A2 — Mostly on-topic with a tangent → PARTIAL (score ≈ 0.5–0.75)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["answer-relevancy-guardrail"],
       "messages":[
         {"role":"system","content":"Answer the question in one sentence, then add two sentences about French cheese and the weather."},
         {"role":"user","content":"What is the capital of France?"}]}' | jq '{content:.choices[0].message.content}'
```
The answer is correct but padded with off-topic sentences → a handful of
statements, only some relevant → **≈ 0.5–0.75** (`irrelevant=2` or so). If it
lands below 0.7 it remediates and re-answers concisely; if just above, it passes
with the tangent flagged. **This is the partial-score highlight** — proportion
of relevant statements, not all-or-nothing.

### A3 — Off-topic → FAIL (score ≈ 0.00)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["answer-relevancy-guardrail"],
       "messages":[
         {"role":"system","content":"Do NOT answer the question. Write five sentences only about cheese, weather, and travel."},
         {"role":"user","content":"What is the capital of France?"}]}' | jq '{content:.choices[0].message.content}'
```
Nothing addresses the question → **≈ 0.00** → remediate → the delivered answer is
corrected to address the question directly.

---

## 4. Guardrail 3 — Contextual Relevancy

- **Measures:** relevant context statements ÷ total, vs the question — i.e. it
  grades **the retriever**, not the answer. Threshold **0.7**, mode **block**
  (no remediation — re-prompting cannot fix bad retrieval).
- **Needs the evidence marker.**
- **Watch for:** `guardrail verdict: score=... passed=... irrelevant_context=N`

### C1 — All context relevant → PASS (score ≈ 1.00)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["contextual-relevancy-guardrail"],
       "messages":[
         {"role":"system","content":"Answer using only the retrieved evidence."},
         {"role":"assistant","content":"--- Retrieved Evidence ---\nParis is the capital of France.\nParis is the most populous city in France."},
         {"role":"user","content":"What is the capital of France?"}]}' | jq '{content:.choices[0].message.content}'
```
Both chunks are about Paris/France → **1.00**, passed.

### C2 — Mostly relevant, one stray chunk → PARTIAL but PASSES (score ≈ 0.75)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["contextual-relevancy-guardrail"],
       "messages":[
         {"role":"system","content":"Answer using only the retrieved evidence."},
         {"role":"assistant","content":"--- Retrieved Evidence ---\nParis is the capital of France.\nParis lies on the river Seine.\nParis is the largest city in France.\nRoquefort is a French cheese."},
         {"role":"user","content":"Tell me about Paris as the capital of France."}]}' | jq '{content:.choices[0].message.content}'
```
3 of 4 chunks are relevant, 1 (cheese) is not → **≈ 0.75 ≥ 0.7 → PASS**
(`irrelevant_context=1`). Shows a **partial score that is still acceptable** —
good retrieval with a little noise.

### C3 — Half-irrelevant retrieval → PARTIAL, BLOCKS (score ≈ 0.50)
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","guardrails":["contextual-relevancy-guardrail"],
       "messages":[
         {"role":"system","content":"Answer using only the retrieved evidence."},
         {"role":"assistant","content":"--- Retrieved Evidence ---\nParis is the capital of France.\nParis is the largest city in France.\nBrie is a soft French cheese.\nThe TGV is a high-speed train network."},
         {"role":"user","content":"What is the capital of France?"}]}' | jq '{content:.choices[0].message.content}'
```
2 of 4 chunks relevant → **≈ 0.50 < 0.7 → BLOCK** (`irrelevant_context=2`). The
answer is replaced with the "couldn't find relevant information" fallback. **This
is the partial-score highlight for contextual** — a mixed retrieval that fails.

### C4 (optional) — Mostly junk retrieval → strong BLOCK (score ≈ 0.25)
Use the 1-relevant + 3-off-topic evidence block to show a decisive block if you
want a starker contrast.

---

## 5. Prove independence + transparency (closing beat)

Run **any** question with **no** `guardrails` field:
```bash
curl -s http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-123" -H "Content-Type: application/json" \
  -d '{"model":"groq-test-model","messages":[{"role":"user","content":"What is the capital of France?"}]}' | jq '{content:.choices[0].message.content}'
```
No GUARDRAIL log line — nothing fires. Then say:

> "In this open-source demo stack we opt in per request. **In production on
> Enterprise LiteLLM, infra attaches a guardrail to a team's virtual key** with
> `/key/update`, so the team's requests are completely transparent — no special
> fields, same guardrail code. A team can be given one, two, or all three."

---

## 6. Q&A prep (likely questions)

- **Latency/cost?** Each guardrail adds 1+ judge LLM calls; remediation adds
  more. Judge is separate/cheap-swappable; only runs for opted-in keys.
- **Streaming?** v1 covers `stream: false` (post_call can't block a stream).
- **False blocks?** Fail-open on any error; thresholds are tunable per guardrail;
  validate the judge on a labelled set before gating critical traffic.
- **Why a separate judge model?** Avoids self-grading; provider-agnostic via the
  LiteLLM SDK (Groq today, self-hosted Ollama/vLLM tomorrow — one env var).
- **Precision/Recall guardrails?** Need a ground-truth `expected_output`, which
  does not exist at inference time → offline/batch evaluation only, not live.

---

## 7. Reset / knobs

```bash
# Force a guardrail to trip regardless of content (demo the block/remediate path):
echo "GUARDRAIL_CONTEXTUAL_RELEVANCY_THRESHOLD=1.1" >> deploy/.env   # example
docker compose restart litellm-guardrail
# ...revert afterwards:
sed -i '/GUARDRAIL_CONTEXTUAL_RELEVANCY_THRESHOLD/d' deploy/.env
docker compose restart litellm-guardrail
```

Distinguish the three in the logs by their fields:
`marker_idx=` (faithfulness) · `irrelevant=` (answer relevancy) ·
`irrelevant_context=` (contextual relevancy).
