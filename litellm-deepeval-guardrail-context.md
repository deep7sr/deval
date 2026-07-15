# LiteLLM + DeepEval Hallucination Guardrail — Project Context

## Goal
Build a guardrail on a LiteLLM proxy that detects hallucinated (non-grounded) LLM responses and prevents them from reaching the user. Using DeepEval's `FaithfulnessMetric` as the scoring mechanism.

## Key constraint that shaped the design
Initially: "we cannot tell teams to follow any contract" — the guardrail had to work on raw, unstructured `messages` arrays with no cooperation from calling teams. This led to exploring an LLM-based **segregation step** (a judge LLM that splits a raw request into `context` vs `query`) as a contract-free fallback.

**This constraint was later relaxed.** Final decision: teams **must** follow a strict contract to get guardrail coverage. Non-compliant requests simply skip the check (no fallback, no segregation LLM, no inference).

---

## Final Contract (locked in)

1. Retrieved evidence goes in a message with `"role": "assistant"`.
2. That message's `content` must **start with the exact marker string**: `--- Retrieved Evidence ---` (this exact spelling and spacing — not the typo'd "Retreived" seen in an early reference screenshot).
3. Everything after the marker in that message is the evidence, one fact/chunk per line.
4. The actual question is the final `"role": "user"` message, as normal.
5. `content` must be a **plain string**, not a content-parts array, for both the evidence message and the user message.
6. **Multi-turn rule**: if multiple marker-tagged messages exist in a conversation (re-retrieval across turns), only the one **closest to (immediately preceding) the final user message** is used. Older ones are ignored.
7. **No marker found → guardrail does not run** for that request. Silent skip, no block, no fallback.

### Example — single-turn
```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are a factual assistant. Answer using only the retrieved evidence." },
    { "role": "assistant", "content": "--- Retrieved Evidence ---\nAll customers are eligible for a 30 day full refund at no extra cost.\nRefunds are processed within 5 business days." },
    { "role": "user", "content": "What is the refund policy?" }
  ]
}
```

### Example — multi-turn (re-retrieval)
```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "user", "content": "What's your return policy?" },
    { "role": "assistant", "content": "--- Retrieved Evidence ---\nAll customers are eligible for a 30 day full refund at no extra cost." },
    { "role": "assistant", "content": "You get a full refund within 30 days, no extra cost." },
    { "role": "user", "content": "What about the warranty on premium plans?" },
    { "role": "assistant", "content": "--- Retrieved Evidence ---\nPremium plan customers get a 2-year extended warranty covering parts and labor." },
    { "role": "user", "content": "Does that cover accidental damage too?" }
  ]
}
```
Only the second (warranty) evidence block is used for the current turn — the refund evidence is stale and ignored, per rule 6.

### Known residual risks (documented, not solved)
- **Multi-turn token trimming risk**: if a team's app strips old evidence messages from history to save tokens, later turns lose marker coverage entirely and silently fall under rule 7 (skipped, not blocked). Teams need to know to re-send/retain the evidence message for every turn they want covered.
- **Staleness/relevance risk**: "closest preceding marker" is a proxy for "correct context," not a guarantee. If an app doesn't re-retrieve for a genuinely new sub-topic, the evidence checked may not be truly relevant to the current question — a source of possible false flags, not a parsing bug.
- **Trust/injection risk**: pinning the exact marker string makes matching reliable but doesn't address trust — anything upstream capable of writing into an `assistant`-role message (a prior tool result, or an app bug) could produce the marker string and have arbitrary content treated as trusted evidence. Accepted as a known limitation for now, not solved by the contract.

---

## DeepEval Metric Decision

**Use `FaithfulnessMetric` (single-turn `LLMTestCase`), not `TurnFaithfulnessMetric`.**

Reasons:
- LiteLLM's `post_call` guardrail hook fires per request/response pair — a block/allow decision is inherently single-turn, regardless of how much conversation history is in the request.
- `TurnFaithfulnessMetric` averages faithfulness across a conversation — averaging is the wrong semantics for a hard gate (one bad turn could be diluted by earlier good turns and slip through).
- Turn Faithfulness requires per-turn `retrieval_context` attribution across the whole conversation (heavier `ConversationalTestCase`/`Turn` construction) — unnecessary complexity given the marker contract only needs the *current* turn's evidence.
- Turn-level checking is strictly more expensive (claim-extraction-and-verification per turn, not once).
- Turn Faithfulness may still be useful later as an **offline/retrospective quality metric** (e.g. batch-scored over logged conversations, possibly via MLflow tracing) — but that's a separate system from the live guardrail.

### `LLMTestCase` field mapping (what DeepEval actually needs)
DeepEval does **not** auto-extract context from a raw request. It only understands three fields on a constructed test case object:

| DeepEval field | Source |
|---|---|
| `input` | content of the last `role: user` message |
| `actual_output` | the LLM's actual completion response (not part of the request) |
| `retrieval_context` (list of strings) | lines after the marker in the nearest preceding marker-tagged `assistant` message |

`retrieval_context` is the field that drives Faithfulness's scoring (claim extraction from `actual_output`, then claim verification against `retrieval_context`). `input` is required by the schema but not load-bearing for Faithfulness's own scoring logic.

**Manager alignment note**: this was clarified explicitly with the manager — DeepEval cannot work directly off a raw message request. A separate structuring/parsing step (the contract-based extractor above) must run *before* DeepEval is ever called; DeepEval's world begins only once `input`/`actual_output`/`retrieval_context` already exist as clean fields.

---

## Architecture (decided)

Two decoupled components, orchestrated by a thin LiteLLM guardrail hook:

1. **Parser/adapter module** — owns 100% of the contract logic in isolation (find marker message, apply "most recent wins" rule, handle content-as-string requirement, extract user query). Knows nothing about DeepEval or LiteLLM. Returns either `{input, retrieval_context, actual_output}` or an explicit "non-compliant, skip" signal.
   - Purely deterministic string/substring matching — no LLM call, minimal latency cost.
   - Independently unit-testable against synthetic request shapes (this is where most of the correctness risk lives, given how many edge cases were discussed).
2. **DeepEval invocation wrapper** — takes only the parser's clean output, builds `LLMTestCase`, runs `FaithfulnessMetric`, returns score/pass-fail/reason. Knows nothing about message arrays or the marker contract.
3. **LiteLLM `post_call` custom code guardrail hook** — orchestrates: call parser → if non-compliant, allow and skip → if compliant, call DeepEval wrapper → block or allow based on threshold.

### Efficiency notes
- Instantiate `FaithfulnessMetric` once, reuse across requests (avoid per-request judge-model client setup overhead).
- Use DeepEval's async measure path if available, so the guardrail doesn't block the event loop.
- Log parser decisions (marker found? which message matched? why skipped?) separately from final DeepEval verdicts — needed for teams to self-diagnose "why didn't my guardrail run."
- Keep the marker string and staleness rule as named constants/config, not inline literals.

---

## Other tools considered (context, not adopted as primary path)

- **PromptGuard / vendor guardrails**: built-in hallucination detection exists but is black-box — can't see team-specific retrieved context unless explicitly passed. Not chosen as the primary mechanism.
- **MLflow (`mlflow.genai`)**: has native judges (e.g. `RetrievalGroundedness`) and also wraps DeepEval, RAGAS, and Phoenix as scorers under one API (`mlflow.genai.scorers.deepeval.*` etc.), with built-in tracing/UI. Does **not** remove the context-extraction requirement — same underlying need for a populated context field. Could be layered on top later for observability/logging and for offline comparison across judge frameworks (useful for validating parser + judge reliability against a labeled set), but not required for the core guardrail.
- **GEval, AnswerRelevancyMetric**: considered as context-free alternatives but rejected as not actually answering the "is this grounded in fact" question (Answer Relevancy checks topical relevance, not truthfulness).

## Reliability recommendations for the DeepEval judge itself (if/when tuning accuracy)
- Use a strong model for the judge, not just a cheap one — errors here are silent and propagate.
- Validate with a held-out labeled dataset before trusting it to block real traffic (DeepEval's own `deepeval test run` / offline eval tooling is suited for this).
- Log all intermediate outputs (extracted evidence, extracted claims, final score/reason), not just the final block/allow decision.
- Consider self-consistency/ensembling on the faithfulness verdict specifically if false blocks are costly.

---

## Current Status / Next Steps

**Environment**: LiteLLM hosted locally via Docker on the user's laptop.

**Plan**: 
1. Build end-to-end test setup (parser module + DeepEval wrapper + LiteLLM `config.yaml` custom code guardrail registration).
2. Validate with targeted test cases:
   - Faithful response, marker present → should pass.
   - Unfaithful/hallucinated response, marker present → should block (may need a mocked/hardcoded completion response to reliably trigger, since live models resist hallucinating on demand).
   - No marker present → should skip check entirely, pass through.
   - Multi-turn with stale evidence → confirms "nearest preceding marker" rule picks correct block.
   - Malformed marker (typo/spacing) → confirms correctly treated as "no marker."
3. Once validated locally, prepare a demo for the team.

**Open items flagged before implementation continues**:
- Which LLM provider/API key will be used for the DeepEval judge model (separate from whatever model the actual chat completion uses)?
- Confirm LiteLLM Docker setup: stock image vs. custom Dockerfile (DeepEval package needs to be installed in the image; custom code guardrail files need to be volume-mounted if not inline in `config.yaml`).
- Decide mocking vs. live-model approach for reliably producing an "unfaithful" test case.
- Confirm environment variables (judge model API key) are passed into the Docker container explicitly, not just present on the host shell.

**Not yet built**: no code has been generated yet in this conversation — all discussion so far has been architecture/design only, per explicit request to discuss approach before generating code. Code generation (parser module, DeepEval wrapper, `config.yaml`) is the next phase.
