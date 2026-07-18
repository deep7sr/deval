# Experiment: LiteLLM ↔ Langfuse Integration Test — Results

**Date:** 2026-07-18 · **Versions:** litellm `1.92.0` (proxy mode), langfuse SDK `2.60.10`

## What was tested and how

A real LiteLLM proxy was run with `success_callback: ["langfuse"]` /
`failure_callback: ["langfuse"]` (see `litellm_test_config.yaml`). Because the
test environment's network policy blocks all container-registry CDNs (Docker
Hub, ghcr.io, quay.io, ECR public — all 403 through the egress proxy), the full
Langfuse server stack could not be deployed *in this sandbox*. Instead, the
integration seam was tested against `langfuse_stub.py` — a stub implementing
Langfuse's real ingestion API (`POST /api/public/ingestion`) that records every
event the proxy emits. The LiteLLM side (the part we're actually deciding on)
is fully real: real proxy, real Langfuse SDK, real callback code path. Upstream
model calls used `mock_response` (no provider keys in the sandbox), which does
not affect the callback pipeline.

Traffic sent: 3 plain requests with zero metadata (simulating a team that
knows nothing about the eval system), 1 request with opt-in metadata
(tags/session/user), 1 failing request (unknown model), then a 40-request
latency benchmark against callback-on and callback-off proxies, then a
resilience test with the Langfuse endpoint killed.

Raw captures: `captured_events.jsonl`.

## Findings

### 1. Contract-free capture works exactly as designed ✅
Every request produced a `trace-create` + `generation-create` event pair with
**no cooperation from the client**: full input messages, full output, model
name, token usage, and — notably — **cost already computed by LiteLLM**
(`usage.totalCost`, e.g. `$0.000225` for the gpt-4o call) plus precise
start/end timestamps. This is the field set Langfuse's LLM-as-a-judge
evaluators need (`{{input}}`/`{{output}}` mapping), confirmed present for
100% of requests.

### 2. Opt-in enrichment flows through ✅
The request carrying `metadata: {tags, session_id, trace_user_id,
generation_name}` produced a trace with `tags=['team-platform','rag',...]`,
`sessionId=sess-9`, `userId=user-42`. LiteLLM also auto-tags traces with
caller `User-Agent`. Plain requests are still fully traced — enrichment is
additive, never required. (In production, virtual keys per team give
attribution even without any metadata.)

### 3. Failures are traced too ✅
The invalid-model request produced a generation with `level=ERROR` and the
full 400 error as `statusMessage`, `cost=0`. Error-rate dashboards come for
free via `failure_callback`.

### 4. Latency overhead is small and asynchronous ✅
40 sequential requests per proxy (mocked model ⇒ worst case: overhead is a
huge *fraction* here because model time is ~0; in production it is amortized
against seconds of real model latency):

| | p50 | p95 | mean |
|---|---|---|---|
| callback ON | 6.2 ms | 12.9 ms | 6.8 ms |
| callback OFF | 4.8 ms | 6.0 ms | 4.9 ms |

≈ **1–2 ms typical added overhead** (callback scheduling only — the HTTP
export to Langfuse happens out of band, batched by the SDK). Negligible
against real LLM calls that take hundreds to thousands of ms.

### 5. Fire-and-forget resilience confirmed ✅
With the Langfuse endpoint killed entirely, **10/10 requests still returned
HTTP 200** with normal responses. An eval-system outage cannot take down
gateway traffic. (Corollary: events emitted during an outage are dropped, not
queued — acceptable for observability/eval data.)

## Not testable in this sandbox (test on the laptop Docker setup)

- The Langfuse server UI/dashboards and ClickHouse aggregation themselves.
- LLM-as-a-judge scoring quality (needs a real judge model — no Groq key
  here). The judge's *input contract* (trace input/output fields) is verified
  present; judge quality validation against a small labeled set is the
  recommended next step, per the reliability notes in
  `../../litellm-deepeval-guardrail-context.md`.

## Verdict

**Langfuse is a good decision for gateway-level evals on LiteLLM.** The
integration is first-class (two config lines), captures everything needed for
LLM-as-a-judge without any developer contract, computes cost centrally, adds
~1–2 ms overhead, and fails safe. No blocker was found on the LiteLLM side.
Remaining risk is operational (running the 6-container Langfuse stack) and
judge quality — both to be validated in the laptop/staging deployment per
`../../docs/langfuse-litellm-setup.md`.
