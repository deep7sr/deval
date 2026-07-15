# Hallucination Guardrail — LiteLLM install

A custom LiteLLM guardrail that blocks ungrounded (hallucinated) answers: it
scores each response against the retrieved evidence in the request and, if the
answer isn't grounded, retries the model then falls back to a safe message.

Install = 4 steps in your image/config + enable it per key. ~15 minutes.

## Prerequisites
- Your LiteLLM proxy runs from a Docker image you can rebuild.
- A judge model the proxy can reach (any LiteLLM model id + its API key).
- A database on the proxy (needed to enable guardrails per key — Enterprise).

## 1. Add DeepEval to your image
In your LiteLLM Dockerfile:
```dockerfile
USER root
RUN /app/.venv/bin/python -m pip install --no-cache-dir "deepeval==4.1.0"
```

## 2. Add the `guardrail/` package to your image
Copy this repo's `guardrail/` folder **next to your config.yaml**, and put that
directory on `PYTHONPATH`. If your config is at `/app/config.yaml`:
```dockerfile
COPY guardrail /app/guardrail
ENV PYTHONPATH=/app
```
Rule: the `guardrail/` folder must sit in the **same directory as config.yaml**,
and that directory must be on `PYTHONPATH`. (Otherwise you get
`No module named 'guardrail'` at startup.)

## 3. Add the guardrail to config.yaml
Append (nothing else changes):
```yaml
guardrails:
  - guardrail_name: "hallucination-guardrail"
    litellm_params:
      guardrail: guardrail.hook.HallucinationGuardrail
      mode: "post_call"
      default_on: false
```

## 4. Set environment variables (in the container)
```
GUARDRAIL_JUDGE_MODEL=groq/llama-3.3-70b-versatile   # your judge model
GROQ_API_KEY=...                                     # that model's key
```
Self-hosted judge (Ollama/vLLM), instead of the key:
```
GUARDRAIL_JUDGE_MODEL=ollama_chat/gemma3:4b
GUARDRAIL_JUDGE_API_BASE=http://<host>:11434
```
Other optional settings (all have defaults): `GUARDRAIL_FAITHFULNESS_THRESHOLD`
(0.7), `GUARDRAIL_MODE` (`remediate` | `block`), `GUARDRAIL_MAX_RETRIES` (3),
`GUARDRAIL_JUDGE_JSON_MODE` (set `false` if the judge rejects JSON output),
`GUARDRAIL_SSL_VERIFY` (false). Full list: top of `guardrail/config.py`.

## 5. Rebuild, restart, and enable it on a key
Redeploy the proxy. The guardrail is **off** until attached to a virtual key.

Admin UI: **Virtual Keys** → edit key → **Guardrails** → select
`hallucination-guardrail`. (Or set it on a **Team** so all its keys inherit it.)

Or via API:
```bash
curl -X POST 'http://<proxy>/key/generate' \
  -H 'Authorization: Bearer <MASTER_KEY>' -H 'Content-Type: application/json' \
  -d '{"guardrails": ["hallucination-guardrail"]}'
```

## 6. Verify
```bash
# registered?
curl -s http://<proxy>/guardrails/list -H 'Authorization: Bearer <MASTER_KEY>'
```
Then send a request **with a guardrail-attached key** and the evidence marker
(below); the proxy logs should show `GUARDRAIL ... verdict: score=... passed=...`.

---

## Request format (share with teams that use the guardrail)
The guardrail only runs when the request carries evidence like this — otherwise
it passes the request through untouched:
```json
"messages": [
  { "role": "system", "content": "Answer using only the retrieved evidence." },
  { "role": "assistant", "content": "--- Retrieved Evidence ---\nFact one.\nFact two." },
  { "role": "user", "content": "The question?" }
]
```
- Evidence goes in an `assistant` message whose content **starts exactly** with
  `--- Retrieved Evidence ---`, one fact per line.
- The question is the final `user` message. Both must be plain strings.
- A `system` message (or any other message) may be present anywhere — only the
  final `user` message and the nearest preceding `assistant` evidence message
  are used.

## Before you let it block real traffic
- **Validate the judge on your own labelled examples and tune the threshold** —
  `0.7` is a starting guess; a small judge model needs this most.
- Use a production-grade judge endpoint (free tiers throttle under load).
- Each check is 2+ judge calls (plus retries) — latency-test it.
- It **fails open** (judge unreachable → request passes unchecked) and only
  covers non-streaming (`stream: false`) responses.
- Roll out to one team's key first, watch, then expand.
