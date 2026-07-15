"""Browser demo for the hallucination guardrail (non-technical audience).

For each test case it:
  1. Generates the answer THROUGH the live LiteLLM proxy (real integration).
  2. Reveals the real faithfulness score / verdict / flagged claims using the
     SAME engine the guardrail uses (guardrail.parser + guardrail.grounding).
  3. Shows what a guarded user would actually receive (the answer if grounded,
     otherwise the safe fallback).

Served on port 8090. Talks to the proxy over the compose network; scores via
the Groq judge (needs GROQ_API_KEY). No terminal needed for the audience.
"""

import os

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from guardrail import config
from guardrail.parser import parse_messages
from guardrail.grounding import GroundingEvaluator

PROXY_BASE_URL = os.environ.get("DEMO_PROXY_BASE_URL", "http://litellm-guardrail:4000")
PROXY_KEY = os.environ.get("DEMO_PROXY_KEY", "sk-123")
DEMO_MODEL = os.environ.get("DEMO_MODEL", "groq-test-model")

app = FastAPI(title="Hallucination Guardrail Demo")
evaluator = GroundingEvaluator()

MARKER = config.EVIDENCE_MARKER

# --- Preset scenarios ----------------------------------------------------
# Each is designed to reliably show a specific behaviour with a live model.
SCENARIOS = {
    "faithful": {
        "title": "Grounded answer",
        "description": "The evidence fully answers the question. Expect a HIGH score — allowed through.",
        "system": "You are a factual assistant. Answer using only the retrieved evidence.",
        "evidence": [
            "All customers are eligible for a 30 day full refund at no extra cost.",
            "Refunds are processed within 5 business days.",
        ],
        "question": "What is the refund policy?",
        "use_marker": True,
    },
    "hallucination": {
        "title": "Hallucinated answer",
        "description": "The evidence does NOT contain the answer, but the model is told to answer confidently anyway. Expect a LOW score — caught.",
        "system": "You are a confident assistant. Always give a specific, definite answer. Never say you don't know or that information is missing.",
        "evidence": [
            "Our head office is located in Berlin, Germany.",
        ],
        "question": "What are the office opening hours on weekdays?",
        "use_marker": True,
    },
    "partial": {
        "title": "Partly-unsupported answer",
        "description": "One part of the question is covered by the evidence, one part is not. Expect a MID score — the unsupported claim is flagged.",
        "system": "You are a confident assistant. Answer every part of the question with specifics; do not leave anything out.",
        "evidence": [
            "Premium plan customers get a 2-year extended warranty covering parts and labor.",
        ],
        "question": "What does the premium warranty cover, and how much does the premium plan cost per month?",
        "use_marker": True,
    },
    "no_marker": {
        "title": "No evidence provided",
        "description": "No retrieved-evidence marker in the request. The guardrail does not run — the request passes through unchecked (this is the contract).",
        "system": "You are a helpful assistant.",
        "evidence": [],
        "question": "What is the refund policy?",
        "use_marker": False,
    },
}


class RunRequest(BaseModel):
    scenario_id: str | None = None
    # Custom mode (optional):
    system: str | None = None
    evidence: list[str] | None = None
    question: str | None = None
    use_marker: bool = True


def build_messages(system, evidence, question, use_marker):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if use_marker and evidence:
        content = MARKER + "\n" + "\n".join(evidence)
        messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": question})
    return messages


async def generate_via_proxy(messages):
    """Get the model's answer THROUGH the live proxy (unguarded key, so we see
    the raw answer to score and explain)."""
    async with httpx.AsyncClient(timeout=60, verify=config.SSL_VERIFY) as client:
        resp = await client.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {PROXY_KEY}"},
            json={"model": DEMO_MODEL, "messages": messages},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


@app.get("/config")
async def get_config():
    return {
        "threshold": config.FAITHFULNESS_THRESHOLD,
        "mode": config.MODE,
        "fallback_message": config.FALLBACK_MESSAGE,
        "marker": MARKER,
        "scenarios": [
            {"id": sid, "title": s["title"], "description": s["description"]}
            for sid, s in SCENARIOS.items()
        ],
    }


@app.post("/run")
async def run(req: RunRequest):
    if req.scenario_id:
        s = SCENARIOS.get(req.scenario_id)
        if not s:
            return JSONResponse({"error": "unknown scenario"}, status_code=400)
        system, evidence, question, use_marker = (
            s["system"], s["evidence"], s["question"], s["use_marker"],
        )
    else:
        system = req.system or ""
        evidence = req.evidence or []
        question = req.question or ""
        use_marker = req.use_marker

    messages = build_messages(system, evidence, question, use_marker)

    try:
        answer = await generate_via_proxy(messages)
    except Exception as exc:  # surface proxy errors to the UI
        return JSONResponse({"error": f"proxy call failed: {exc}"}, status_code=502)

    result = {
        "evidence": evidence,
        "question": question,
        "answer": answer,
        "threshold": config.FAITHFULNESS_THRESHOLD,
        "mode": config.MODE,
    }

    parsed = parse_messages(messages)
    if not parsed.compliant:
        # No marker (or non-compliant) -> guardrail does not run.
        result.update(
            {
                "guardrail_ran": False,
                "skip_reason": parsed.skip_reason,
                "delivered": answer,  # passes through unchecked
            }
        )
        return result

    try:
        verdict = await evaluator.a_evaluate(
            parsed.input, answer, parsed.retrieval_context
        )
    except Exception as exc:
        return JSONResponse({"error": f"scoring failed: {exc}"}, status_code=502)

    result.update(
        {
            "guardrail_ran": True,
            "score": round(verdict.score, 3),
            "passed": verdict.passed,
            "reason": verdict.reason,
            "unsupported_claims": verdict.unsupported_claims,
            # What a guarded user receives: the answer if grounded, else fallback.
            "delivered": answer if verdict.passed else config.FALLBACK_MESSAGE,
        }
    )
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hallucination Guardrail — Live Demo</title>
<style>
  :root { --ok:#15803d; --okbg:#dcfce7; --bad:#b91c1c; --badbg:#fee2e2;
          --skip:#92400e; --skipbg:#fef3c7; --ink:#0f172a; --muted:#64748b;
          --line:#e2e8f0; --card:#ffffff; --bg:#f1f5f9; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:var(--ink); }
  header { background:#0f172a; color:#fff; padding:22px 28px; }
  header h1 { margin:0; font-size:20px; }
  header p { margin:6px 0 0; color:#cbd5e1; font-size:14px; }
  .wrap { max-width:1000px; margin:0 auto; padding:24px 20px 60px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:16px; cursor:pointer; transition:.15s; }
  .card:hover { border-color:#94a3b8; box-shadow:0 4px 14px rgba(2,6,23,.06); transform:translateY(-1px); }
  .card h3 { margin:0 0 6px; font-size:15px; }
  .card p { margin:0; font-size:13px; color:var(--muted); line-height:1.4; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:14px;
           padding:0; margin-top:22px; overflow:hidden; display:none; }
  .panel.show { display:block; }
  .row { padding:16px 20px; border-bottom:1px solid var(--line); }
  .row:last-child { border-bottom:none; }
  .label { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-bottom:6px; }
  .evi { font-size:14px; }
  .evi li { margin:3px 0; }
  .q { font-size:16px; font-weight:600; }
  .answer { font-size:15px; line-height:1.5; white-space:pre-wrap; }
  .scorebar { height:16px; background:#e2e8f0; border-radius:8px; overflow:hidden; position:relative; margin-top:8px; }
  .scorefill { height:100%; transition:width .5s; }
  .scorehead { display:flex; align-items:baseline; gap:12px; }
  .scorenum { font-size:34px; font-weight:800; }
  .badge { display:inline-block; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:700; }
  .b-ok{ background:var(--okbg); color:var(--ok);} .b-bad{ background:var(--badbg); color:var(--bad);} .b-skip{ background:var(--skipbg); color:var(--skip);}
  .delivered.ok { background:var(--okbg); } .delivered.bad { background:var(--badbg); }
  .delivered { border-radius:10px; padding:12px 14px; font-size:15px; line-height:1.5; }
  .claims li { color:var(--bad); font-size:14px; margin:4px 0; }
  .reason { font-size:13px; color:var(--muted); font-style:italic; margin-top:6px; }
  .spinner { display:none; padding:26px; text-align:center; color:var(--muted); }
  .spinner.show { display:block; }
  .meta { font-size:12px; color:var(--muted); margin-top:4px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle;}
</style>
</head>
<body>
<header>
  <h1>🛡️ Hallucination Guardrail — Live Demo</h1>
  <p>Each answer is generated through the live LiteLLM proxy, then scored for faithfulness against the provided evidence.</p>
</header>
<div class="wrap">
  <div id="cards" class="cards"></div>
  <div id="spinner" class="spinner">⏳ Generating answer through the proxy and scoring it…</div>
  <div id="panel" class="panel"></div>
</div>
<script>
let CFG = null;
async function load() {
  CFG = await (await fetch('/config')).json();
  const cards = document.getElementById('cards');
  cards.innerHTML = CFG.scenarios.map(s =>
    `<div class="card" onclick="run('${s.id}')">
       <h3>${s.title}</h3><p>${s.description}</p>
     </div>`).join('');
}
function pct(x){ return Math.round(x*100); }
async function run(id) {
  document.getElementById('panel').classList.remove('show');
  document.getElementById('spinner').classList.add('show');
  let r;
  try {
    r = await (await fetch('/run', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({scenario_id:id})})).json();
  } catch(e) { r = {error: String(e)}; }
  document.getElementById('spinner').classList.remove('show');
  render(r);
}
function render(r) {
  const p = document.getElementById('panel');
  if (r.error) { p.innerHTML = `<div class="row"><b style="color:var(--bad)">Error:</b> ${r.error}</div>`; p.classList.add('show'); return; }

  const evi = (r.evidence && r.evidence.length)
      ? `<ul class="evi">${r.evidence.map(e=>`<li>${e}</li>`).join('')}</ul>`
      : `<div class="meta">— none provided —</div>`;

  let scoreBlock = '';
  if (r.guardrail_ran === false) {
    scoreBlock = `<div class="row">
        <div class="label">Guardrail decision</div>
        <span class="badge b-skip"><span class="dot" style="background:var(--skip)"></span>Skipped — no evidence marker, request not checked</span>
        <div class="reason">The guardrail only runs when retrieved evidence is supplied per the contract.</div>
      </div>`;
  } else {
    const p100 = pct(r.score), thr = pct(r.threshold);
    const ok = r.passed;
    const color = ok ? 'var(--ok)' : 'var(--bad)';
    const claims = (r.unsupported_claims && r.unsupported_claims.length)
      ? `<div class="row"><div class="label">Claims not supported by the evidence</div>
           <ul class="claims">${r.unsupported_claims.map(c=>`<li>⚠️ ${c}</li>`).join('')}</ul></div>` : '';
    scoreBlock = `
      <div class="row">
        <div class="label">Faithfulness score (0–100%, threshold ${thr}%)</div>
        <div class="scorehead">
          <div class="scorenum" style="color:${color}">${p100}%</div>
          <span class="badge ${ok?'b-ok':'b-bad'}">${ok?'✅ Grounded — allowed':'⛔ Not grounded — blocked'}</span>
        </div>
        <div class="scorebar"><div class="scorefill" style="width:${p100}%; background:${color}"></div></div>
        <div class="reason">${r.reason||''}</div>
      </div>
      ${claims}`;
  }

  const deliveredOk = (r.guardrail_ran === false) || r.passed;
  const delivered = `<div class="row">
      <div class="label">What the user receives ${r.guardrail_ran!==false ? '(mode: '+r.mode+')' : ''}</div>
      <div class="delivered ${deliveredOk?'ok':'bad'}">${r.delivered}</div>
    </div>`;

  p.innerHTML = `
    <div class="row"><div class="label">Evidence provided to the AI</div>${evi}</div>
    <div class="row"><div class="label">Question</div><div class="q">${r.question}</div></div>
    <div class="row"><div class="label">AI's raw answer (from the proxy)</div><div class="answer">${r.answer}</div></div>
    ${scoreBlock}
    ${delivered}`;
  p.classList.add('show');
}
load();
</script>
</body>
</html>
"""
