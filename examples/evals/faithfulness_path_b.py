"""Path B faithfulness — deterministic marker parse + clean judge call (production).

See MLFLOW_LITELLM_HANDOFF.md §9. Path B replaces the "let the LLM find the marker"
step of Path A with the deterministic parser in ``evidence_marker.py`` (exact marker,
nearest-preceding rule, plain-string content, skip-if-absent — all unit-tested). The
judge then only decides GROUNDEDNESS over already-clean evidence, which removes the
fuzzy multi-turn extraction risk (Path A test case 4b) and makes verdicts reproducible.

Why not ``make_judge`` for Path B: MLflow judges accept only reserved template vars,
so a pre-parsed ``{{ retrieval_context }}`` cannot be injected (handoff §6.6). Path B
therefore calls the judge itself (OpenAI-compatible, through the LiteLLM proxy) with
the parsed evidence, and wraps the result as an MLflow ``@scorer`` so it still plugs
into ``mlflow.genai.evaluate`` / scheduled scorers like any native scorer.

Runs OFFLINE + SAMPLED (handoff §5.6).

Setup (on the VM):
    export OPENAI_API_KEY=sk-123
    export OPENAI_API_BASE=http://127.0.0.1:4010/v1     # the LiteLLM proxy
    export OPENAI_BASE_URL=http://127.0.0.1:4010/v1
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Usage:
    python faithfulness_path_b.py                 # run the labeled set, report agreement
    python faithfulness_path_b.py --sample-traces # score a sample of real traces
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
from mlflow.genai.scorers import scorer
from openai import OpenAI

from evidence_marker import EVIDENCE_MARKER, parse_evidence

# The judge model as registered in the proxy (handoff §4). Instruct 70B in dev;
# infra repoints it to a self-hosted OSS instruct model in prod.
JUDGE_MODEL_NAME = os.environ.get("FAITHFULNESS_JUDGE_MODEL", "judge-model")

LABELED_SET = Path(__file__).with_name("faithfulness_labeled_set.jsonl")

PASS, FAIL, SKIP = "pass", "fail", "skip"

_client: OpenAI | None = None


def _judge_client() -> OpenAI:
    """Reused OpenAI client pointed at the LiteLLM proxy (instantiate once)."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:4010/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "sk-123"),
        )
    return _client


_JUDGE_SYSTEM = (
    "You grade FAITHFULNESS. You are given EVIDENCE (the only source of truth), a "
    "QUESTION, and a RESPONSE. Decide whether every factual claim in the RESPONSE is "
    "supported by (or directly entailed by) the EVIDENCE. Do not use outside knowledge. "
    "Correctly declining to answer, or stating the evidence does not cover something, "
    "is faithful. Reply with a single JSON object and nothing else: "
    '{"verdict": "pass" | "fail", "unsupported_claims": [string], "rationale": string}. '
    'Use "fail" if the response asserts anything unsupported or contradicting the evidence.'
)


def _judge_faithfulness(evidence: list[str], question: str, response: str) -> Feedback:
    """Call the instruct judge over CLEAN evidence and return an MLflow Feedback."""
    user = (
        "EVIDENCE:\n" + "\n".join(f"- {e}" for e in evidence) +
        f"\n\nQUESTION:\n{question}\n\nRESPONSE:\n{response}"
    )
    source = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=JUDGE_MODEL_NAME)
    try:
        completion = _judge_client().chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        verdict = str(data.get("verdict", "")).strip().lower()
        verdict = FAIL if verdict == FAIL else (PASS if verdict == PASS else FAIL)
        unsupported = data.get("unsupported_claims") or []
        rationale = data.get("rationale", "")
        if unsupported:
            rationale = f"{rationale} Unsupported: {unsupported}"
        return Feedback(name="faithfulness", value=verdict, rationale=rationale, source=source)
    except Exception as e:  # noqa: BLE001 — surface judge errors as an error assessment, not a crash
        return Feedback(name="faithfulness", error=e, source=source)


def _messages_from(inputs, trace) -> list | None:
    """Recover the raw ``messages`` array from a dataset row or a captured trace."""
    if isinstance(inputs, dict):
        msgs = inputs.get("messages")
        if isinstance(msgs, list):
            return msgs
    # Fall back to the trace's request payload (real captured traffic).
    if trace is not None:
        request = getattr(trace, "request", None) or getattr(getattr(trace, "info", None), "request_preview", None)
        if isinstance(request, str):
            try:
                payload = json.loads(request)
                if isinstance(payload.get("messages"), list):
                    return payload["messages"]
            except (json.JSONDecodeError, AttributeError):
                return None
    return None


@scorer
def faithfulness(inputs=None, outputs=None, trace=None) -> Feedback | None:
    """Path B MLflow scorer: deterministic parse -> judge over clean evidence.

    Returns ``None`` (skip) when the request carries no compliant marker, so
    contract-free traffic is never falsely failed.
    """
    messages = _messages_from(inputs, trace)
    if messages is None:
        return None
    parsed = parse_evidence(messages)
    if parsed is None:
        return None  # no marker -> skip (opt-in contract)
    response = outputs if isinstance(outputs, str) else json.dumps(outputs)
    return _judge_faithfulness(parsed.evidence, parsed.question, response)


# --- Validation against the shared labeled set -----------------------------
def _load_labeled():
    with LABELED_SET.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_against_labeled_set() -> int:
    """Score the labeled set and report agreement (handoff §9 test 5)."""
    rows = _load_labeled()
    correct = 0
    tp = fp = fn = 0
    print(f"Path B validation on {len(rows)} labeled examples — judge={JUDGE_MODEL_NAME}\n")
    for row in rows:
        parsed = parse_evidence(row["messages"])
        if parsed is None:
            got = SKIP
            rationale = "no compliant marker -> skipped deterministically"
        else:
            fb = _judge_faithfulness(parsed.evidence, parsed.question, row["response"])
            got = str(fb.value).lower() if fb.value else "error"
            rationale = fb.rationale or (str(fb.error) if fb.error else "")
        exp = row["label"]
        hit = got == exp
        correct += hit
        if exp == FAIL and got == FAIL:
            tp += 1
        elif exp != FAIL and got == FAIL:
            fp += 1
        elif exp == FAIL and got != FAIL:
            fn += 1
        print(f"[{'OK ' if hit else 'XX '}] {row['id']}: expected={exp} got={got}  ({row.get('note','')})")
        if not hit:
            print(f"      rationale: {rationale}")
    acc = correct / len(rows) if rows else 0.0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"\nagreement={acc:.0%} ({correct}/{len(rows)})")
    print(f"fail-detection precision={prec:.2f} recall={rec:.2f} (tp={tp} fp={fp} fn={fn})")
    print("Path B's skip decisions are deterministic; only pass/fail depends on the judge.")
    return len(rows) - correct


def sample_traces_and_evaluate(experiment: str, sample: int):
    """Path B on real captured traffic via mlflow.genai.evaluate (offline+sampled)."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"experiment not found: {experiment}")
    traces = mlflow.search_traces(locations=[exp.experiment_id], max_results=max(sample * 5, 50))
    has_marker = traces[traces["request"].astype(str).str.contains(EVIDENCE_MARKER, regex=False)]
    subset = has_marker.head(sample)
    print(f"{len(subset)} marker-bearing traces sampled from '{experiment}' (of {len(traces)} fetched)")
    if subset.empty:
        print("No marker-bearing traces to grade — the faithfulness contract is opt-in.")
        return
    results = mlflow.genai.evaluate(data=subset, scorers=[faithfulness])
    print("Aggregate metrics:", results.metrics)


def main():
    ap = argparse.ArgumentParser(description="Path B deterministic-parse faithfulness scorer")
    ap.add_argument("--sample-traces", metavar="EXPERIMENT", nargs="?", const="litellm-proxy-demo")
    ap.add_argument("--sample", type=int, default=20, help="trace sample size")
    args = ap.parse_args()
    if args.sample_traces:
        sample_traces_and_evaluate(args.sample_traces, args.sample)
        return
    raise SystemExit(1 if validate_against_labeled_set() else 0)


if __name__ == "__main__":
    main()
