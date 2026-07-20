"""Path A faithfulness — an LLM-extract custom judge (MLflow, judged via LiteLLM).

See MLFLOW_LITELLM_HANDOFF.md §9. Path A asks the *judge itself* to find the
``--- Retrieved Evidence ---`` marker inside ``{{ inputs }}`` and grade whether the
response is grounded in it. No code parser — the LLM does the extraction. This is
the fast-to-build, "does the idea work at all" path; Path B (faithfulness_path_b.py)
adds a deterministic parser for production precision.

Why the judge can only see the raw marker: MLflow ``make_judge`` allows ONLY the
reserved template vars ``{{ inputs }}``, ``{{ outputs }}``, ``{{ expectations }}``,
``{{ trace }}``, ``{{ conversation }}`` — you cannot inject a pre-parsed
``{{ retrieval_context }}`` (handoff §6.6). So the marker must ride in via
``{{ inputs }}`` and the judge extracts it.

Runs OFFLINE + SAMPLED (handoff §5.6) — never on 100% of live traffic.

Setup (on the VM):
    export OPENAI_API_KEY=sk-123
    export OPENAI_API_BASE=http://127.0.0.1:4010/v1     # the LiteLLM proxy
    export OPENAI_BASE_URL=http://127.0.0.1:4010/v1
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Usage:
    python faithfulness_path_a.py                 # run the 5 §9 test cases
    python faithfulness_path_a.py --validate      # score the labeled set + agreement
    python faithfulness_path_a.py --sample-traces # score a sample of real traces
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.genai.judges import make_judge

# The judge as registered in the proxy (handoff §4: model_name `judge-model` =
# groq/llama-3.3-70b-versatile, an *instruct* model — reasoning models are
# unreliable judges, handoff §6.1). openai:/ + OPENAI_API_BASE route through the
# proxy; in prod infra repoints `judge-model` at a self-hosted OSS instruct model
# and this file does not change.
JUDGE_MODEL = os.environ.get("FAITHFULNESS_JUDGE", "openai:/judge-model")

LABELED_SET = Path(__file__).with_name("faithfulness_labeled_set.jsonl")

# Categorical verdicts the judge must emit.
PASS, FAIL, SKIP = "pass", "fail", "skip"

INSTRUCTIONS = """\
You are grading FAITHFULNESS: is the response grounded in the retrieved evidence?

The request messages are in {{ inputs }} and the response to grade is {{ outputs }}.

Locating the evidence:
- The retrieved evidence is the message whose content BEGINS WITH the exact string
  "--- Retrieved Evidence ---". The evidence is the text after that marker, one
  chunk per line.
- If several messages carry that marker, use ONLY the one that most closely
  PRECEDES the final user message (the current question). Ignore older marker
  blocks — they are stale context from earlier turns.
- Treat that evidence as the ONLY source of truth. Do NOT use your own world
  knowledge to fill gaps.

Grading:
- If NO message begins with the exact marker, you cannot judge faithfulness.
  Return "skip".
- Otherwise, check every factual claim in {{ outputs }} against the evidence:
  - "pass" if every claim is supported by (or directly entailed by) the evidence.
    Declining to answer, or correctly saying the evidence does not cover something,
    is faithful -> "pass".
  - "fail" if the response asserts anything not supported by the evidence, or that
    contradicts it (wrong numbers, invented details, extra items/channels/tiers).

Return exactly one of: "pass", "fail", "skip". In your rationale, name the specific
unsupported or contradicting claim(s) when you return "fail"."""


def build_judge():
    """The single custom faithfulness judge used by Path A."""
    return make_judge(
        name="faithfulness_llm_extract",
        instructions=INSTRUCTIONS,
        model=JUDGE_MODEL,
    )


# --- The 5 canonical §9 test cases -----------------------------------------
MARKER = "--- Retrieved Evidence ---"

TEST_CASES = [
    {
        "name": "1_faithful_with_marker",
        "expected": PASS,
        "inputs": {"messages": [
            {"role": "assistant", "content": f"{MARKER}\nAll customers get a full refund within 30 days."},
            {"role": "user", "content": "What is the refund policy?"},
        ]},
        "outputs": "You can get a full refund within 30 days.",
    },
    {
        "name": "2_hallucinated_with_marker",
        "expected": FAIL,
        "inputs": {"messages": [
            {"role": "assistant", "content": f"{MARKER}\nAll customers get a full refund within 30 days."},
            {"role": "user", "content": "What is the refund policy?"},
        ]},
        "outputs": "You get a full refund within 30 days, plus a $50 goodwill credit.",
    },
    {
        "name": "3_no_marker_skip",
        "expected": SKIP,
        "inputs": {"messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ]},
        "outputs": "The capital of France is Paris.",
    },
    {
        "name": "4_multiturn_nearest_marker",
        "expected": PASS,
        "inputs": {"messages": [
            {"role": "user", "content": "What is the return policy?"},
            {"role": "assistant", "content": f"{MARKER}\nFull refund within 30 days."},
            {"role": "user", "content": "And the warranty on premium plans?"},
            {"role": "assistant", "content": f"{MARKER}\nPremium plans get a 2-year extended warranty."},
            {"role": "user", "content": "Does the premium plan include a warranty, and how long?"},
        ]},
        "outputs": "Yes, premium plans include a 2-year extended warranty.",
    },
    {
        "name": "4b_multiturn_uses_stale_marker",
        "expected": FAIL,  # response grounded in the STALE block, not the nearest one
        "inputs": {"messages": [
            {"role": "user", "content": "What is the return policy?"},
            {"role": "assistant", "content": f"{MARKER}\nFull refund within 30 days."},
            {"role": "user", "content": "And the warranty on premium plans?"},
            {"role": "assistant", "content": f"{MARKER}\nPremium plans get a 2-year extended warranty."},
            {"role": "user", "content": "How long is the premium warranty?"},
        ]},
        "outputs": "Premium plans get a 30-day warranty.",
    },
]


def _verdict(feedback) -> str:
    """Normalize a judge Feedback's value to one of pass/fail/skip."""
    value = str(getattr(feedback, "value", feedback)).strip().lower()
    for v in (PASS, FAIL, SKIP):
        if v in value:
            return v
    return value


def run_test_cases() -> int:
    """Directly invoke the judge on the 5 §9 cases; report against expectations."""
    judge = build_judge()
    mismatches = 0
    print(f"Path A faithfulness — judge={JUDGE_MODEL}\n")
    for tc in TEST_CASES:
        fb = judge(inputs=tc["inputs"], outputs=tc["outputs"])
        got = _verdict(fb)
        ok = "OK " if got == tc["expected"] else "XX "
        if got != tc["expected"]:
            mismatches += 1
        print(f"[{ok}] {tc['name']}: expected={tc['expected']} got={got}")
        print(f"      rationale: {getattr(fb, 'rationale', '')}\n")
    print(f"{len(TEST_CASES) - mismatches}/{len(TEST_CASES)} matched expectation")
    print(
        "\nNOTE case 4b is the fuzzy one: Path A relies on the LLM honoring the "
        "'nearest-preceding marker' rule. Path B enforces it deterministically."
    )
    return mismatches


def _load_labeled():
    with LABELED_SET.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_against_labeled_set() -> int:
    """Score the hand-labeled set and report agreement (handoff §9 test 5)."""
    judge = build_judge()
    rows = _load_labeled()
    correct = 0
    # Confusion tallies for the fail-detection view that matters for a gate.
    tp = fp = fn = 0
    print(f"Path A validation on {len(rows)} labeled examples — judge={JUDGE_MODEL}\n")
    for row in rows:
        fb = judge(inputs={"messages": row["messages"]}, outputs=row["response"])
        got = _verdict(fb)
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
    acc = correct / len(rows) if rows else 0.0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"\nagreement={acc:.0%} ({correct}/{len(rows)})")
    print(f"fail-detection precision={prec:.2f} recall={rec:.2f} (tp={tp} fp={fp} fn={fn})")
    print("Trust the judge to gate only once agreement is high AND fp (false blocks) is low.")
    return len(rows) - correct


def evaluate_offline_batch():
    """Canonical offline batch via mlflow.genai.evaluate over the test cases."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("litellm-eval")
    data = [{"inputs": tc["inputs"], "outputs": tc["outputs"]} for tc in TEST_CASES]
    results = mlflow.genai.evaluate(data=data, scorers=[build_judge()])
    print("Aggregate metrics:", results.metrics)
    print("Per-row verdicts + rationale in MLflow UI -> Experiments -> litellm-eval")


def sample_traces_and_evaluate(experiment: str, sample: int):
    """Path A on real captured traffic: sample marker-bearing traces and grade.

    Offline + sampled (handoff §5.6). We over-fetch then keep only traces whose
    request actually contains the marker, so the judge is not run on traffic that
    would just skip.
    """
    from evidence_marker import EVIDENCE_MARKER

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"experiment not found: {experiment}")
    # locations= (not the deprecated experiment_ids=) per handoff §6.5.
    traces = mlflow.search_traces(locations=[exp.experiment_id], max_results=max(sample * 5, 50))
    has_marker = traces[traces["request"].astype(str).str.contains(EVIDENCE_MARKER, regex=False)]
    subset = has_marker.head(sample)
    print(f"{len(subset)} marker-bearing traces sampled from '{experiment}' (of {len(traces)} fetched)")
    if subset.empty:
        print("No marker-bearing traces to grade — teams must add the evidence marker (opt-in).")
        return
    results = mlflow.genai.evaluate(data=subset, scorers=[build_judge()])
    print("Aggregate metrics:", results.metrics)


def main():
    ap = argparse.ArgumentParser(description="Path A LLM-extract faithfulness judge")
    ap.add_argument("--validate", action="store_true", help="score the labeled set + agreement")
    ap.add_argument("--batch", action="store_true", help="canonical mlflow.genai.evaluate batch")
    ap.add_argument("--sample-traces", metavar="EXPERIMENT", nargs="?", const="litellm-proxy-demo")
    ap.add_argument("--sample", type=int, default=20, help="trace sample size")
    args = ap.parse_args()

    if args.validate:
        raise SystemExit(1 if validate_against_labeled_set() else 0)
    if args.batch:
        evaluate_offline_batch()
        return
    if args.sample_traces:
        sample_traces_and_evaluate(args.sample_traces, args.sample)
        return
    raise SystemExit(1 if run_test_cases() else 0)


if __name__ == "__main__":
    main()
