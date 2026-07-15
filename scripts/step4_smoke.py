"""Live smoke test for the grounding evaluator (Step 4).

Requires GROQ_API_KEY in the environment. Exercises the real Groq judge:
  1. A faithful answer  -> expect passed=True, no unsupported claims.
  2. An unfaithful answer (hardcoded so we don't have to coax a model into
     hallucinating) -> expect passed=False and populated unsupported_claims.

Run:  python scripts/step4_smoke.py
"""

import os
import sys

# Make the repo root importable no matter where this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from guardrail.grounding import GroundingEvaluator

RETRIEVAL_CONTEXT = [
    "All customers are eligible for a 30 day full refund at no extra cost.",
    "Refunds are processed within 5 business days.",
]


def show(title, verdict):
    print(f"\n=== {title} ===")
    print("score            :", verdict.score)
    print("passed           :", verdict.passed)
    print("reason           :", verdict.reason)
    print("unsupported_claims:", verdict.unsupported_claims)


def main():
    evaluator = GroundingEvaluator()  # uses judge model + threshold from config

    faithful = evaluator.evaluate(
        input="What is the refund policy?",
        actual_output="Customers can get a full refund within 30 days at no extra cost.",
        retrieval_context=RETRIEVAL_CONTEXT,
    )
    show("FAITHFUL (expect passed=True)", faithful)

    unfaithful = evaluator.evaluate(
        input="What is the refund policy?",
        actual_output=(
            "Customers get a 90 day refund and also receive a free replacement "
            "phone with every return."
        ),
        retrieval_context=RETRIEVAL_CONTEXT,
    )
    show("UNFAITHFUL (expect passed=False + unsupported claims)", unfaithful)


if __name__ == "__main__":
    main()
