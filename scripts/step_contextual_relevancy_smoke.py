"""Live smoke test for the contextual-relevancy evaluator.

Requires GROQ_API_KEY (or another judge via GUARDRAIL_JUDGE_* env vars).
Exercises the real judge on the RETRIEVER quality:
  1. Relevant context   -> expect passed=True, no irrelevant context.
  2. Irrelevant context -> expect passed=False and populated irrelevant_context.

Contextual Relevancy grades the retrieved context against the question; it does
not look at any answer. Run:  python scripts/step_contextual_relevancy_smoke.py
"""

import os
import sys

# Make the repo root importable no matter where this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from guardrail.contextual_relevancy import ContextualRelevancyEvaluator

QUESTION = "What is the capital of France?"


def show(title, verdict):
    print(f"\n=== {title} ===")
    print("score            :", verdict.score)
    print("passed           :", verdict.passed)
    print("reason           :", verdict.reason)
    print("irrelevant_context:", verdict.irrelevant_context)


def main():
    evaluator = ContextualRelevancyEvaluator()  # judge model + threshold from config

    relevant = evaluator.evaluate(
        input=QUESTION,
        retrieval_context=[
            "Paris is the capital and most populous city of France.",
            "The French government is seated in Paris.",
        ],
    )
    show("RELEVANT CONTEXT (expect passed=True)", relevant)

    irrelevant = evaluator.evaluate(
        input=QUESTION,
        retrieval_context=[
            "Brie and Roquefort are famous French cheeses.",
            "Summers in the south of France are warm and dry.",
            "The TGV is a high-speed train network.",
        ],
    )
    show("IRRELEVANT CONTEXT (expect passed=False + irrelevant context)", irrelevant)


if __name__ == "__main__":
    main()
