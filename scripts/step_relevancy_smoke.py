"""Live smoke test for the answer-relevancy evaluator.

Requires GROQ_API_KEY in the environment (or configure another judge via the
GUARDRAIL_JUDGE_* env vars). Exercises the real judge:
  1. An on-topic answer   -> expect passed=True, no irrelevant statements.
  2. An off-topic answer   -> expect passed=False and populated
     irrelevant_statements.

Run:  python scripts/step_relevancy_smoke.py
"""

import os
import sys

# Make the repo root importable no matter where this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from guardrail.relevancy import RelevancyEvaluator

QUESTION = "What is the capital of France, and roughly how many people live there?"


def show(title, verdict):
    print(f"\n=== {title} ===")
    print("score               :", verdict.score)
    print("passed              :", verdict.passed)
    print("reason              :", verdict.reason)
    print("irrelevant_statements:", verdict.irrelevant_statements)


def main():
    evaluator = RelevancyEvaluator()  # uses judge model + threshold from config

    on_topic = evaluator.evaluate(
        input=QUESTION,
        actual_output=(
            "The capital of France is Paris, which has a population of "
            "roughly 2.1 million people in the city proper."
        ),
    )
    show("ON-TOPIC (expect passed=True)", on_topic)

    off_topic = evaluator.evaluate(
        input=QUESTION,
        actual_output=(
            "France is a wonderful country to visit. The food is amazing, "
            "especially the cheese and wine, and the countryside is beautiful "
            "in the summer. You should also try to learn some French before "
            "you go."
        ),
    )
    show("OFF-TOPIC (expect passed=False + irrelevant statements)", off_topic)


if __name__ == "__main__":
    main()
