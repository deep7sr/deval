"""Live smoke test for the turn-faithfulness evaluator.

Requires GROQ_API_KEY in the environment (or configure another judge via the
GUARDRAIL_JUDGE_* env vars). Exercises the real judge over a two-exchange
conversation:
  1. A faithful conversation      -> expect passed=True, no unfaithful claims.
  2. The same conversation whose  -> expect passed=False and populated
     final answer contradicts the    unfaithful_claims.
     evidence

Remember (learning carried over from the single-turn metric): faithfulness
penalizes CONTRADICTIONS, not additions - so the failing case must contradict
an evidence fact, not merely add one.

Run:  python scripts/step_turn_faithfulness_smoke.py
"""

import os
import sys

# Make the repo root importable no matter where this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from guardrail.turn_faithfulness import TurnFaithfulnessEvaluator

# History: first exchange (already answered faithfully), then the final
# question. The hook would append the model's new answer as the last turn;
# here we do it by hand for both scenarios.
HISTORY = [
    {
        "role": "user",
        "content": "Where is the Eiffel Tower?",
        "retrieval_context": None,
    },
    {
        "role": "assistant",
        "content": "The Eiffel Tower is in Paris, France.",
        "retrieval_context": ["The Eiffel Tower is located in Paris, France."],
    },
    {
        "role": "user",
        "content": "What is it made of and how tall is it?",
        "retrieval_context": None,
    },
]

FINAL_CONTEXT = [
    "The Eiffel Tower is made of wrought iron.",
    "The Eiffel Tower is 330 metres tall.",
]


def show(title, verdict):
    print(f"\n=== {title} ===")
    print("score            :", verdict.score)
    print("passed           :", verdict.passed)
    print("reason           :", verdict.reason)
    print("windows_evaluated:", verdict.windows_evaluated)
    print("unfaithful_claims:", verdict.unfaithful_claims)


def main():
    evaluator = TurnFaithfulnessEvaluator()  # judge model + threshold from config

    faithful = evaluator.evaluate(
        HISTORY
        + [
            {
                "role": "assistant",
                "content": (
                    "It is made of wrought iron and stands 330 metres tall."
                ),
                "retrieval_context": FINAL_CONTEXT,
            }
        ]
    )
    show("FAITHFUL conversation (expect passed=True)", faithful)

    unfaithful = evaluator.evaluate(
        HISTORY
        + [
            {
                "role": "assistant",
                "content": (
                    "The Eiffel Tower is made entirely of wood and is about "
                    "150 metres tall."
                ),
                "retrieval_context": FINAL_CONTEXT,
            }
        ]
    )
    show("UNFAITHFUL final answer (expect passed=False + claims)", unfaithful)


if __name__ == "__main__":
    main()
