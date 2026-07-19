"""Offline GenAI evaluation with MLflow, judged by a self-hosted OSS model.

The judge model runs on our own hardware and is reached THROUGH the LiteLLM proxy
(OpenAI-compatible), so no eval data leaves the network and there is no frontier cost.

Run:
    export OPENAI_API_BASE=http://litellm:4000/v1     # our proxy, not api.openai.com
    export OPENAI_API_KEY=sk-litellm-virtual-key       # a LiteLLM virtual key
    python run_eval.py

Docs:
    https://mlflow.org/docs/latest/genai/eval-monitor/
    https://docs.litellm.ai/docs/tutorials/eval_suites
"""

import os

import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery
from openai import OpenAI

# --- MLflow connection ------------------------------------------------------
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("litellm-eval")

# The judge model, as registered in the LiteLLM proxy (see litellm_config.yaml).
# MLflow uses provider:/model URIs; openai:/ + OPENAI_API_BASE routes to our proxy.
JUDGE = "openai:/judge-qwen"

# --- The app under test: call it through the same proxy ---------------------
proxy = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE", "http://litellm:4000/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
)


def predict_fn(question: str) -> str:
    resp = proxy.chat.completions.create(
        model="gpt-4o",  # whatever model the app actually uses
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content


# --- A custom faithfulness/groundedness judge (reuses the guardrail idea) ----
faithfulness = make_judge(
    name="faithfulness",
    instructions=(
        "Grade whether the response is fully supported by the retrieved evidence.\n\n"
        "Evidence:\n{{ inputs }}\n\nResponse:\n{{ outputs }}\n\n"
        "Return 'pass' only if every claim in the response is supported by the "
        "evidence; otherwise 'fail'. Give a one-sentence reason."
    ),
    model=JUDGE,
)

# --- Evaluation dataset: inputs (+ optional expectations) -------------------
data = [
    {
        "inputs": {"question": "What is our refund window?"},
        "expectations": {"expected_response": "30 days"},
    },
    {
        "inputs": {"question": "How long does a refund take to process?"},
        "expectations": {"expected_response": "5 business days"},
    },
]

# --- Run: every scorer is judged by the OSS model via the proxy -------------
results = mlflow.genai.evaluate(
    data=data,
    predict_fn=predict_fn,
    scorers=[
        Correctness(model=JUDGE),
        RelevanceToQuery(model=JUDGE),
        Guidelines(
            name="tone",
            guidelines="The answer must be professional and written in English.",
            model=JUDGE,
        ),
        faithfulness,
    ],
)

print("Aggregate metrics:", results.metrics)
print("View per-row scores and judge rationale in the MLflow UI → Experiments → litellm-eval")


# --- Alternative: evaluate REAL logged production traffic -------------------
# Because the proxy logs every call as a trace (success_callback: ["mlflow"]),
# we can score actual traffic instead of synthetic prompts:
#
#   traces = mlflow.search_traces(
#       experiment_names=["litellm-proxy-prod"],
#       filter_string="tags.taskName = 'run_page_classification'",
#       max_results=200,
#   )
#   mlflow.genai.evaluate(data=traces, scorers=[RelevanceToQuery(model=JUDGE)])
