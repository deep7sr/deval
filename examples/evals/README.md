# MLflow GenAI evals (judged via the LiteLLM proxy)

All scripts route the judge through the LiteLLM proxy (OpenAI-compatible), so they
don't change between dev (Groq open-weight model) and prod (self-hosted OSS model) —
infra just repoints the `judge` / `judge-model` entry in the proxy config. See the
top-level `MLFLOW_LITELLM_HANDOFF.md` for the full project context.

## Files

| File | Purpose |
|---|---|
| `run_eval.py` | Variety pack of native scorers (Correctness, RelevanceToQuery, Guidelines) on a small dataset. |
| `evidence_marker.py` | Deterministic parser for the `--- Retrieved Evidence ---` faithfulness contract. Pure Python, no deps. |
| `test_evidence_marker.py` | Unit tests for the parser — runnable with no MLflow/network. |
| `faithfulness_path_a.py` | **Path A** faithfulness: one `make_judge` custom judge that extracts the marker from `{{ inputs }}` itself (LLM-extract). |
| `faithfulness_path_b.py` | **Path B** faithfulness: deterministic parse (via `evidence_marker.py`) → clean judge call, wrapped as an MLflow `@scorer` (production path). |
| `faithfulness_labeled_set.jsonl` | 16 hand-labeled examples (pass/fail/skip) shared by Path A and Path B for validation. |

## Faithfulness contract (opt-in)

A request is covered by faithfulness only if it follows the contract: the retrieved
evidence goes in a message whose `content` **starts with the exact string**
`--- Retrieved Evidence ---` (one chunk per line after it); the question is the final
`user` message; with multiple marker messages only the **nearest-preceding** one is
used. No marker → the check is skipped (never a false fail).

## Running

```bash
# Parser unit tests — no venv or network needed:
python examples/evals/test_evidence_marker.py

# Judge-dependent runs — need the proxy + MLflow + a judge model reachable:
export OPENAI_API_KEY=sk-123
export OPENAI_API_BASE=http://127.0.0.1:4010/v1     # the LiteLLM proxy
export OPENAI_BASE_URL=http://127.0.0.1:4010/v1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python examples/evals/faithfulness_path_a.py             # 5 canonical test cases
python examples/evals/faithfulness_path_a.py --validate  # agreement on the labeled set
python examples/evals/faithfulness_path_b.py             # agreement on the labeled set
python examples/evals/faithfulness_path_b.py --sample-traces   # score sampled real traces
```

Faithfulness is run **offline + sampled** (not on 100% of live traffic).
