# Guardrails Architecture

Diagrams for the DeepEval-based LiteLLM guardrails. These render on GitHub and in
any Mermaid-aware viewer (VS Code preview, etc.).

---

## 1. Request lifecycle (where a guardrail sits)

```mermaid
flowchart LR
    C[Client / calling team] -->|chat request<br/>Bearer virtual key| P[LiteLLM Proxy]
    P -->|model_list route| M[Served model<br/>groq/gpt-oss-120b]
    M -->|completion| H{{post_call guardrail hook}}
    H -->|allow| R[Response to client]
    H -->|remediate / block| R

    subgraph EN [Enablement]
      direction TB
      K[Prod: guardrail attached<br/>per virtual key -> transparent]
      B[Demo OSS: guardrails field<br/>in request body -> opt-in]
    end
    EN -.controls.-> H

    J[(Judge model<br/>llama-3.3-70b<br/>via LiteLLM SDK)]
    H <-->|score the answer| J
```

The judge is a **separate** model reached through the LiteLLM SDK (not the served
model), so the guardrail never self-grades and the judge is swappable with one
env var.

---

## 2. Inside one guardrail (the reusable pattern)

Every guardrail is a thin orchestrator over four decoupled parts. Three are
shared; only the **evaluator** (the DeepEval metric) is metric-specific.

```mermaid
flowchart TD
    IN[messages + model response] --> PARSE[Parser<br/>parser.py<br/>extract input / retrieval_context<br/>or SKIP]
    PARSE -->|not applicable| SKIP[Pass response through]
    PARSE -->|clean fields| EVAL[Evaluator<br/>DeepEval metric<br/>METRIC-SPECIFIC]
    EVAL <-->|judge call| JUDGE[LiteLLMJudge<br/>groq_judge.py<br/>shared]
    EVAL --> VERDICT{score >= threshold?}
    VERDICT -->|yes| ALLOW[Allow original answer]
    VERDICT -->|no| ACT[By MODE]
    ACT --> REMEDIATE[remediate: re-answer<br/>remediation.py<br/>shared loop]
    ACT --> BLOCK[block: replace with<br/>fallback message]
    REMEDIATE --> OUT[Final response<br/>reasoning cleared on overwrite]
    BLOCK --> OUT
    ALLOW --> OUT

    classDef shared fill:#e6f0ff,stroke:#3b78e7,color:#0b3d91;
    classDef specific fill:#fff2e6,stroke:#e8892b,color:#7a3e00;
    class PARSE,JUDGE,REMEDIATE shared;
    class EVAL specific;
```

Blue = reused verbatim across guardrails · Orange = the one metric-specific piece.

---

## 3. The three guardrails (independent identities)

```mermaid
flowchart TB
    subgraph SHARED [Shared plumbing - one copy]
      direction LR
      PJ[parser.py]:::s
      GJ[groq_judge.py<br/>LiteLLMJudge]:::s
      RM[remediation.py]:::s
      CF[config.py<br/>namespaced settings]:::s
    end

    F[Faithfulness<br/>hook.py<br/>grounding.py<br/>metric: Faithfulness<br/>fields: input, output, context<br/>mode: remediate]:::g
    A[Answer Relevancy<br/>relevancy_hook.py<br/>relevancy.py<br/>metric: AnswerRelevancy<br/>fields: input, output<br/>mode: remediate]:::g
    X[Contextual Relevancy<br/>contextual_relevancy_hook.py<br/>contextual_relevancy.py<br/>metric: ContextualRelevancy<br/>fields: input, context<br/>mode: block]:::g

    SHARED --> F
    SHARED --> A
    SHARED --> X

    classDef s fill:#e6f0ff,stroke:#3b78e7,color:#0b3d91;
    classDef g fill:#eaf7ea,stroke:#2e8b57,color:#14532d;
```

Each guardrail has its own class, `guardrail_name`, and config namespace, and is
registered as a **separate** entry in `config.yaml` — enable any combination
independently (per key in production).

---

## 4. Decision outcomes

```mermaid
flowchart LR
    S[score vs threshold] --> P{pass?}
    P -->|yes| AL[ALLOW original]
    P -->|no, faithfulness / answer relevancy| RE[REMEDIATE<br/>re-answer, re-score,<br/>fallback if still failing]
    P -->|no, contextual relevancy| BL[BLOCK<br/>safe fallback<br/>no remediation]
    RE --> D[Deliver]
    BL --> D
    AL --> D
```

**Why contextual relevancy blocks instead of remediating:** it grades the
*retriever*. Retrieval happened upstream in the caller's RAG pipeline, so
re-prompting the model cannot improve the score — the safe action is to block.

---

## 5. Live vs offline (scope of the DeepEval RAG metrics)

```mermaid
flowchart TB
    subgraph LIVE [Live guardrails - reference-free]
      F2[Faithfulness]:::ok
      A2[Answer Relevancy]:::ok
      X2[Contextual Relevancy]:::ok
    end
    subgraph OFFLINE [Offline only - need ground-truth expected_output]
      CP[Contextual Precision]:::no
      CR[Contextual Recall]:::no
    end
    NOTE[expected_output does not exist<br/>at inference time]:::note
    OFFLINE --- NOTE

    classDef ok fill:#eaf7ea,stroke:#2e8b57,color:#14532d;
    classDef no fill:#fdeaea,stroke:#c0392b,color:#7a1f1f;
    classDef note fill:#f5f5f5,stroke:#999,color:#333;
```
