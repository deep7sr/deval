#!/usr/bin/env bash
# Single-terminal demo helper for the hallucination guardrail.
#
#   ./demo.sh grounded     # answer matches the evidence   -> high score, allowed
#   ./demo.sh offsource    # answer contradicts the doc    -> caught, retried
#   ./demo.sh nomarker     # no evidence supplied          -> guardrail skipped
#
# Runs the request through the proxy, prints the delivered answer, then shows
# that request's guardrail log lines (score / retries / outcome) underneath.
set -euo pipefail
cd "$(dirname "$0")"

PROXY="${PROXY:-http://localhost:4001}"
KEY="${KEY:-sk-123}"
MODEL="${MODEL:-groq-test-model}"

case "${1:-}" in
  grounded)
    DESC="GROUNDED — the answer is supported by the evidence"
    DATA='{"model":"'"$MODEL"'","messages":[
      {"role":"system","content":"Answer using only the retrieved evidence."},
      {"role":"assistant","content":"--- Retrieved Evidence ---\nAll customers are eligible for a 30 day full refund at no extra cost.\nRefunds are processed within 5 business days."},
      {"role":"user","content":"What is the refund policy?"}
    ]}' ;;
  offsource)
    DESC="OFF-SOURCE — the AI answers from memory, not the provided document"
    DATA='{"model":"'"$MODEL"'","messages":[
      {"role":"assistant","content":"--- Retrieved Evidence ---\nThe Eiffel Tower is located in Rome, Italy."},
      {"role":"user","content":"Where is the Eiffel Tower located?"}
    ]}' ;;
  partial)
    DESC="PARTIAL — evidence mixes a correct and an incorrect fact (~0.5)"
    # France->Paris matches the evidence (grounded); Italy->London is wrong, so
    # the model contradicts it (unfaithful). One of two claims -> ~0.5.
    DATA='{"model":"'"$MODEL"'","messages":[
      {"role":"assistant","content":"--- Retrieved Evidence ---\nParis is the capital of France.\nLondon is the capital of Italy."},
      {"role":"user","content":"What is the capital of France, and what is the capital of Italy?"}
    ]}' ;;
  nomarker)
    DESC="NO EVIDENCE — no marker in the request, guardrail does not run"
    DATA='{"model":"'"$MODEL"'","messages":[
      {"role":"user","content":"What is the refund policy?"}
    ]}' ;;
  *)
    echo "usage: ./demo.sh <grounded|offsource|nomarker>"; exit 1 ;;
esac

echo "════════════════════════════════════════════════════════════════"
echo " $DESC"
echo "════════════════════════════════════════════════════════════════"
START="$(date -u +%Y-%m-%dT%H:%M:%S)"
ANSWER="$(curl -s "$PROXY/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "$DATA" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')"

echo
echo "AI answer delivered to the user:"
echo "  $ANSWER"
echo
echo "Guardrail decision:"
sleep 1
if ! docker compose logs --since "$START" litellm-guardrail 2>/dev/null \
      | grep GUARDRAIL | sed 's/.*GUARDRAIL/  GUARDRAIL/'; then
  echo "  (no guardrail line found for this request)"
fi
echo
