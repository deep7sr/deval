# Self-contained LiteLLM proxy image with the hallucination guardrail's judge
# dependency (DeepEval) baked in. The guardrail code itself is volume-mounted
# at /app/guardrail during testing (see docker-compose.yml), so iterating on it
# only needs a container restart, not an image rebuild.
FROM ghcr.io/berriai/litellm:v1.92.0

USER root

# DeepEval provides the FaithfulnessMetric used by the guardrail. litellm is
# already present in the base image. Pinned to the version validated locally.
# --trusted-host bypasses TLS verification for PyPI during the build, because
# the corporate SSL-interception CA is not present in this build image. For a
# hardened production build, prefer baking the corporate CA into the image (or
# using an internal PyPI mirror) instead - see deploy/README.md.
RUN /app/.venv/bin/python -m ensurepip --upgrade \
    && /app/.venv/bin/python -m pip install --no-cache-dir \
       --trusted-host pypi.org \
       --trusted-host files.pythonhosted.org \
       "deepeval==4.1.0"

# /app is where the guardrail package is mounted; putting it on PYTHONPATH lets
# `guardrail.hook.HallucinationGuardrail` import as a package so its relative
# imports resolve (litellm file-loads the class, which otherwise breaks them).
ENV PYTHONPATH=/app
