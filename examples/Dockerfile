# LiteLLM proxy image with MLflow installed for tracing.
# The proxy needs `mlflow>=3.1.4` for the "mlflow" logging callback.
FROM ghcr.io/berriai/litellm:main-stable

RUN pip install --no-cache-dir "mlflow>=3.1.4"
