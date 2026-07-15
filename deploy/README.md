# Guardrail test stack

Minimal, self-contained LiteLLM proxy that runs the hallucination guardrail,
for validating it end-to-end before the infra team promotes it to production.
Runs on **port 4001** so it won't collide with a proxy already on 4000.

## Run

```bash
cd deploy
cp .env.example .env          # then edit .env and set your real GROQ_API_KEY
docker compose up --build
```

The guardrail package is volume-mounted from `../guardrail`, so iterating on
the code only needs `docker compose restart` (no rebuild). A rebuild is only
needed when the DeepEval dependency in the Dockerfile changes.

## Corporate SSL note (build time)

The Dockerfile installs DeepEval with `--trusted-host pypi.org
--trusted-host files.pythonhosted.org`, which bypasses TLS verification for
PyPI during the build. This is a pragmatic unblock for the corporate
SSL-interception environment where the build image lacks the corporate CA.

For a **hardened production build**, do one of the following instead:

- **Bake the corporate CA into the image** — put the CA `.crt` in the build
  context and, before the pip install:
  ```dockerfile
  COPY corp-ca.crt /usr/local/share/ca-certificates/corp-ca.crt
  RUN update-ca-certificates
  ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
  ```
- **Use the company's internal PyPI mirror** via `--index-url` / `PIP_INDEX_URL`.

## Runtime SSL

Separately, the guardrail's own outbound calls (judge + retries) honour
`GUARDRAIL_SSL_VERIFY` (default `false`, matching the proxy's
`ssl_verify: false`). Flip to `true` once proper CAs are in place.
