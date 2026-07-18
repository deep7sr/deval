"""Minimal stub of the Langfuse ingestion API (v2, /api/public/ingestion).

Used to verify — without a full Langfuse deployment — exactly what the LiteLLM
proxy's `langfuse` success/failure callback emits for traffic passing through
the gateway. Every received event is appended to captured_events.jsonl.
"""

import base64
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

CAPTURE_FILE = os.environ.get("CAPTURE_FILE", "captured_events.jsonl")

app = FastAPI()


@app.get("/api/public/health")
async def health():
    return {"status": "OK"}


@app.post("/api/public/ingestion")
async def ingestion(request: Request):
    auth = request.headers.get("authorization", "")
    creds = ""
    if auth.startswith("Basic "):
        creds = base64.b64decode(auth[6:]).decode(errors="replace")
    body = await request.json()
    batch = body.get("batch", [])
    with open(CAPTURE_FILE, "a") as f:
        for event in batch:
            f.write(json.dumps({
                "received_at": time.time(),
                "auth_public_key": creds.split(":")[0] if ":" in creds else creds,
                "event": event,
            }) + "\n")
    return JSONResponse(
        status_code=207,
        content={
            "successes": [{"id": e.get("id"), "status": 201} for e in batch],
            "errors": [],
        },
    )
