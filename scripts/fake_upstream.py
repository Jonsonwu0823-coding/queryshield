"""An OpenAI-compatible fake upstream: the deterministic Fake model and Fake embedding behind HTTP.

A test double for the OpenAI-compatible path (a model gateway in front of it, or the
Real adapters pointed at it directly); it is not product code and never runs in
``src/``.  Every decision comes from ``FakeModel.complete`` and every vector from the
hashed-feature Fake embedding, so a run through this server matches the in-process
Fake run step for step.

Everything it returns says it is fake: the ``model`` field and the ``X-Fake-Upstream``
header are ``qs-fake-upstream-v1``.  Results that went through it are never real-model
evidence.

Usage rule (fixed, so a gateway can settle it; it does not approximate any tokenizer):
one token per 4 UTF-8 bytes, rounded up, counted for each piece of text separately.
  prompt_tokens      every message's content, plus the compact JSON of ``tools`` when sent
  completion_tokens  the reply text (json protocol), or each function name and its arguments
  total_tokens       prompt_tokens + completion_tokens
Embeddings: prompt_tokens = total_tokens = the tokens of every input.

The ``Authorization`` header is never read; any key, or none, gives the same answer.

    python scripts/fake_upstream.py --host 127.0.0.1 --port 8090
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from queryshield.knowledge.runtime import FAKE_EMBEDDING_DIMENSIONS, feature_vector  # noqa: E402
from queryshield.providers.fake_model import FakeModel  # noqa: E402

FAKE_UPSTREAM_MODEL = "qs-fake-upstream-v1"
FAKE_UPSTREAM_HEADER = "X-Fake-Upstream"
BYTES_PER_TOKEN = 4

app = FastAPI(title="fake upstream", docs_url=None, redoc_url=None, openapi_url=None)
_MODEL = FakeModel()


class _InvalidRequest(Exception):
    def __init__(self, param: str) -> None:
        super().__init__(param)
        self.param = param


def evidence_failures(mode: str, model_names: Iterable[str]) -> list[str]:
    """A run through the fake upstream is never real-model evidence, and a fake-upstream run used nothing else.

    ``model_names`` are the chat model names the run recorded and the embedding model it used;
    a model gateway passes the upstream's model name on, so the mark survives it.
    """

    names = set(model_names)
    if mode == "real" and FAKE_UPSTREAM_MODEL in names:
        return ["fake_upstream_in_real_mode"]
    if mode == "fake-upstream" and names != {FAKE_UPSTREAM_MODEL}:
        return ["model_not_fake_upstream"]
    return []


def tokens(text: str) -> int:
    return -(-len(text.encode("utf-8")) // BYTES_PER_TOKEN)


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@app.middleware("http")
async def _mark_fake(request: Request, call_next):
    response = await call_next(request)
    response.headers[FAKE_UPSTREAM_HEADER] = FAKE_UPSTREAM_MODEL
    response.headers["x-request-id"] = f"fake-req-{uuid4().hex}"
    return response


@app.exception_handler(_InvalidRequest)
async def _invalid_request(request: Request, exc: _InvalidRequest) -> JSONResponse:
    # A fixed message naming only the field; nothing from the request is echoed.
    error = {"message": f"invalid or missing field: {exc.param}", "type": "invalid_request_error", "param": exc.param, "code": "invalid_request"}
    return JSONResponse({"error": error}, status_code=400)


async def _body(request: Request) -> dict:
    try:
        body = json.loads(await request.body())
    except ValueError as exc:
        raise _InvalidRequest("body") from exc
    if not isinstance(body, dict):
        raise _InvalidRequest("body")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _InvalidRequest("model")
    return body


def _messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise _InvalidRequest("messages")
    if not all(isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str) for item in value):
        raise _InvalidRequest("messages")
    return [{"role": item["role"], "content": item["content"]} for item in value]


def _tools(body: dict) -> list[dict] | None:
    if "tools" not in body:
        return None
    tools = body["tools"]
    if not isinstance(tools, list) or not tools:
        raise _InvalidRequest("tools")
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or tool.get("type") != "function" or not isinstance(function.get("name"), str):
            raise _InvalidRequest("tools")
    return tools


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    body = await _body(request)
    messages = _messages(body.get("messages"))
    tools = _tools(body)
    if body.get("stream", False) is not False:
        raise _InvalidRequest("stream")
    result = _MODEL.complete(messages, tools=tools)
    if tools is None:
        message: dict[str, object] = {"role": "assistant", "content": result.content}
        written = [result.content]
        finish_reason = "stop"
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call_{uuid4().hex}", "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in result.tool_calls
            ],
        }
        written = [part for call in result.tool_calls for part in (call.name, call.arguments)]
        finish_reason = "tool_calls"
    prompt = sum(tokens(item["content"]) for item in messages) + (tokens(_compact(tools)) if tools is not None else 0)
    completion = sum(tokens(text) for text in written)
    return {
        "id": f"fake-chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": FAKE_UPSTREAM_MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> dict:
    body = await _body(request)
    raw = body.get("input")
    inputs = [raw] if isinstance(raw, str) else raw
    if not isinstance(inputs, list) or not inputs or not all(isinstance(text, str) and text for text in inputs):
        raise _InvalidRequest("input")
    # The Fake embedding has one length; any other request is refused, never answered with it.
    dimensions = body.get("dimensions", FAKE_EMBEDDING_DIMENSIONS)
    if type(dimensions) is not int or dimensions != FAKE_EMBEDDING_DIMENSIONS:
        raise _InvalidRequest("dimensions")
    used = sum(tokens(text) for text in inputs)
    return {
        "id": f"fake-embd-{uuid4().hex}",
        "object": "list",
        "model": FAKE_UPSTREAM_MODEL,
        "data": [
            {"object": "embedding", "index": index, "embedding": list(feature_vector(text, dimensions=dimensions))}
            for index, text in enumerate(inputs)
        ],
        "usage": {"prompt_tokens": used, "total_tokens": used},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible fake upstream (Fake model and Fake embedding)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
