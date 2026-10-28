#!/usr/bin/env python3
"""POST /v1/systemone in front of the d3-flash MLX loader.

vLLM-SR ships `d3_mlx.py` with a `D3.from_pretrained(...).decide(state, questions)`
API and no HTTP server. The other decision engines answer the TypeSafe wire format, so
this thin adapter makes the MLX arm comparable without touching the model code.

    python3 benchmarks/decision/serve_d3_mlx.py --model sergolsub/mlx-d3-flash-9B --port 8093
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model_store import resolve  # pyright: ignore[reportMissingImports]  # noqa: E402 - repo-local helper

MODEL: Any = None
SERVED_NAME = "d3-flash"


def load_module(model_dir: Path):
    path = model_dir / "d3_mlx.py"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location("d3_mlx", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["d3_mlx"] = module
    spec.loader.exec_module(module)
    return module


class Handler(BaseHTTPRequestHandler):
    server_version = "d3-mlx-systemone"

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/v1/models"):
            self._send(200, {"status": "ok", "object": "list", "data": [{"id": SERVED_NAME}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/v1/systemone", "/systemone"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or "state" not in request:
                raise ValueError("state is required")
            questions = request.get("questions")
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a non-empty object")
            answers = MODEL.decide(request["state"], questions)
            self._send(200, {"model": request.get("model", SERVED_NAME), "answers": answers})
        except Exception as error:  # noqa: BLE001 - report and keep serving
            self._send(400, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base class name
        if not QUIET:
            super().log_message(format, *args)


def main() -> None:
    global MODEL, SERVED_NAME, QUIET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="alias, org/repo, or local dir of an MLX d3-flash checkpoint")
    ap.add_argument("--name", default="d3-flash")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    model_dir = Path(resolve(args.model)).resolve()
    QUIET = args.quiet
    SERVED_NAME = args.name
    module = load_module(model_dir)
    MODEL = module.D3.from_pretrained(str(model_dir))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {SERVED_NAME} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    QUIET = False
    main()
