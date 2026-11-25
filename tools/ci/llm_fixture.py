#!/usr/bin/env python3
"""Deterministic local OpenAI /v1/chat/completions (and /v1/messages) HTTP fixture for Tribunal testing.

Serves valid canonical Tribunal JSON actions (including one single-fenced JSON
reply on the first call to verify Markdown fence unwrapping), records every
request to a JSONL log, supports injected outage/invalid-reply modes, and exits
cleanly on SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def decide_for_prompt(user_prompt: str, call_index: int) -> str:
    if "SEALED BALLOT." in user_prompt:
        payload = {
            "vote": "guilty" if call_index % 2 == 1 else "not_guilty",
            "reason": "Weighing the public record strengths.",
            "notes": "final ballot cast",
        }
    elif "YOUR HAND:" in user_prompt:
        held = re.findall(r"^(E\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*still held\b", user_prompt, re.M)
        picks = held[:2]
        arg = (
            f"Introducing {', '.join(picks)} to support our case."
            if picks
            else "The public record already establishes our position."
        )
        payload = {
            "introduce": picks,
            "argument": arg,
            "notes": f"held remaining: {len(held) - len(picks)}",
        }
    else:
        payload = {
            "whisper": "Comparing shown card strengths and disclosure counts.",
            "lean": "undecided",
            "notes": "tracking record",
        }
    body = json.dumps(payload)
    if call_index == 1:
        return f"```json\n{body}\n```"
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--invalid-response", default=None, help="Return this raw text on every call (for negative tests)")
    parser.add_argument("--fail-after-calls", type=int, default=None, help="Return --fail-status once call count exceeds this")
    parser.add_argument("--fail-status", type=int, default=503, help="HTTP status code when failing")
    args = parser.parse_args()

    lock = threading.Lock()
    counter = [0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *msg_args: object) -> None:
            pass

        def do_POST(self) -> None:
            path = self.path.rstrip("/")
            if path not in ("/v1/chat/completions", "/v1/messages"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
            slot_header = self.headers.get("X-Coworld-Player-Slot")
            model = req.get("model", "")
            user_prompt = ""
            for msg in req.get("messages") or []:
                if msg.get("role") == "user":
                    user_prompt = str(msg.get("content", ""))

            with lock:
                counter[0] += 1
                idx = counter[0]
                should_fail = args.fail_after_calls is not None and idx > args.fail_after_calls
                reply_text = None if should_fail else (
                    args.invalid_response if args.invalid_response is not None else decide_for_prompt(user_prompt, idx)
                )
                if args.log_file is not None:
                    record = {
                        "call": idx,
                        "path": path,
                        "slot": int(slot_header) if slot_header and slot_header.isdigit() else slot_header,
                        "model": model,
                        "status": args.fail_status if should_fail else 200,
                        "fenced": bool(reply_text and reply_text.startswith("```")),
                        "reply": reply_text,
                    }
                    with args.log_file.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")

            if should_fail:
                err_body = json.dumps({"error": {"message": f"injected backend failure ({args.fail_status})"}}).encode("utf-8")
                self.send_response(args.fail_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
                return

            if path == "/v1/chat/completions":
                payload = {
                    "id": f"chatcmpl_fixture_{idx}",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": reply_text},
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                }
            else:
                payload = {
                    "id": f"msg_fixture_{idx}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": reply_text}],
                }
            resp = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    actual_port = server.server_address[1]
    if args.port_file is not None:
        args.port_file.write_text(f"{actual_port}\n", encoding="utf-8")
    print(f"tribunal llm fixture listening on {args.host}:{actual_port}", flush=True)

    def handle_stop(signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
