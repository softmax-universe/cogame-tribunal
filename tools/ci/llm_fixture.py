#!/usr/bin/env python3
"""Deterministic local Anthropic /v1/messages HTTP fixture for Tribunal testing.

Serves valid canonical Tribunal JSON actions (including one single-fenced JSON
reply on the first call to verify Markdown fence unwrapping), records every
request to a JSONL log, and exits cleanly on SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
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
        # Extract up to 2 still-held card ids from YOUR HAND table.
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
    # Return a single Markdown JSON fence on the first call to verify fence unwrapping.
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
    args = parser.parse_args()

    lock = threading.Lock()
    counter = [0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *msg_args: object) -> None:
            pass

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/messages":
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
                if args.invalid_response is not None:
                    reply_text = args.invalid_response
                else:
                    reply_text = decide_for_prompt(user_prompt, idx)
                if args.log_file is not None:
                    record = {
                        "call": idx,
                        "slot": int(slot_header) if slot_header and slot_header.isdigit() else slot_header,
                        "model": model,
                        "fenced": reply_text.startswith("```"),
                        "reply": reply_text,
                    }
                    with args.log_file.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")

            resp = json.dumps({
                "id": f"msg_fixture_{idx}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": reply_text}],
            }).encode("utf-8")
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
