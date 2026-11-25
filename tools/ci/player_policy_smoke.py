#!/usr/bin/env python3
"""Native end-to-end HTTP/WS smoke test for Tribunal game + player binaries.

Verifies:
1. Ordinary mixed gameplay (2 model seats + 3 explicit scripted opponents)
   completes against the local HTTP /v1/messages fixture, records real model
   calls ONLY for model seats (including a single Markdown JSON fence), and
   writes valid results.json + replay.json with zero fallbacks.
2. Missing LLM credentials for a model seat fails terminally with Coworld's
   strict GamePlayerFailure schema and NO results.json / NO scores.
3. Invalid model reply fails immediately (no retry, no fallback) with
   GamePlayerFailure preserving the raw reply as diagnostic evidence and NO
   results.json / NO scores.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "coworld_manifest_template.json"
FIXTURE_SCRIPT = Path(__file__).resolve().parent / "llm_fixture.py"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_fixture(work: Path, invalid_response: str | None = None) -> tuple[subprocess.Popen, int, Path]:
    port_file = work / "fixture.port"
    log_file = work / "fixture_calls.jsonl"
    cmd = [
        sys.executable,
        str(FIXTURE_SCRIPT),
        "--host", "127.0.0.1",
        "--port", "0",
        "--port-file", str(port_file),
        "--log-file", str(log_file),
    ]
    if invalid_response is not None:
        cmd += ["--invalid-response", invalid_response]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(50):
        if port_file.exists() and port_file.read_text().strip():
            return proc, int(port_file.read_text().strip()), log_file
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("llm_fixture failed to start")


def run_episode(
    game_bin: Path,
    player_bin: Path,
    work: Path,
    llm_endpoint: str | None,
) -> tuple[int, str, Path, Path, Path]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    cert = manifest["certification"]
    cfg = dict(cert["game_config"])
    cfg["tokens"] = [f"tok-{i}" for i in range(5)]
    cfg["player_connect_timeout_seconds"] = 15
    cfg["llmTimeoutSeconds"] = 10
    config_path = work / "config.json"
    results_path = work / "results.json"
    replay_path = work / "replay.json"
    failure_path = work / "player_failure.json"
    config_path.write_text(json.dumps(cfg))

    port = free_port()
    base_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {
            "COWORLD_LLM_ENDPOINT",
            "COWORLD_LLM_MODEL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY_URI",
            "PLAYER_SCRIPTED",
            "PLAYER_PROMPT",
        }
    }
    game_env = dict(base_env)
    game_env.update({
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(port),
        "COGAME_CONFIG_URI": f"file://{config_path}",
        "COGAME_RESULTS_URI": f"file://{results_path}",
        "COGAME_SAVE_REPLAY_URI": f"file://{replay_path}",
        "COGAME_PLAYER_FAILURE_URI": f"file://{failure_path}",
    })
    if llm_endpoint:
        game_env["COWORLD_LLM_ENDPOINT"] = llm_endpoint
        game_env["COWORLD_LLM_MODEL"] = "anthropic/claude-haiku-4.5"

    game_proc = subprocess.Popen(
        [str(game_bin)], env=game_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    time.sleep(0.3)

    by_id = {p["id"]: p for p in manifest["player"]}
    player_procs = []
    for slot, seat_cfg in enumerate(cert["players"]):
        entry = by_id[seat_cfg["player_id"]]
        p_env = dict(base_env)
        p_env["COWORLD_PLAYER_WS_URL"] = f"ws://127.0.0.1:{port}/player?slot={slot}&token=tok-{slot}"
        for k, v in (entry.get("env") or {}).items():
            p_env[k] = str(v)
        player_procs.append(
            subprocess.Popen([str(player_bin)], env=p_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        )

    try:
        out, _ = game_proc.communicate(timeout=30)
    finally:
        for p in player_procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
    return game_proc.returncode, out, results_path, replay_path, failure_path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: player_policy_smoke.py GAME_BIN PLAYER_BIN")
    game_bin = Path(sys.argv[1]).resolve()
    player_bin = Path(sys.argv[2]).resolve()

    # 1. Ordinary mixed episode with local HTTP fixture.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fix_proc, fix_port, log_file = start_fixture(work)
        try:
            rc, out, results_path, replay_path, failure_path = run_episode(
                game_bin, player_bin, work, f"http://127.0.0.1:{fix_port}"
            )
        finally:
            fix_proc.terminate()
            fix_proc.wait(timeout=5)
        assert rc == 0, f"expected exit 0, got {rc}\n{out}"
        assert not failure_path.exists(), f"unexpected player_failure.json: {failure_path.read_text()}"
        results = json.loads(results_path.read_text())
        replay = json.loads(replay_path.read_text())
        assert results["reason"] == "complete"
        assert len(results["scores"]) == 5
        calls = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        assert len(calls) > 0, "expected real HTTP model calls"
        seen_slots = {c["slot"] for c in calls}
        assert seen_slots <= {0, 2} and seen_slots == {0, 2}, f"expected calls only from model slots {{0, 2}}, got {seen_slots}"
        assert any(c["fenced"] for c in calls), "expected at least one single-fenced JSON response"
        model_events = [e for e in replay["events"] if e.get("seat") in (0, 2) and e.get("kind") in ("argue", "whisper", "vote")]
        assert model_events and all(e["scripted"] is False for e in model_events), "model seats must not be marked scripted"
        print(f"mixed episode OK: calls={len(calls)} slots={sorted(seen_slots)} fenced={sum(c['fenced'] for c in calls)} reason={results['reason']}")

    # 2. Missing credentials -> terminal GamePlayerFailure, no results/scores.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rc, out, results_path, replay_path, failure_path = run_episode(game_bin, player_bin, work, None)
        assert rc != 0, f"expected non-zero exit without credentials\n{out}"
        assert not results_path.exists(), "results.json must not be written on policy failure"
        assert not replay_path.exists(), "replay.json must not be written on policy failure"
        failure = json.loads(failure_path.read_text())
        assert set(failure.keys()) == {"failed_policy_index", "message", "kind"}
        assert failure["failed_policy_index"] in (0, 2)
        assert failure["kind"] == "player_error"
        assert "no_credentials" in failure["message"]
        print(f"no-credentials failure OK: {failure}")

    # 3. Invalid model reply -> immediate terminal failure (no retry, no fallback), raw reply preserved.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fix_proc, fix_port, log_file = start_fixture(work, invalid_response='{"lean": "convict"}')
        try:
            rc, out, results_path, replay_path, failure_path = run_episode(
                game_bin, player_bin, work, f"http://127.0.0.1:{fix_port}"
            )
        finally:
            fix_proc.terminate()
            fix_proc.wait(timeout=5)
        assert rc != 0, f"expected non-zero exit on invalid reply\n{out}"
        assert not results_path.exists(), "results.json must not be written on invalid reply"
        assert not replay_path.exists(), "replay.json must not be written on invalid reply"
        failure = json.loads(failure_path.read_text())
        assert set(failure.keys()) == {"failed_policy_index", "message", "kind"}
        assert failure["failed_policy_index"] in (0, 2)
        assert failure["kind"] == "player_error"
        assert "convict" in failure["message"], f"expected raw invalid reply preserved in message: {failure}"
        calls = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        # 2 model seats in round 1 = 1 batch of 2 calls (zero retries!)
        assert len(calls) == 2, f"expected exactly 2 calls (1 batch, zero retries), got {len(calls)}"
        print(f"invalid-reply failure OK: calls={len(calls)} failure={failure}")


if __name__ == "__main__":
    main()
