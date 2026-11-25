#!/usr/bin/env python3
"""Native end-to-end HTTP/WS smoke test for Tribunal game + player binaries.

Verifies:
1. Ordinary mixed gameplay (2 model seats + 3 explicit scripted opponents)
   completes against the local HTTP /v1/chat/completions fixture, records real
   model calls ONLY for model seats (including a single Markdown JSON fence and
   per-seat COWORLD_LLM_MODEL propagation), and writes valid results.json +
   replay.json with zero fallbacks.
2. Missing LLM credentials for a model seat fails terminally with Coworld's
   strict GamePlayerFailure schema (player_error) and NO results.json / NO scores.
3. Invalid model reply fails immediately (no retry, no fallback) with
   GamePlayerFailure (player_error) preserving the raw reply and NO results.json.
4. Mid-episode backend outage (HTTP 503 after round 1 succeeds) preserves
   genuine HTTP-to-game progress from round 1 and terminates cleanly with
   GamePlayerFailure (episode_inconclusive) and NO results.json / NO default scores.
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


def start_fixture(
    work: Path,
    invalid_response: str | None = None,
    fail_after_calls: int | None = None,
    fail_status: int = 503,
) -> tuple[subprocess.Popen, int, Path]:
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
    if fail_after_calls is not None:
        cmd += ["--fail-after-calls", str(fail_after_calls), "--fail-status", str(fail_status)]
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
    slot_model_overrides: dict[int, str] | None = None,
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
        if slot_model_overrides and slot in slot_model_overrides:
            p_env["COWORLD_LLM_MODEL"] = slot_model_overrides[slot]
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

    # 1. Ordinary mixed episode with per-seat registry model propagation.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fix_proc, fix_port, log_file = start_fixture(work)
        try:
            rc, out, results_path, replay_path, failure_path = run_episode(
                game_bin,
                player_bin,
                work,
                f"http://127.0.0.1:{fix_port}",
                slot_model_overrides={2: "openai/gpt-4o-mini"},
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
        assert all(c["path"] == "/v1/chat/completions" for c in calls), "expected /v1/chat/completions sidecar route"
        seen_slots = {c["slot"] for c in calls}
        assert seen_slots == {0, 2}, f"expected calls only from model slots {{0, 2}}, got {seen_slots}"
        models_by_slot = {c["slot"]: c["model"] for c in calls}
        assert models_by_slot[0] == "anthropic/claude-haiku-4.5", f"unexpected slot 0 model: {models_by_slot}"
        assert models_by_slot[2] == "openai/gpt-4o-mini", f"expected per-seat override for slot 2: {models_by_slot}"
        assert any(c["fenced"] for c in calls), "expected at least one single-fenced JSON response"
        model_events = [e for e in replay["events"] if e.get("seat") in (0, 2) and e.get("kind") in ("argue", "whisper", "vote")]
        assert model_events and all(e["scripted"] is False for e in model_events), "model seats must not be marked scripted"
        print(f"mixed episode OK: calls={len(calls)} slots={sorted(seen_slots)} models={models_by_slot} fenced={sum(c['fenced'] for c in calls)} reason={results['reason']}")

    # 2. Missing credentials -> terminal GamePlayerFailure (player_error), no results/scores.
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
        assert len(calls) == 2, f"expected exactly 2 calls (1 batch, zero retries), got {len(calls)}"
        print(f"invalid-reply failure OK: calls={len(calls)} failure={failure}")

    # 4. Mid-episode backend 503 after round 1 -> partial progress + episode_inconclusive, no results/scores.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fix_proc, fix_port, log_file = start_fixture(work, fail_after_calls=2, fail_status=503)
        try:
            rc, out, results_path, replay_path, failure_path = run_episode(
                game_bin, player_bin, work, f"http://127.0.0.1:{fix_port}"
            )
        finally:
            fix_proc.terminate()
            fix_proc.wait(timeout=5)
        assert rc != 0, f"expected non-zero exit on backend outage\n{out}"
        assert not results_path.exists(), "results.json must not be written on backend outage"
        assert not replay_path.exists(), "replay.json must not be written on backend outage"
        failure = json.loads(failure_path.read_text())
        assert set(failure.keys()) == {"failed_policy_index", "message", "kind"}
        assert failure["failed_policy_index"] in (0, 2)
        assert failure["kind"] == "episode_inconclusive", f"expected episode_inconclusive for 503 outage: {failure}"
        calls = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        ok_calls = [c for c in calls if c["status"] == 200]
        err_calls = [c for c in calls if c["status"] == 503]
        assert len(ok_calls) == 2 and len(err_calls) >= 1, f"expected round 1 ok calls + round 2 503s, got {calls}"
        assert "argument round 2 of 2" in out, f"expected round 1 progress before round 2 failure:\n{out}"
        print(f"partial-progress inconclusive failure OK: ok_calls={len(ok_calls)} err_calls={len(err_calls)} failure={failure}")


if __name__ == "__main__":
    main()
