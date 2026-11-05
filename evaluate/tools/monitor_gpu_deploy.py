#!/usr/bin/env python3
"""Wait for idle GPUs, then clean-start each bundled vLLM preset once.

This checks deployment only. It never changes the deployment script, uses a
unique deployment ID and log directory, and stops only the instances it starts.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler, Request


PYTHON_ENV = {
    "qwen3-vl-8b": "Qwen3-vllm",
    "qwen3-vl-8b-thinking": "Qwen3-vllm",
    "qwen3-vl-32b": "Qwen3-vllm",
    "qwen3-vl-32b-thinking": "Qwen3-vllm",
    "qwen3.5-27b": "GLM53-vllm",
    "qwen3.6-27b": "GLM53-vllm",
    "qwen3.8-27b": "GLM53-vllm",
    "glm-4.6v": "GLM-vllm",
    "glm-5.3-flash": "GLM53-vllm",
}


def gpu_snapshot() -> list[tuple[int, int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=15,
    )
    return [tuple(int(value.strip()) for value in line.split(","))
            for line in output.splitlines() if line.strip()]


def idle(snapshot: list[tuple[int, int]], *, expected: int,
         memory_mib: int, utilization: int) -> bool:
    return len(snapshot) == expected and all(
        used <= memory_mib and busy <= utilization for used, busy in snapshot
    )


def port_free(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def write_status(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def model_request(port: int, model: str, timeout: int) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 16,
        "temperature": 0,
    }).encode("utf-8")
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        body = json.load(response)
    return {"http_ok": True, "response_id": body.get("id"),
            "choices": len(body.get("choices", []))}


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", type=Path, help="vLLM Python interpreter for all selected presets")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--presets", required=True,
                        help="comma-separated vLLM presets to check; select only unresolved cases")
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--max-memory-mib", type=int, default=2048)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--port", type=int, default=22397)
    parser.add_argument("--startup-timeout", type=int, default=7200)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    script = Path(__file__).resolve().parent / "deploy_vllm.sh"
    if not script.is_file():
        parser.error(f"deployment script does not exist: {script}")
    presets = [part.strip() for part in args.presets.split(",") if part.strip()]
    if not presets or len(set(presets)) != len(presets):
        parser.error("--presets must list distinct nonempty model keys")
    listed = set(subprocess.check_output(
        ["bash", str(script), "--list"], text=True, timeout=15,
    ).splitlines())
    for preset in presets:
        if preset not in PYTHON_ENV or preset not in listed:
            parser.error(f"unknown or unavailable preset: {preset}")
        python = args.python or root / "conda" / PYTHON_ENV[preset] / "bin" / "python"
        if not python.is_file():
            parser.error(f"model environment is missing: {python}")
    if (args.expected_gpus < 1 or args.stable_polls < 1 or args.poll_seconds < 1
            or args.startup_timeout < 1 or args.request_timeout < 1):
        parser.error("GPU count, polls and timeouts must be positive")
    snapshot = gpu_snapshot()
    log(f"GPU snapshot (used MiB, utilization %): {snapshot}")
    checks = presets
    if args.check_only:
        log(f"configuration valid; checks={checks}; idle="
            f"{idle(snapshot, expected=args.expected_gpus, memory_mib=args.max_memory_mib, utilization=args.max_utilization)}")
        return 0

    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "monitor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another deployment monitor is already running")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    status_file = state_dir / "status.json"
    status = {"state": "waiting", "checks": checks, "results": {},
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_status(status_file, status)
    log(f"waiting for {args.expected_gpus} idle GPUs; state={status_file}")

    for preset in checks:
        stable = 0
        last_report = 0.0
        while stable < args.stable_polls:
            try:
                snapshot = gpu_snapshot()
                free = idle(snapshot, expected=args.expected_gpus,
                            memory_mib=args.max_memory_mib,
                            utilization=args.max_utilization)
            except Exception as exc:
                snapshot, free = [str(exc)], False
            stable = stable + 1 if free else 0
            if time.monotonic() - last_report >= 3600 or stable == 1:
                log(f"waiting for {preset}: stable={stable}/{args.stable_polls}; GPUs={snapshot}")
                last_report = time.monotonic()
            if stable < args.stable_polls:
                time.sleep(args.poll_seconds)

        if not port_free(args.port):
            status["results"][preset] = {"deployment": "skipped", "reason": "port occupied"}
            write_status(status_file, status)
            log(f"port {args.port} occupied; skipping {preset}")
            continue
        deploy_id = f"deployment_check_{preset}_{int(time.time())}"
        model_python = (args.python or root / "conda" / PYTHON_ENV[preset] / "bin" / "python").resolve()
        model_env = model_python.parent.parent
        env = os.environ.copy()
        env.update({"PYTHON": str(model_python),
                    "CONDA_ENV_PREFIX": str(model_env),
                    "DEPLOY_ID": deploy_id,
                    "VLLM_LOG_DIR": str(state_dir / "vllm_logs"),
                    "DEPLOY_PROJECT_ROOT": str(root),
                    "VLLM_WATCHDOG": "0", "VLLM_REPLICAS_OVERRIDE": "1", "VLLM_PORT": str(args.port),
                    "VLLM_READY_TIMEOUT": str(args.startup_timeout),
                    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
        for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env.pop(proxy, None)
        result = {"deployment": "running", "deploy_id": deploy_id,
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        status["state"] = "testing"
        status["current"] = preset
        status["results"][preset] = result
        write_status(status_file, status)
        log(f"testing {preset} via bundled deployment script")
        deploy_log = state_dir / f"{preset}.deploy.log"
        stop_log = state_dir / f"{preset}.stop.log"
        try:
            with deploy_log.open("w") as output:
                try:
                    completed = subprocess.run(
                        ["bash", str(script), "--model", preset, "--port", str(args.port)],
                        env=env, stdout=output, stderr=subprocess.STDOUT,
                        timeout=args.startup_timeout + 120, check=False,
                    )
                    result["deploy_exit_code"] = completed.returncode
                    result["deployment"] = "ready" if completed.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    result["deployment"] = "timeout"
            if result["deployment"] == "ready":
                try:
                    # The script already checks /v1/models. A real completion
                    # checks that the newly started engine handles requests.
                    with build_opener(ProxyHandler({})).open(
                        f"http://127.0.0.1:{args.port}/v1/models", timeout=15,
                    ) as response:
                        model = json.load(response)["data"][0]["id"]
                    result["served_model"] = model
                    result["completion"] = model_request(args.port, model, args.request_timeout)
                except Exception as exc:
                    result["completion"] = {"http_ok": False, "error": str(exc)}
        finally:
            with stop_log.open("w") as output:
                try:
                    stopped = subprocess.run(
                        ["bash", str(script), "--stop"], env=env, stdout=output,
                        stderr=subprocess.STDOUT, timeout=180, check=False,
                    )
                    result["stop_exit_code"] = stopped.returncode
                except subprocess.TimeoutExpired:
                    result["stop_exit_code"] = "timeout"
            result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            status.pop("current", None)
            status["state"] = "waiting"
            write_status(status_file, status)
            log(f"{preset}: deployment={result['deployment']}; completion={result.get('completion')}; stop={result['stop_exit_code']}")
        if result["stop_exit_code"] != 0:
            status["state"] = "stopped_after_cleanup_failure"
            write_status(status_file, status)
            log("cleanup did not complete; refusing to start another model")
            return 1
    passed = all(
        result.get("deployment") == "ready"
        and result.get("completion", {}).get("http_ok")
        and result.get("completion", {}).get("choices", 0) > 0
        and result.get("stop_exit_code") == 0
        for result in status["results"].values()
    )
    status["state"] = "complete" if passed else "failed"
    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_status(status_file, status)
    log("all selected deployment presets checked")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
