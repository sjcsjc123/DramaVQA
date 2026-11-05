"""The background deployment monitor must not treat a partly busy host as idle."""

from pathlib import Path
import runpy


def test_deployment_monitor_waits_for_every_gpu():
    script = Path(__file__).parents[1] / "tools" / "monitor_gpu_deploy.py"
    idle = runpy.run_path(str(script))["idle"]
    clean = [(512, 0)] * 8
    assert idle(clean, expected=8, memory_mib=2048, utilization=10)
    assert not idle(clean[:7], expected=8, memory_mib=2048, utilization=10)
    assert not idle(clean[:7] + [(3000, 0)], expected=8,
                    memory_mib=2048, utilization=10)
    assert not idle(clean[:7] + [(512, 50)], expected=8,
                    memory_mib=2048, utilization=10)


def test_deployment_dry_run_preserves_gpu_mapping_and_writes_no_state(tmp_path):
    import os
    import subprocess
    script = Path(__file__).parents[1] / "tools" / "deploy_vllm.sh"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "3,7",
           "VLLM_LOG_DIR": str(tmp_path / "logs"), "DEPLOY_ID": "test",
           "PYTHON": "/unused/python"}
    result = subprocess.run(
        ["bash", str(script), "--model", "qwen3-vl-8b",
         "--model-path", str(tmp_path), "--tp", "1", "--replicas", "2",
         "--dry-run"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA_VISIBLE_DEVICES=3 " in result.stdout
    assert "CUDA_VISIBLE_DEVICES=7 " in result.stdout
    assert "--port 22002" in result.stdout
    assert "--port 22003" in result.stdout
    assert not (tmp_path / "logs").exists()


def test_deployment_rejects_shared_ports(tmp_path):
    import os
    import subprocess
    script = Path(__file__).parents[1] / "tools" / "deploy_vllm.sh"
    result = subprocess.run(
        ["bash", str(script), "--model-path", str(tmp_path), "--tp", "1",
         "--replicas", "2", "--port-mode", "same", "--dry-run"],
        env={**os.environ, "VLLM_LOG_DIR": str(tmp_path / "logs")},
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert not (tmp_path / "logs").exists()


def test_deployment_does_not_overwrite_existing_state(tmp_path):
    import os
    import subprocess
    script = Path(__file__).parents[1] / "tools" / "deploy_vllm.sh"
    state = tmp_path / "vllm_state_test.tsv"
    state.write_text("existing deployment")
    result = subprocess.run(
        ["bash", str(script), "--model-path", str(tmp_path), "--dry-run"],
        env={**os.environ, "VLLM_LOG_DIR": str(tmp_path), "DEPLOY_ID": "test"},
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert state.read_text() == "existing deployment"


def test_monitor_reports_occupied_port_as_failure(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    script = Path(__file__).parents[1] / "tools" / "monitor_gpu_deploy.py"
    spec = importlib.util.spec_from_file_location("deployment_monitor_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "gpu_snapshot", lambda: [(0, 0)])
    monkeypatch.setattr(module, "port_free", lambda _: False)
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda *a, **k: "qwen3-vl-8b\n")
    monkeypatch.setattr(module.signal, "signal", lambda *a: None)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--python", sys.executable, "--presets", "qwen3-vl-8b",
        "--state-dir", str(tmp_path), "--expected-gpus", "1",
        "--stable-polls", "1"])
    assert module.main() == 1
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"
