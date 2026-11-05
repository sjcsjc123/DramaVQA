# vLLM deployment tools

Run these commands from the repository root. Install a vLLM version compatible
with your model and GPU in a separate Python environment. Model weights must
already exist locally; these tools do not download weights.

## Start and stop a service

List the available model presets:

```bash
bash evaluate/tools/deploy_vllm.sh --list
```

Select a preset, local weights, Python environment, and GPU allocation:

```bash
export PYTHON=/path/to/vllm-env/bin/python
export DEPLOY_ID=qwen3-vl-8b
export VLLM_LOG_DIR="$PWD/results/vllm"
export CUDA_VISIBLE_DEVICES=0

bash evaluate/tools/deploy_vllm.sh --model qwen3-vl-8b \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --tp 1 --replicas 1 --port 22002
```

The endpoint is `http://127.0.0.1:22002/v1`, with served model ID
`qwen3-vl-8b`. Use these values for the evaluator's `--base-url` and `--model`.
Set `VLLM_MAX_LEN` to a context length supported by the model and available GPU
memory. Preset GPU counts and extra flags may require adjustment for your
hardware and installed vLLM version. For a custom vLLM source checkout or CUDA
compatibility libraries, set `VLLM_SOURCE_ROOT` or `CUDA_COMPAT_LIB_DIR` explicitly.

Add `--dry-run` to print commands without starting services or writing state.
Multiple replicas use consecutive ports and disjoint GPU groups selected from
`CUDA_VISIBLE_DEVICES`.

Stop the service and its watchdog with the same environment variables:

```bash
bash evaluate/tools/deploy_vllm.sh --stop
```

Keep `DEPLOY_ID` and `VLLM_LOG_DIR` for subsequent stop operations.
Logs, process IDs, and instance state are saved in `VLLM_LOG_DIR`.
Use a different deployment ID for each concurrent deployment.

## Watchdog

`vllm_watchdog.sh` starts automatically after service readiness. It checks
processes and HTTP availability, periodically sends a short inference request,
and restarts an instance after consecutive failures.

Set these variables before deployment to change its behavior:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VLLM_WATCHDOG` | `1` | Set to `0` to disable the watchdog |
| `WATCHDOG_INTERVAL` | `30` | Seconds between checks |
| `WATCHDOG_FAIL_THRESHOLD` | `3` | Consecutive failures before restarting |
| `WATCHDOG_MAX_RESTARTS` | `5` | Restart limit per instance and time window |
| `WATCHDOG_RESTART_WINDOW` | `3600` | Restart accounting window, in seconds |
| `WATCHDOG_INFER_PROBE` | `1` | Set to `0` to disable inference probes |

## Check deployments after GPUs become idle

The optional monitor waits for all expected GPUs to remain idle, then starts,
requests a completion from, and stops each selected preset in sequence.
It requires `nvidia-smi`. Weights are resolved from
`<project-root>/models/<preset model directory>`; alternatively set
`MODELS_ROOT`. This command runs in the foreground:

```bash
python evaluate/tools/monitor_gpu_deploy.py \
  --python /path/to/vllm-env/bin/python \
  --presets qwen3-vl-8b \
  --expected-gpus 1 \
  --state-dir results/deployment-check
```

Set `VLLM_TP_OVERRIDE=1` when checking on one GPU.
Use `--check-only` to check configuration and GPU availability without launching
models. The monitor saves `status.json` and deployment logs under `--state-dir`
and returns a nonzero exit status if any deployment, completion, or cleanup fails.
