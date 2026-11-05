#!/usr/bin/env bash
# Deploy one or more vLLM instances using the named model presets.
# Usage: bash deploy_vllm.sh --list
#        bash deploy_vllm.sh --model qwen3-vl-32b [--port 22002] [--dry-run]
#        DEPLOY_ID=<deployment-id> bash deploy_vllm.sh --stop
# The watchdog starts after all instances become ready unless VLLM_WATCHDOG=0.
# Use --model-path PATH and --python /path/to/python to select local resources.
# Options: --tp N --replicas N --gpu-util FRACTION --port-mode increment|same
# MODELS_ROOT, VLLM_LOG_DIR, PYTHON, and CONDA_ENV_PREFIX are overridable.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${DEPLOY_PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}"

MODELS_ROOT="${MODELS_ROOT:-${project_root}/models}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-${project_root}/result/logs/vllm}"

#
declare -A MODEL_PRESETS=(
    ["qwen3-vl-32b"]="Qwen3-VL-32B-Instruct|qwen3-vl-32b|8|0.75|1|--enforce-eager"
    ["qwen3-vl-32b-thinking"]="Qwen3-VL-32B-Thinking|qwen3-vl-32b-thinking|8|0.75|1|--mm-encoder-tp-mode data --async-scheduling --media-io-kwargs {\"video\":{\"num_frames\":-1}}"
    ["qwen3.5-27b"]="Qwen3.5-27B|qwen3.5-27b|8|0.75|1|--trust-remote-code --enforce-eager"
    ["qwen3.6-27b"]="Qwen3.6-27B|qwen3.6-27b|8|0.75|1|--trust-remote-code --enforce-eager"
    ["qwen3.8-27b"]="Qwen3.8-27B|qwen3.8-27b|8|0.75|1|--trust-remote-code --enforce-eager"
    ["qwen3-vl-8b"]="Qwen3-VL-8B-Instruct|qwen3-vl-8b|4|0.9|2|--trust-remote-code --enforce-eager"
    ["qwen3-vl-8b-thinking"]="Qwen3-VL-8B-Thinking|qwen3-vl-8b-thinking|4|0.9|2|--trust-remote-code --enforce-eager"
    ["glm-4.6v"]="GLM-4.6V|glm-4.6v|8|0.75|1|--reasoning-parser glm45 --mm-encoder-tp-mode data --async-scheduling --media-io-kwargs {\"video\":{\"num_frames\":-1}}"
    #       GLM53_REASONING_PARSER=deepseek_r1 bash deploy_vllm.sh -m glm-5.3-flash
    ["glm-5.3-flash"]="GLM-5.3-Flash|glm-5.3-flash|8|0.85|1|--reasoning-parser ${GLM53_REASONING_PARSER:-glm45} --no-enable-flashinfer-autotune --disable-custom-all-reduce --enforce-eager --kernel-config {\"enable_jit_warmup\":false,\"enable_cutedsl_warmup\":false} --linear-backend triton --moe-backend triton ${GLM53_ATTENTION_BACKEND:+--attention-backend ${GLM53_ATTENTION_BACKEND}}"
)

MODEL_KEY="${MODEL_KEY:-${MODEL_NAME:-qwen3-vl-32b}}"
BASE_PORT="${VLLM_PORT:-22002}"
VLLM_PORT_MODE="${VLLM_PORT_MODE:-increment}"
MAX_LEN="${VLLM_MAX_LEN:-128000}"
MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
SWAP_SPACE="${VLLM_SWAP_SPACE:-8}"
READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
dry_run=0
do_stop=0
stop_all=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)   MODEL_KEY="$2"; shift 2 ;;
        -p|--port)    BASE_PORT="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; MODEL_PATH_EXPLICIT=1; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --port-mode)  VLLM_PORT_MODE="$2"; shift 2 ;;
        --tp)         VLLM_TP_OVERRIDE="$2"; shift 2 ;;
        --replicas)   VLLM_REPLICAS_OVERRIDE="$2"; shift 2 ;;
        --gpu-util)   VLLM_GPU_UTIL_OVERRIDE="$2"; shift 2 ;;
        --stop)       do_stop=1; shift ;;
        --all)        stop_all=1; shift ;;
        --dry-run)    dry_run=1; shift ;;
        --list)       printf '%s\n' "${!MODEL_PRESETS[@]}" | sort; exit 0 ;;
        -h|--help)    sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DEPLOY_ID="${DEPLOY_ID:-${MODEL_KEY}_$$}"
PID_FILE="${VLLM_LOG_DIR}/vllm_pids_${DEPLOY_ID}.txt"
STATE_FILE="${VLLM_LOG_DIR}/vllm_state_${DEPLOY_ID}.tsv"
WATCHDOG_PID_FILE="${VLLM_LOG_DIR}/vllm_watchdog_${DEPLOY_ID}.pid"
log() { echo "[deploy-vllm] $*"; }

if ! declare -F conda_nvidia_ld_prefix >/dev/null 2>&1; then
    # Prefer CUDA libraries installed in the selected conda environment.
    conda_nvidia_ld_prefix() {
        local env_root="${1:-${CONDA_ENV_PREFIX:-}}"
        local acc="" d root
        [ -n "${env_root}" ] || { echo ""; return 0; }
        for root in "${env_root}"/lib/python3.*/site-packages/nvidia; do
            [ -d "${root}" ] || continue
            for d in "${root}/nvjitlink/lib" "${root}/cuda_runtime/lib" "${root}"/*/lib; do
                [ -d "${d}" ] || continue
                case ":${acc}:" in *":${d}:"*) continue ;; esac
                acc="${acc}${d}:"
            done
        done
        printf '%s' "${acc}"
    }
fi

resolve_vllm_cmd() {
    if [ -z "${PYTHON:-}" ]; then
        for _py in "${project_root}/conda/Qwen3-vllm/bin/python" \
                   "$(command -v python 2>/dev/null)"; do
            [ -x "${_py:-}" ] || continue
            PYTHON="$_py"; break
        done
    fi
    [ -n "${PYTHON:-}" ] || { log "ERROR: Python interpreter not found"; return 1; }

    VLLM_CMD=("${PYTHON}" -m vllm.entrypoints.cli.main)
    log "Launch command: ${PYTHON} -m vllm.entrypoints.cli.main"
    return 0
}

vllm_watchdog_start() {
    if [ "${VLLM_WATCHDOG:-1}" != "1" ]; then
        log "Watchdog disabled (VLLM_WATCHDOG=0)"
        return 0
    fi
    [ "${dry_run}" -eq 1 ] && return 0

    local wd_sh="${script_dir}/vllm_watchdog.sh"
    if [ ! -f "${wd_sh}" ]; then
        log "WARN: Watchdog script not found: ${wd_sh}; skipping"
        return 0
    fi

    local wd_log="${VLLM_LOG_DIR}/watchdog_${DEPLOY_ID}.log"
    DEPLOY_ID="${DEPLOY_ID}" VLLM_LOG_DIR="${VLLM_LOG_DIR}" PROJECT_ROOT="${script_dir}" \
        DEPLOY_PROJECT_ROOT="${project_root}" MODELS_ROOT="${MODELS_ROOT}" \
        MODEL_KEY="${MODEL_KEY}" PYTHON="${PYTHON:-}" CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}" \
        setsid nohup bash "${wd_sh}" >>"${wd_log}" 2>&1 &
    disown 2>/dev/null || true
    log "Watchdog started; log: ${wd_log}"
    log "  Stop: DEPLOY_ID=${DEPLOY_ID} bash ${wd_sh} --stop"
    return 0
}

vllm_watchdog_stop() {
    local pid_file="${1:-${WATCHDOG_PID_FILE}}"
    [ -f "${pid_file}" ] || return 0
    local wpid
    wpid="$(cat "${pid_file}" 2>/dev/null)"
    if [ -n "${wpid}" ] && kill -0 "${wpid}" 2>/dev/null; then
        log "Stopping watchdog pid=${wpid}"
        kill -TERM "${wpid}" 2>/dev/null || true
        local w
        for w in $(seq 1 15); do
            kill -0 "${wpid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${wpid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
}

_stop_pid_file() {
    local pid_file="$1" pid
    [ -f "${pid_file}" ] || return 0
    while read -r pid; do
        [ -n "${pid}" ] || continue
        if kill -0 "${pid}" 2>/dev/null; then
            log "Stopping pid=${pid}"
            kill -TERM "${pid}" 2>/dev/null || true
            for _ in $(seq 1 30); do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 1
            done
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done < "${pid_file}"
    rm -f "${pid_file}"
}

vllm_stop() {
    vllm_watchdog_stop
    if [ ! -f "${PID_FILE}" ]; then
        log "No recorded instances (${PID_FILE})"
        log "Use --stop --all, or DEPLOY_ID=<id> --stop for a specific deployment"
        return 0
    fi
    _stop_pid_file "${PID_FILE}"
    rm -f "${STATE_FILE}"
    log "Deployment instances stopped"
}

vllm_stop_all() {
    local f found=0 did
    for f in "${VLLM_LOG_DIR}"/vllm_watchdog_*.pid; do
        [ -e "${f}" ] || continue
        vllm_watchdog_stop "${f}"
    done
    for f in "${VLLM_LOG_DIR}"/vllm_pids_*.txt; do
        [ -e "${f}" ] || continue
        found=1
        log "Removing ${f}"
        _stop_pid_file "${f}"
        did="$(basename "${f}")"; did="${did#vllm_pids_}"; did="${did%.txt}"
        rm -f "${VLLM_LOG_DIR}/vllm_state_${did}.tsv"
    done
    [ "${found}" -eq 0 ] && log "No PID records found"
    return 0
}

build_cmd() {
    local model_dir="$1" served_name="$2" tp="$3" gpu_util="$4" port="$5" extra="$6"
    CMD=("${VLLM_CMD[@]}" serve "${model_dir}"
        --served-model-name "${served_name}"
        --tensor-parallel-size "${tp}"
        --gpu-memory-utilization "${gpu_util}"
        --enable-prefix-caching
        --host 0.0.0.0
        --port "${port}"
        --max-model-len "${MAX_LEN}"
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
        --max-num-seqs "${MAX_NUM_SEQS}")
    case "${MODEL_KEY:-}" in
        *glm-5.3*|*glm5.3*|*glm53*) ;;
        *) CMD+=(--swap-space "${SWAP_SPACE}") ;;
    esac
    if [ -n "${extra}" ]; then
        # shellcheck disable=SC2206
        local parts=(${extra})
        CMD+=("${parts[@]}")
    fi
}

vllm_deploy() {
    local preset="${MODEL_PRESETS[${MODEL_KEY}]:-}"
    if [ -z "${preset}" ]; then
        log "ERROR: Unknown model preset '${MODEL_KEY}'; available presets:"
        printf '  %s\n' "${!MODEL_PRESETS[@]}" | sort >&2
        return 1
    fi

    IFS='|' read -r dir_name served_name tp gpu_util replicas extra <<< "${preset}"

    if [ "${VLLM_TP_EXPLICIT:-}" = "1" ]; then tp="${VLLM_TP:-${tp}}"; fi
    if [ "${VLLM_GPU_UTIL_EXPLICIT:-}" = "1" ]; then gpu_util="${VLLM_GPU_UTIL:-${gpu_util}}"; fi
    tp="${VLLM_TP_OVERRIDE:-${tp}}"
    gpu_util="${VLLM_GPU_UTIL_OVERRIDE:-${gpu_util}}"
    replicas="${VLLM_REPLICAS_OVERRIDE:-${VLLM_REPLICAS:-${replicas}}}"
    # Explicit override may be empty to reproduce a recorded non-eager launch.
    if [ "${VLLM_PRESET_EXTRA_OVERRIDE+x}" = x ]; then
        extra="${VLLM_PRESET_EXTRA_OVERRIDE}"
    fi

    local model_dir="${MODELS_ROOT}/${dir_name}"
    if [ "${MODEL_PATH_EXPLICIT:-}" = "1" ]; then model_dir="${MODEL_PATH:-${model_dir}}"; fi
    if [ ! -d "${model_dir}" ]; then
        log "ERROR: Model directory not found: ${model_dir}"
        return 1
    fi

    resolve_vllm_cmd || return 1
    [[ "${tp}" =~ ^[1-9][0-9]*$ && "${replicas}" =~ ^[1-9][0-9]*$ && "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || {
        log "ERROR: TP, replicas and port must be positive integers"; return 1;
    }
    if [[ "${VLLM_PORT_MODE}" != increment && "${VLLM_PORT_MODE}" != same ]] ||
       { [[ "${VLLM_PORT_MODE}" == same ]] && (( replicas > 1 )); } ||
       (( BASE_PORT + replicas - 1 > 65535 )); then
        log "ERROR: Invalid port range or shared port for multiple replicas"; return 1
    fi
    if [[ -e "${PID_FILE}" || -e "${STATE_FILE}" ]]; then
        log "ERROR: Deployment ID already exists; stop it or choose a new DEPLOY_ID"; return 1
    fi
    if [[ "${dry_run}" -eq 0 ]]; then mkdir -p "${VLLM_LOG_DIR}"; fi

    local total_gpus need
    total_gpus=0
    if [[ "${dry_run}" -eq 0 ]]; then total_gpus="$("${PYTHON}" -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"; fi
    need=$((tp * replicas))
    if [ "${total_gpus}" -gt 0 ] && [ "${need}" -gt "${total_gpus}" ]; then
        log "ERROR: Requested ${need} GPUs (TP=${tp} x ${replicas} instances), but only ${total_gpus} are available"
        return 1
    fi

    log "Model     : ${MODEL_KEY}  ->  ${model_dir}"
    log "Served ID : ${served_name}"
    log "Topology  : TP=${tp} x ${replicas} instances (${need} GPUs), memory utilization ${gpu_util}"
    if [ "${VLLM_PORT_MODE}" = "increment" ] && [ "${replicas}" -gt 1 ]; then
        log "Ports     : ${BASE_PORT} ~ $((BASE_PORT + replicas - 1)) (increment)"
    else
        log "Ports     : ${BASE_PORT}$([ "${replicas}" -gt 1 ] && echo " (${replicas} instances sharing this port)")"
    fi
    [ -n "${extra}" ] && log "Extra args: ${extra}"

    if [[ "${dry_run}" -eq 0 ]]; then
        : > "${PID_FILE}"
        printf '#\t%s\t%s\t%s\t%s\t%s\n' \
            "${model_dir}" "${served_name}" "${tp}" "${gpu_util}" "${extra}" > "${STATE_FILE}"
    fi
    DEPLOYED_PORTS=()
    DEPLOYED_LOGS=()

    local i port gpus log_file
    for ((i = 0; i < replicas; i++)); do
        if [ "${VLLM_PORT_MODE}" = "increment" ]; then
            port=$((BASE_PORT + i))
        else
            port="${BASE_PORT}"
        fi
        if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
            local visible_devices
            IFS=, read -ra visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
            if (( ${#visible_devices[@]} < need )); then
                log "ERROR: CUDA_VISIBLE_DEVICES contains fewer devices than requested"; return 1
            fi
            gpus="$(IFS=,; echo "${visible_devices[*]:i*tp:tp}")"
        else
            gpus="$(seq -s, $((i * tp)) $(((i + 1) * tp - 1)))"
        fi
        log_file="${VLLM_LOG_DIR}/vllm_${DEPLOY_ID}_i${i}_p${port}.log"
        DEPLOYED_LOGS+=("${log_file}")

        build_cmd "${model_dir}" "${served_name}" "${tp}" "${gpu_util}" "${port}" "${extra}"

        if [ "${dry_run}" -eq 1 ]; then
            echo "[dry-run] CUDA_VISIBLE_DEVICES=${gpus} ${CMD[*]}"
            DEPLOYED_PORTS+=("${port}")
            continue
        fi

        log ">>> Instance $((i + 1))/${replicas}: GPU=${gpus} port=${port} log=${log_file}"
        local _launch_ld
        _launch_ld="$(conda_nvidia_ld_prefix "${CONDA_ENV_PREFIX:-}" 2>/dev/null || true)"
        _launch_ld="${_launch_ld}${CONDA_ENV_PREFIX:+${CONDA_ENV_PREFIX}/lib:}${LD_LIBRARY_PATH:-}"
        local _launch_pypath="${PYTHONPATH:-}"
        case "${MODEL_KEY:-}" in
            *glm-5.3*|*glm5.3*|*glm53*)
                local _cu13="${CONDA_ENV_PREFIX:-}/lib/python3.10/site-packages/nvidia/cu13/lib"
                [ -d "${_cu13}" ] && _launch_ld="${_cu13}:${_launch_ld}"
                local _compat=""
                if [ -n "${CUDA_COMPAT_LIB_DIR:-}" ]; then
                _compat="${CUDA_COMPAT_LIB_DIR}"
            fi
                [ -n "${_compat}" ] && _launch_ld="${_compat}:${_launch_ld}"
                local _vllm_src_dir
                _vllm_src_dir="${VLLM_SOURCE_ROOT:-}"
                if [ -n "${_vllm_src_dir}" ]; then _launch_pypath="${_vllm_src_dir}${PYTHONPATH:+:${PYTHONPATH}}"; fi
                export VLLM_USE_DEEP_GEMM=0
                export VLLM_MOE_USE_DEEP_GEMM=0
                export VLLM_ALLREDUCE_USE_FLASHINFER=0
                log "GLM53 PYTHONPATH=${_vllm_src_dir} (DeepGEMM=0, FI-AR=0)"
                ;;
        esac
        if [ -n "${_launch_pypath}" ]; then
            CUDA_VISIBLE_DEVICES="${gpus}" LD_LIBRARY_PATH="${_launch_ld}" PYTHONPATH="${_launch_pypath}" PYTHONUNBUFFERED=1 env -u VLLM_PORT "${CMD[@]}" >>"${log_file}" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="${gpus}" LD_LIBRARY_PATH="${_launch_ld}" PYTHONUNBUFFERED=1 env -u VLLM_PORT "${CMD[@]}" >>"${log_file}" 2>&1 &
        fi
        echo "$!" >> "${PID_FILE}"
        printf '%s\t%s\t%s\t%s\t%s\n' "${i}" "$!" "${gpus}" "${port}" "${log_file}" >> "${STATE_FILE}"
        DEPLOYED_PORTS+=("${port}")
    done

    local uniq_ports
    uniq_ports="$(printf '%s\n' "${DEPLOYED_PORTS[@]}" | awk '!seen[$0]++' | paste -sd, -)"
    export VLLM_PORTS="${uniq_ports}"
    export VLLM_SERVED_MODEL_NAME="${served_name}"
    log "Instances started; ports: ${VLLM_PORTS}"
    return 0
}

vllm_restart_instance() {
    local idx="$1"
    [ -f "${STATE_FILE}" ] || { log "ERROR: State file not found: ${STATE_FILE}"; return 1; }

    local model_dir served_name tp gpu_util extra
    model_dir="$(awk -F'\t' '$1=="#"{print $2;exit}' "${STATE_FILE}")"
    served_name="$(awk -F'\t' '$1=="#"{print $3;exit}' "${STATE_FILE}")"
    tp="$(awk -F'\t' '$1=="#"{print $4;exit}' "${STATE_FILE}")"
    gpu_util="$(awk -F'\t' '$1=="#"{print $5;exit}' "${STATE_FILE}")"
    extra="$(awk -F'\t' '$1=="#"{print $6;exit}' "${STATE_FILE}")"
    [ -n "${model_dir}" ] || { log "ERROR: State file is missing its header"; return 1; }

    local line old_pid gpus port log_file
    line="$(awk -F'\t' -v i="${idx}" '$1==i{print;exit}' "${STATE_FILE}")"
    [ -n "${line}" ] || { log "ERROR: No state record for instance ${idx}"; return 1; }
    old_pid="$(awk -F'\t' -v i="${idx}" '$1==i{print $2;exit}' "${STATE_FILE}")"
    gpus="$(awk -F'\t' -v i="${idx}" '$1==i{print $3;exit}' "${STATE_FILE}")"
    port="$(awk -F'\t' -v i="${idx}" '$1==i{print $4;exit}' "${STATE_FILE}")"
    log_file="$(awk -F'\t' -v i="${idx}" '$1==i{print $5;exit}' "${STATE_FILE}")"

    if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
        log "Terminating previous process pid=${old_pid} for instance ${idx}"
        kill -TERM "${old_pid}" 2>/dev/null || true
        local w
        for w in $(seq 1 30); do
            kill -0 "${old_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${old_pid}" 2>/dev/null || true
        sleep 5
    fi

    resolve_vllm_cmd || return 1
    build_cmd "${model_dir}" "${served_name}" "${tp}" "${gpu_util}" "${port}" "${extra}"

    log "Restarting instance ${idx}: GPU=${gpus} port=${port} log=${log_file}"
    local _restart_ld
    _restart_ld="$(conda_nvidia_ld_prefix "${CONDA_ENV_PREFIX:-}" 2>/dev/null || true)"
    _restart_ld="${_restart_ld}${CONDA_ENV_PREFIX:+${CONDA_ENV_PREFIX}/lib:}${LD_LIBRARY_PATH:-}"
    local _restart_pypath="${PYTHONPATH:-}"
    case "${MODEL_KEY:-}" in
        *glm-5.3*|*glm5.3*|*glm53*)
            local _cu13="${CONDA_ENV_PREFIX:-}/lib/python3.10/site-packages/nvidia/cu13/lib"
            [ -d "${_cu13}" ] && _restart_ld="${_cu13}:${_restart_ld}"
            local _compat=""
            if [ -n "${CUDA_COMPAT_LIB_DIR:-}" ]; then
                _compat="${CUDA_COMPAT_LIB_DIR}"
            fi
            [ -n "${_compat}" ] && _restart_ld="${_compat}:${_restart_ld}"
            local _vllm_src_dir
            _vllm_src_dir="${VLLM_SOURCE_ROOT:-}"
            if [ -n "${_vllm_src_dir}" ]; then _restart_pypath="${_vllm_src_dir}${PYTHONPATH:+:${PYTHONPATH}}"; fi
            export VLLM_USE_DEEP_GEMM=0
            export VLLM_MOE_USE_DEEP_GEMM=0
            export VLLM_ALLREDUCE_USE_FLASHINFER=0
            ;;
    esac
    if [ -n "${gpus}" ]; then
        if [ -n "${_restart_pypath}" ]; then
            CUDA_VISIBLE_DEVICES="${gpus}" LD_LIBRARY_PATH="${_restart_ld}" PYTHONPATH="${_restart_pypath}" PYTHONUNBUFFERED=1 env -u VLLM_PORT "${CMD[@]}" >>"${log_file}" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="${gpus}" LD_LIBRARY_PATH="${_restart_ld}" PYTHONUNBUFFERED=1 env -u VLLM_PORT "${CMD[@]}" >>"${log_file}" 2>&1 &
        fi
    else
        if [ -n "${_restart_pypath}" ]; then
            LD_LIBRARY_PATH="${_restart_ld}" PYTHONPATH="${_restart_pypath}" PYTHONUNBUFFERED=1 env -u VLLM_PORT "${CMD[@]}" >>"${log_file}" 2>&1 &
        else
            LD_LIBRARY_PATH="${_restart_ld}" PYTHONUNBUFFERED=1 env -u VLLM_PORT "${CMD[@]}" >>"${log_file}" 2>&1 &
        fi
    fi
    local new_pid=$!
    log "Instance ${idx} new pid=${new_pid}"

    local tmp="${STATE_FILE}.tmp.$$"
    awk -F'\t' -v OFS='\t' -v i="${idx}" -v p="${new_pid}" \
        '$1==i{$2=p} {print}' "${STATE_FILE}" > "${tmp}" && mv "${tmp}" "${STATE_FILE}"
    awk -F'\t' '$1!="#"{print $2}' "${STATE_FILE}" > "${PID_FILE}"
    return 0
}


vllm_wait_ready() {
    [ "${dry_run}" -eq 1 ] && return 0
    DEPLOYED_LOGS=("${DEPLOYED_LOGS[@]:-}")
    local deadline=$((SECONDS + READY_TIMEOUT))
    local port ready_all pid_list
    mapfile -t pid_list < "${PID_FILE}" 2>/dev/null || pid_list=()

    while [ "${SECONDS}" -lt "${deadline}" ]; do
        ready_all=1
        for port in ${VLLM_PORTS//,/ }; do
            curl -sf -m 5 "http://localhost:${port}/v1/models" >/dev/null 2>&1 || ready_all=0
        done
        if [ "${ready_all}" -eq 1 ]; then
            log "All instances ready (elapsed approximately ${SECONDS}s; ports ${VLLM_PORTS})"
            return 0
        fi
        local pid
        for pid in "${pid_list[@]}"; do
            [ -n "${pid}" ] || continue
            if ! kill -0 "${pid}" 2>/dev/null; then
                log "ERROR: Instance pid=${pid} exited; log tail:"
                tail -n 40 "${DEPLOYED_LOGS[@]}" >&2 || true
                return 1
            fi
        done
        sleep 5
    done
    log "ERROR: Readiness timeout (${READY_TIMEOUT}s)"
    tail -n 40 "${DEPLOYED_LOGS[@]}" >&2 || true
    return 1
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    if [ "${do_stop}" -eq 1 ]; then
        if [ "${stop_all}" -eq 1 ]; then
            vllm_stop_all
        else
            vllm_stop
        fi
        exit 0
    fi
    vllm_deploy || exit 1
    vllm_wait_ready || { vllm_stop; exit 1; }
    vllm_watchdog_start
    [[ "${dry_run}" -eq 1 ]] || log "Stop: DEPLOY_ID=${DEPLOY_ID} VLLM_LOG_DIR=${VLLM_LOG_DIR} bash ${BASH_SOURCE[0]} --stop"
fi
