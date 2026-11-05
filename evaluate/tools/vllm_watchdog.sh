#!/usr/bin/env bash
# Monitor vLLM processes, HTTP readiness, and occasional real inference.
# Restart only the failed instance after consecutive failures.
# Usage: DEPLOY_ID=<deployment-id> bash vllm_watchdog.sh
#        DEPLOY_ID=<deployment-id> bash vllm_watchdog.sh --stop
# The deployment script starts this automatically after readiness.
# WATCHDOG_INTERVAL, WATCHDOG_TIMEOUT, WATCHDOG_FAIL_THRESHOLD,
# WATCHDOG_READY_TIMEOUT, WATCHDOG_MAX_RESTARTS, WATCHDOG_RESTART_WINDOW,
# WATCHDOG_INFER_PROBE, WATCHDOG_INFER_EVERY, WATCHDOG_INFER_TIMEOUT,
# and WATCHDOG_INFER_MODEL control the monitoring behavior.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${script_dir}}"
project_root="${DEPLOY_PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}"

VLLM_LOG_DIR="${VLLM_LOG_DIR:-${project_root}/result/logs/vllm}"
INTERVAL="${WATCHDOG_INTERVAL:-30}"
TIMEOUT="${WATCHDOG_TIMEOUT:-10}"
FAIL_THRESHOLD="${WATCHDOG_FAIL_THRESHOLD:-3}"
READY_TIMEOUT="${WATCHDOG_READY_TIMEOUT:-1800}"
MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-5}"
RESTART_WINDOW="${WATCHDOG_RESTART_WINDOW:-3600}"
INFER_PROBE="${WATCHDOG_INFER_PROBE:-1}"
INFER_EVERY="${WATCHDOG_INFER_EVERY:-4}"
INFER_TIMEOUT="${WATCHDOG_INFER_TIMEOUT:-60}"
INFER_MODEL="${WATCHDOG_INFER_MODEL:-}"

do_stop=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop)     do_stop=1; shift ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        -h|--help)  sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DEPLOY_ID="${DEPLOY_ID:-}"
if [ -z "${DEPLOY_ID}" ]; then
    echo "[watchdog] ERROR: Set DEPLOY_ID to the value used by deploy_vllm.sh" >&2
    exit 1
fi

STATE_FILE="${VLLM_LOG_DIR}/vllm_state_${DEPLOY_ID}.tsv"
WATCHDOG_PID_FILE="${VLLM_LOG_DIR}/vllm_watchdog_${DEPLOY_ID}.pid"

log() { echo "[watchdog $(date '+%m-%d %H:%M:%S')] $*"; }

if [ "${do_stop}" -eq 1 ]; then
    if [ -f "${WATCHDOG_PID_FILE}" ]; then
        wpid="$(cat "${WATCHDOG_PID_FILE}" 2>/dev/null)"
        if [ -n "${wpid}" ] && kill -0 "${wpid}" 2>/dev/null; then
            kill -TERM "${wpid}" 2>/dev/null || true
            log "Stopped watchdog pid=${wpid}"
        else
            log "Watchdog process not found (pid=${wpid:-<empty>})"
        fi
        rm -f "${WATCHDOG_PID_FILE}"
    else
        log "No recorded watchdog (${WATCHDOG_PID_FILE})"
    fi
    exit 0
fi

if [ ! -f "${STATE_FILE}" ]; then
    echo "[watchdog] ERROR: State file not found: ${STATE_FILE}" >&2
    echo "[watchdog]   Ensure deploy_vllm.sh started successfully and DEPLOY_ID matches" >&2
    exit 1
fi

# The state header holds the served model name, which matches the preset key.
# This also covers a watchdog started manually without MODEL_KEY in its environment.
MODEL_KEY="${MODEL_KEY:-$(awk -F'\t' '$1=="#"{print $3;exit}' "${STATE_FILE}")}"
DEPLOY_PROJECT_ROOT="${project_root}"
# shellcheck disable=SC1090
source "${PROJECT_ROOT}/deploy_vllm.sh"

echo "${$}" > "${WATCHDOG_PID_FILE}"

running=1
on_term() { log "Stop signal received; shutting down watchdog"; running=0; }
trap on_term INT TERM
trap 'rm -f "${WATCHDOG_PID_FILE}"' EXIT

mapfile -t inst_idx < <(awk -F'\t' '$1!="#"{print $1}' "${STATE_FILE}")
if [ "${#inst_idx[@]}" -eq 0 ]; then
    echo "[watchdog] ERROR: No instance records in state file" >&2
    exit 1
fi

declare -A fail_count infer_fail_count restart_count window_start
for i in "${inst_idx[@]}"; do
    fail_count["$i"]=0
    infer_fail_count["$i"]=0
    restart_count["$i"]=0
    window_start["$i"]=$(date +%s)
done

if [ -z "${INFER_MODEL}" ]; then
    INFER_MODEL="$(awk -F'\t' '$1=="#"{print $3;exit}' "${STATE_FILE}")"
fi

log "Starting: DEPLOY_ID=${DEPLOY_ID}  instances=${#inst_idx[@]}"
log "  check_interval=${INTERVAL}s  timeout=${TIMEOUT}s  failure_threshold=${FAIL_THRESHOLD}  restart_limit=${MAX_RESTARTS}/${RESTART_WINDOW}s"
if [ "${INFER_PROBE}" = "1" ]; then
    log "  inference_probe=enabled  every ${INFER_EVERY} rounds  timeout=${INFER_TIMEOUT}s  model=${INFER_MODEL:-<unknown>}"
else
    log "  inference_probe=disabled (process and HTTP checks only)"
fi
log "  state_file=${STATE_FILE}"

probe_inference() {
    local port="$1" body resp
    [ -n "${INFER_MODEL}" ] || return 0

    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0,"stream":false}' "${INFER_MODEL}")"
    resp="$(curl -s -m "${INFER_TIMEOUT}" \
        -H 'Content-Type: application/json' \
        -d "${body}" \
        "http://localhost:${port}/v1/chat/completions" 2>/dev/null)"

    [ -n "${resp}" ] || return 1
    case "${resp}" in
        *'"choices"'*) return 0 ;;
        *) return 1 ;;
    esac
}

#
check_instance() {
    local idx="$1" probe="${2:-0}" line pid port
    line="$(awk -F'\t' -v i="${idx}" '$1==i{print;exit}' "${STATE_FILE}")"
    [ -n "${line}" ] || return 1
    IFS=$'\t' read -r _ pid _ port _ <<< "${line}"

    [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null || return 1
    curl -sf -m "${TIMEOUT}" "http://localhost:${port}/v1/models" >/dev/null 2>&1 || return 1

    if [ "${probe}" = "1" ] && [ "${INFER_PROBE}" = "1" ]; then
        if ! probe_inference "${port}"; then
            log "Instance ${idx}: HTTP check passed but inference probe failed (engine may be stalled)"
            return 2
        fi
    fi
    return 0
}

wait_instance_ready() {
    local idx="$1" deadline=$((SECONDS + READY_TIMEOUT))
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if check_instance "${idx}" 1; then
            log "Instance ${idx} ready (inference probe passed)"
            return 0
        fi
        local line pid
        line="$(awk -F'\t' -v i="${idx}" '$1==i{print;exit}' "${STATE_FILE}")"
        IFS=$'\t' read -r _ pid _ _ _ <<< "${line}"
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
            log "ERROR: Instance ${idx}: process pid=${pid} exited after restart"
            return 1
        fi
        sleep 10
    done
    log "ERROR: Instance ${idx}: readiness timeout after restart (${READY_TIMEOUT}s)"
    return 1
}

round=0
while [ "${running}" -eq 1 ]; do
    round=$(( round + 1 ))
    do_probe=0
    if [ "${INFER_PROBE}" = "1" ] && [ "${INFER_EVERY}" -gt 0 ] \
        && [ $(( round % INFER_EVERY )) -eq 0 ]; then
        do_probe=1
    fi

    for idx in "${inst_idx[@]}"; do
        [ "${running}" -eq 1 ] || break

        check_instance "${idx}" "${do_probe}"
        rc=$?

        if [ "${rc}" -eq 0 ]; then
            if [ "${fail_count[$idx]}" -gt 0 ] || [ "${infer_fail_count[$idx]}" -gt 0 ]; then
                log "Instance ${idx} recovered (previous HTTP failures: ${fail_count[$idx]}; inference failures: ${infer_fail_count[$idx]})"
            fi
            fail_count["$idx"]=0
            [ "${do_probe}" = "1" ] && infer_fail_count["$idx"]=0
            continue
        fi

        if [ "${rc}" -eq 2 ]; then
            infer_fail_count["$idx"]=$(( infer_fail_count["$idx"] + 1 ))
            log "Instance ${idx}: inference probe failed (${infer_fail_count[$idx]}/${FAIL_THRESHOLD})"
            [ "${infer_fail_count[$idx]}" -ge "${FAIL_THRESHOLD}" ] || continue
        else
            fail_count["$idx"]=$(( fail_count["$idx"] + 1 ))
            log "Instance ${idx}: health check failed (${fail_count[$idx]}/${FAIL_THRESHOLD})"
            [ "${fail_count[$idx]}" -ge "${FAIL_THRESHOLD}" ] || continue
        fi

        now=$(date +%s)
        if [ $(( now - window_start["$idx"] )) -ge "${RESTART_WINDOW}" ]; then
            restart_count["$idx"]=0
            window_start["$idx"]=${now}
        fi

        if [ "${restart_count[$idx]}" -ge "${MAX_RESTARTS}" ]; then
            log "ERROR: Instance ${idx}: restart limit reached (${restart_count[$idx]} restarts in ${RESTART_WINDOW}s); automatic recovery disabled"
            log "  Inspect logs and redeploy manually; continuing to monitor other instances"
            fail_count["$idx"]=0
            infer_fail_count["$idx"]=0
            continue
        fi

        restart_count["$idx"]=$(( restart_count["$idx"] + 1 ))
        log "Instance ${idx} unhealthy; starting restart ${restart_count[$idx]}"

        if vllm_restart_instance "${idx}"; then
            if wait_instance_ready "${idx}"; then
                log "Instance ${idx} restarted successfully"
            else
                log "WARN: Instance ${idx} not ready after restart; checking again next round"
            fi
        else
            log "ERROR: Instance ${idx}: restart command failed"
        fi
        fail_count["$idx"]=0
        infer_fail_count["$idx"]=0
    done

    [ "${running}" -eq 1 ] || break
    sleep "${INTERVAL}"
done

log "Watchdog exited"
exit 0
