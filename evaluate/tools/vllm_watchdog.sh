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
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

DEPLOY_ID="${DEPLOY_ID:-}"
if [ -z "${DEPLOY_ID}" ]; then
    echo "[watchdog] ERROR: 需要设置 DEPLOY_ID（与 deploy_vllm.sh 一致）" >&2
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
            log "已停止看门狗 pid=${wpid}"
        else
            log "看门狗进程不存在 (pid=${wpid:-<空>})"
        fi
        rm -f "${WATCHDOG_PID_FILE}"
    else
        log "没有记录在案的看门狗 (${WATCHDOG_PID_FILE})"
    fi
    exit 0
fi

if [ ! -f "${STATE_FILE}" ]; then
    echo "[watchdog] ERROR: 状态表不存在: ${STATE_FILE}" >&2
    echo "[watchdog]   请确认 deploy_vllm.sh 已成功拉起实例，且 DEPLOY_ID 一致" >&2
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
on_term() { log "收到停止信号，看门狗退出"; running=0; }
trap on_term INT TERM
trap 'rm -f "${WATCHDOG_PID_FILE}"' EXIT

mapfile -t inst_idx < <(awk -F'\t' '$1!="#"{print $1}' "${STATE_FILE}")
if [ "${#inst_idx[@]}" -eq 0 ]; then
    echo "[watchdog] ERROR: 状态表里没有实例记录" >&2
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

log "启动: DEPLOY_ID=${DEPLOY_ID}  实例数=${#inst_idx[@]}"
log "  检查间隔=${INTERVAL}s  超时=${TIMEOUT}s  失败阈值=${FAIL_THRESHOLD}  最多重启=${MAX_RESTARTS}/${RESTART_WINDOW}s"
if [ "${INFER_PROBE}" = "1" ]; then
    log "  推理探测=开  每 ${INFER_EVERY} 轮一次  超时=${INFER_TIMEOUT}s  模型=${INFER_MODEL:-<未知>}"
else
    log "  推理探测=关（仅进程+端口检查）"
fi
log "  状态表=${STATE_FILE}"

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
            log "实例 ${idx} 端口应答正常，但推理探测失败（引擎可能卡死）"
            return 2
        fi
    fi
    return 0
}

wait_instance_ready() {
    local idx="$1" deadline=$((SECONDS + READY_TIMEOUT))
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if check_instance "${idx}" 1; then
            log "实例 ${idx} 已就绪（推理探测通过）"
            return 0
        fi
        local line pid
        line="$(awk -F'\t' -v i="${idx}" '$1==i{print;exit}' "${STATE_FILE}")"
        IFS=$'\t' read -r _ pid _ _ _ <<< "${line}"
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
            log "ERROR: 实例 ${idx} 重启后进程 pid=${pid} 已退出"
            return 1
        fi
        sleep 10
    done
    log "ERROR: 实例 ${idx} 重启后等待就绪超时 (${READY_TIMEOUT}s)"
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
                log "实例 ${idx} 恢复正常（此前端口失败 ${fail_count[$idx]} 次，推理失败 ${infer_fail_count[$idx]} 次）"
            fi
            fail_count["$idx"]=0
            [ "${do_probe}" = "1" ] && infer_fail_count["$idx"]=0
            continue
        fi

        if [ "${rc}" -eq 2 ]; then
            infer_fail_count["$idx"]=$(( infer_fail_count["$idx"] + 1 ))
            log "实例 ${idx} 推理探测失败 (${infer_fail_count[$idx]}/${FAIL_THRESHOLD})"
            [ "${infer_fail_count[$idx]}" -ge "${FAIL_THRESHOLD}" ] || continue
        else
            fail_count["$idx"]=$(( fail_count["$idx"] + 1 ))
            log "实例 ${idx} 健康检查失败 (${fail_count[$idx]}/${FAIL_THRESHOLD})"
            [ "${fail_count[$idx]}" -ge "${FAIL_THRESHOLD}" ] || continue
        fi

        now=$(date +%s)
        if [ $(( now - window_start["$idx"] )) -ge "${RESTART_WINDOW}" ]; then
            restart_count["$idx"]=0
            window_start["$idx"]=${now}
        fi

        if [ "${restart_count[$idx]}" -ge "${MAX_RESTARTS}" ]; then
            log "ERROR: 实例 ${idx} 在 ${RESTART_WINDOW}s 内已重启 ${restart_count[$idx]} 次，放弃自动恢复"
            log "  请人工排查日志后重新部署；看门狗继续监控其它实例"
            fail_count["$idx"]=0
            infer_fail_count["$idx"]=0
            continue
        fi

        restart_count["$idx"]=$(( restart_count["$idx"] + 1 ))
        log "判定实例 ${idx} 已挂，执行第 ${restart_count[$idx]} 次重启"

        if vllm_restart_instance "${idx}"; then
            if wait_instance_ready "${idx}"; then
                log "实例 ${idx} 重启成功"
            else
                log "WARN: 实例 ${idx} 重启后未能就绪，下轮继续观察"
            fi
        else
            log "ERROR: 实例 ${idx} 重启命令执行失败"
        fi
        fail_count["$idx"]=0
        infer_fail_count["$idx"]=0
    done

    [ "${running}" -eq 1 ] || break
    sleep "${INTERVAL}"
done

log "看门狗已退出"
exit 0
