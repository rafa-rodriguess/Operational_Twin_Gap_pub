#!/usr/bin/env bash
# Poll the private repo every 10s. Wake the local execution LLM only when
# there is a job, a queue conflict, a recovery case, invalid queue files, or a git error.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INTERVAL="${EXECUTOR_POLL_SECONDS:-10}"
LOG="$ROOT/outputs/logs/executor_loop.log"
PIDFILE="$ROOT/outputs/logs/executor_loop.pid"
LAST="$ROOT/outputs/logs/executor_loop.last_wake"
POLL=(python3 "$ROOT/scripts/executor_poll.py")
CLAIM=(python3 "$ROOT/scripts/executor_poll.py" --claim --push --no-pull)
DISCORD_CFG="$ROOT/config/discord_executor.yaml"

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

mkdir -p "$ROOT/outputs/logs" \
         "$ROOT/prompts/prompts_running" \
         "$ROOT/prompts/prompts_executed" \
         "$ROOT/prompts/prompts_answers" \
         "$ROOT/prompts/prompts_superseded"

payload_field() {
    python3 -c 'import json,sys
d=json.loads(sys.argv[1])
kind=sys.argv[2]
if kind=="label":
    print(d.get("job") or (d.get("running") or [None])[0] or "")
elif kind=="invalid":
    print(", ".join(d.get("invalid_jobs") or []))
elif kind=="reason":
    print(d.get("conflict_reason") or "")
elif kind=="status":
    print(d.get("status") or "")
' "$1" "$2" 2>/dev/null || true
}

notify_discord() {
    local msg="$1"
    if [[ "${EXECUTOR_DISCORD:-1}" != "1" ]]; then
        return 0
    fi
    if [[ -z "${DISCORD_WEBHOOK_URL:-}" ]]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) discord skip: DISCORD_WEBHOOK_URL unset" >> "$LOG"
        return 0
    fi
    if [[ ! -d "$ROOT/util/discord_msg" ]]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) discord skip: util/discord_msg missing" >> "$LOG"
        return 0
    fi
    set +e
    PYTHONPATH="$ROOT/util" python3 -m discord_msg -c "$DISCORD_CFG" -- "$msg" >>"$LOG" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) discord send failed rc=$rc" >> "$LOG"
    fi
}

echo "$$" > "$PIDFILE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) executor_loop start pid=$$ interval=${INTERVAL}s" >> "$LOG"

while true; do
    sleep "$INTERVAL"
    set +e
    result="$("${POLL[@]}" 2>&1)"
    rc=$?
    set -e
    line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $result"
    echo "$line" >> "$LOG"
    status="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("status","GIT_ERROR"))' "$result" 2>/dev/null || echo GIT_ERROR)"
    if [[ "$status" == "JOB" ]]; then
        set +e
        claimed="$("${CLAIM[@]}" 2>&1)"
        claim_rc=$?
        set -e
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) claim $claimed" >> "$LOG"
        if [[ "$claim_rc" -eq 0 ]]; then
            result="$claimed"
            line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $result"
            status="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("status","GIT_ERROR"))' "$result" 2>/dev/null || echo GIT_ERROR)"
        else
            status="GIT_ERROR"
            result="$claimed"
            line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $result"
        fi
    fi
    wake_key="$(python3 -c '
import json, sys
raw = sys.argv[1]
try:
    d = json.loads(raw)
except Exception:
    print("GIT_ERROR:unparseable")
    raise SystemExit
print(json.dumps({
    "status": d.get("status"),
    "jobs": d.get("jobs") or [],
    "running": d.get("running") or [],
    "job": d.get("job"),
    "invalid_jobs": d.get("invalid_jobs") or [],
    "conflict_reason": d.get("conflict_reason"),
    "git_error": None if not d.get("git_error") else "present",
}, sort_keys=True, ensure_ascii=True))
' "$result" 2>/dev/null || echo "GIT_ERROR:unparseable")"
    last_key="$(cat "$LAST" 2>/dev/null || true)"
    label="$(payload_field "$result" label)"
    invalid_list="$(payload_field "$result" invalid)"
    reason="$(payload_field "$result" reason)"
    prev_status="$(python3 -c '
import json, sys
raw = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
if raw in ("", "IDLE"):
    print("IDLE")
else:
    try:
        print(json.loads(raw).get("status") or "IDLE")
    except Exception:
        print("IDLE")
' "$last_key")"
    if [[ "$status" == "IDLE" ]]; then
        if [[ "$prev_status" == "JOB" || "$prev_status" == "RECOVERY_REQUIRED" ]]; then
            notify_discord "Término da execução. De volta para idle."
        fi
        printf 'IDLE\n' > "$LAST"
        printf '\033[H\033[2J'
        printf '%s\n' "$line"
    elif [[ "$wake_key" != "$last_key" ]]; then
        printf '%s\n' "$wake_key" > "$LAST"
        printf '%s\n' "$line"
        echo "AGENT_LOOP_TICK_executor {\"prompt\":\"Execute the Git job per docs/executor_communication_protocol.md\",\"status\":\"$status\",\"payload\":$result}"
        case "$status" in
            JOB)
                notify_discord "Novo prompt encontrado${label:+: ${label}}"
                ;;
            RECOVERY_REQUIRED)
                notify_discord "Em execução${label:+: ${label}}"
                ;;
            QUEUE_INVALID)
                notify_discord "Fila inválida${invalid_list:+: ${invalid_list}}"
                ;;
            QUEUE_CONFLICT)
                notify_discord "Conflito de fila${reason:+: ${reason}}${label:+ (${label})}"
                ;;
        esac
    fi
done
