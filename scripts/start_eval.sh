#!/usr/bin/env bash
set +x
set -euo pipefail

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

prompt() {
    local label=$1 default=${2:-} secret=${3:-}
    local read_args=(-r)
    printf '%s' "$label" >&2
    if [[ -n "$default" ]]; then
        if [[ "$secret" == secret ]]; then
            printf ' [saved]' >&2
        else
            printf ' [%s]' "$default" >&2
        fi
    fi
    printf ': ' >&2
    if [[ "$secret" == secret ]]; then
        read_args+=(-s)
    fi
    if ! IFS= read "${read_args[@]}" REPLY; then
        fail 'Input ended before setup was complete; no evaluation started.'
    fi
    if [[ "$secret" == secret ]]; then
        printf '\n' >&2
    fi
    if [[ -z "$REPLY" ]]; then
        REPLY=$default
    elif [[ "$REPLY" == "-" || "$REPLY" == "none" || "$REPLY" == "clear" ]]; then
        REPLY=""
    fi
}

choose() {
    local label=$1 default=$2 option
    shift 2
    while true; do
        prompt "$label" "$default"
        for option in "$@"; do
            if [[ "$REPLY" == "$option" ]]; then
                return
            fi
        done
        printf 'Choose one of: %s\n' "$*" >&2
    done
}

confirm() {
    while true; do
        prompt "$1 (y/n)" "$2"
        case "$REPLY" in
            y|Y|yes|YES) return 0 ;;
            n|N|no|NO) return 1 ;;
            *) printf 'Enter y or n.\n' >&2 ;;
        esac
    done
}

endpoint() {
    local label=$1 default=${2:-} required=${3:-}
    while true; do
        prompt "$label" "$default"
        if [[ -z "$REPLY" ]]; then
            if [[ "$required" == required ]]; then
                printf 'An endpoint is required.\n' >&2
                continue
            fi
            return
        fi
        if [[ "$REPLY" == http://?* || "$REPLY" == https://?* ]]; then
            if [[ "$REPLY" != *[[:space:]]* && "$REPLY" != *@* ]]; then
                return
            fi
        fi
        printf 'Enter an http:// or https:// endpoint without spaces or embedded credentials.\n' >&2
    done
}

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
STATE_FILE="$ROOT/.start_eval.cache"

PREV_ENDPOINT=""
PREV_MODEL=""
PREV_MAX_CONTEXT=""
PREV_PARALLEL=""
PREV_JUDGE_ENDPOINT=""
PREV_JUDGE_MODEL=""
PREV_JUDGE_PARALLEL=""
PREV_MODE=""
PREV_LANGUAGE=""
PREV_HARDWARE=""
PREV_WORKLOAD_TTFT=""
PREV_WORKLOAD_COMPLETE=""

load_state() {
    local file=$1
    [[ -f "$file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# || -z "${line// }" ]] && continue
        if [[ "$line" =~ ^([A-Za-z0-9_]+)=(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local val="${BASH_REMATCH[2]}"
            if [[ ("$val" == \"*\" && "$val" == *\") || ("$val" == \'*\' && "$val" == *\') ]]; then
                val="${val:1:${#val}-2}"
            fi
            case "$key" in
                CACHED_ENDPOINT) PREV_ENDPOINT=$val ;;
                CACHED_MODEL) PREV_MODEL=$val ;;
                CACHED_MAX_CONTEXT) PREV_MAX_CONTEXT=$val ;;
                CACHED_PARALLEL) PREV_PARALLEL=$val ;;
                CACHED_JUDGE_ENDPOINT) PREV_JUDGE_ENDPOINT=$val ;;
                CACHED_JUDGE_MODEL) PREV_JUDGE_MODEL=$val ;;
                CACHED_JUDGE_PARALLEL) PREV_JUDGE_PARALLEL=$val ;;
                CACHED_MODE) PREV_MODE=$val ;;
                CACHED_LANGUAGE) PREV_LANGUAGE=$val ;;
                CACHED_HARDWARE) PREV_HARDWARE=$val ;;
                CACHED_WORKLOAD_TTFT) PREV_WORKLOAD_TTFT=$val ;;
                CACHED_WORKLOAD_COMPLETE) PREV_WORKLOAD_COMPLETE=$val ;;
            esac
        fi
    done < "$file"
}

save_state() {
    local file=$1
    local tmp="${file}.tmp.$$"
    (
        umask 077
        cat > "$tmp" <<EOF
CACHED_ENDPOINT=$OPENAI_BASE_URL
CACHED_MODEL=$OPENAI_MODEL
CACHED_MAX_CONTEXT=$MAX_CONTEXT
CACHED_PARALLEL=$PARALLEL
CACHED_JUDGE_ENDPOINT=$JUDGE_ENDPOINT
CACHED_JUDGE_MODEL=$JUDGE_MODEL
CACHED_JUDGE_PARALLEL=$JUDGE_PARALLEL
CACHED_MODE=$MODE
CACHED_LANGUAGE=$LANGUAGE
CACHED_HARDWARE=$HARDWARE
CACHED_WORKLOAD_TTFT=$WORKLOAD_TTFT
CACHED_WORKLOAD_COMPLETE=$WORKLOAD_COMPLETE
EOF
        mv -f "$tmp" "$file"
        chmod 600 "$file" 2>/dev/null || true
    )
}

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
        printf 'Usage: %s [--reset]\n' "$0"
        printf 'Interactively run a Docker model evaluation on Linux.\n'
        printf 'Asks for model, optional judge, test mode and language settings.\n'
        printf 'Remembers previous settings as defaults on subsequent runs.\n'
        printf 'Use --reset to clear remembered settings.\n'
        exit 0
    elif [[ $# -eq 1 && ( "$1" == --reset || "$1" == --clear ) ]]; then
        rm -f "$STATE_FILE"
        printf 'Cleared remembered evaluation defaults.\n'
        exit 0
    fi
    fail 'No arguments are accepted; use --help for usage.'
fi

IMAGE=llm-eval-suite:latest
command -v docker >/dev/null || fail 'Docker is not installed or is not on PATH.'
docker info >/dev/null || fail 'Cannot reach Docker. Start Docker and check your Docker permissions.'
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if ! confirm "Image $IMAGE is missing. Build it now?" y; then
        printf 'Cancelled; no evaluation started.\n'
        exit 0
    fi
    docker build --tag "$IMAGE" "$ROOT"
fi

load_state "$STATE_FILE"

DEFAULT_ENDPOINT=${PREV_ENDPOINT:-http://127.0.0.1:8080/v1}
endpoint 'Model endpoint' "$DEFAULT_ENDPOINT" required
OPENAI_BASE_URL=$REPLY

prompt 'Model ID (blank = auto-discover)' "$PREV_MODEL"
OPENAI_MODEL=$REPLY

prompt 'Model API key (blank = no authentication)' "" secret
OPENAI_API_KEY=$REPLY

while true; do
    prompt 'Context window in tokens (blank = auto-detect)' "$PREV_MAX_CONTEXT"
    MAX_CONTEXT=$REPLY
    if [[ -z "$MAX_CONTEXT" || "$MAX_CONTEXT" =~ ^[1-9][0-9]*$ ]]; then
        break
    fi
    printf 'Enter a positive whole number or leave blank.\n' >&2
done

while true; do
    prompt 'Model concurrency / parallel requests (blank = auto-detect via Test 3)' "$PREV_PARALLEL"
    PARALLEL=$REPLY
    if [[ -z "$PARALLEL" || "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
        break
    fi
    printf 'Enter a positive whole number or leave blank.\n' >&2
done

endpoint 'Independent judge endpoint (blank = objective grading)' "$PREV_JUDGE_ENDPOINT"
JUDGE_ENDPOINT=$REPLY
JUDGE_MODEL=
JUDGE_API_KEY=
JUDGE_PARALLEL=
if [[ -n "$JUDGE_ENDPOINT" ]]; then
    prompt 'Judge model ID (blank = auto-discover; must differ from tested model)' "$PREV_JUDGE_MODEL"
    JUDGE_MODEL=$REPLY
    prompt 'Judge API key (blank = no authentication)' "" secret
    JUDGE_API_KEY=$REPLY
    while true; do
        prompt 'Judge concurrency / parallel requests (blank = auto-detect via judge /health)' "$PREV_JUDGE_PARALLEL"
        JUDGE_PARALLEL=$REPLY
        if [[ -z "$JUDGE_PARALLEL" || "$JUDGE_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
            break
        fi
        printf 'Enter a positive whole number or leave blank.\n' >&2
    done
fi

DEFAULT_MODE=${PREV_MODE:-all}
choose 'Test mode: all / core / quick' "$DEFAULT_MODE" all core quick
MODE=$REPLY

DEFAULT_LANGUAGE=${PREV_LANGUAGE:-polyglot}
choose 'Coding language: polyglot / python / typescript / go / java / rust / csharp' \
    "$DEFAULT_LANGUAGE" polyglot python typescript go java rust csharp
LANGUAGE=$REPLY

prompt 'GPU hardware description (optional, e.g. 1x NVIDIA RTX PRO 6000)' "$PREV_HARDWARE"
HARDWARE=$REPLY

while true; do
    prompt 'Workload Test 46 max TTFT in seconds (blank = default 90, e.g. 300 for long docs)' "$PREV_WORKLOAD_TTFT"
    WORKLOAD_TTFT=$REPLY
    if [[ -z "$WORKLOAD_TTFT" || "$WORKLOAD_TTFT" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]]; then
        break
    fi
    printf 'Enter a positive number or leave blank.\n' >&2
done

while true; do
    prompt 'Workload Test 46 max completion latency in seconds (blank = default 300, e.g. 600)' "$PREV_WORKLOAD_COMPLETE"
    WORKLOAD_COMPLETE=$REPLY
    if [[ -z "$WORKLOAD_COMPLETE" || "$WORKLOAD_COMPLETE" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]]; then
        break
    fi
    printf 'Enter a positive number or leave blank.\n' >&2
done

printf '\nEndpoint: %s\nModel: %s\nContext: %s\nConcurrency: %s\nJudge: %s\nMode: %s\nLanguage: %s\n' \
    "$OPENAI_BASE_URL" "${OPENAI_MODEL:-auto-discover}" "${MAX_CONTEXT:-auto-detect}" \
    "${PARALLEL:-auto-detect}" "${JUDGE_ENDPOINT:-none (objective grading)}" "$MODE" "$LANGUAGE"
printf 'Workload gates: TTFT %s, Complete %s\n' "${WORKLOAD_TTFT:-90s (default)}" "${WORKLOAD_COMPLETE:-300s (default)}"
printf 'Reports: %s/results\nLogs: %s/logs\n' "$ROOT" "$ROOT"
printf 'A full run can take several hours and stress the server.\n'
printf 'Model code has container network access. Use trusted models or a dedicated host/VM.\n'
if ! confirm 'Start evaluation?' n; then
    printf 'Cancelled; no evaluation started.\n'
    exit 0
fi

save_state "$STATE_FILE"

export OPENAI_BASE_URL OPENAI_MODEL OPENAI_API_KEY MAX_CONTEXT
export JUDGE_ENDPOINT JUDGE_MODEL JUDGE_API_KEY
docker_args=(run --rm --init --network host
    --env OPENAI_BASE_URL --env OPENAI_MODEL --env OPENAI_API_KEY --env MAX_CONTEXT
    --env JUDGE_ENDPOINT --env JUDGE_MODEL --env JUDGE_API_KEY)

printf '\nChecking the model endpoint...\n'
if ! docker "${docker_args[@]}" "$IMAGE" --probe; then
    fail 'Endpoint probe failed; the evaluation was not started.'
fi

eval_args=(--suite all --language "$LANGUAGE")
if [[ "$MODE" == core || "$MODE" == quick ]]; then
    eval_args=(--suite core --language "$LANGUAGE")
fi
if [[ "$MODE" == quick ]]; then
    eval_args+=(--quick)
fi
if [[ -n "$HARDWARE" ]]; then
    eval_args+=(--hardware "$HARDWARE")
fi
if [[ -n "$PARALLEL" ]]; then
    eval_args+=(--parallel "$PARALLEL")
fi
if [[ -n "$JUDGE_PARALLEL" ]]; then
    eval_args+=(--judge-parallel "$JUDGE_PARALLEL")
fi
if [[ -n "$WORKLOAD_TTFT" ]]; then
    eval_args+=(--workload-ttft "$WORKLOAD_TTFT")
fi
if [[ -n "$WORKLOAD_COMPLETE" ]]; then
    eval_args+=(--workload-complete "$WORKLOAD_COMPLETE")
fi

mkdir -p "$ROOT/results" "$ROOT/logs"
exec docker "${docker_args[@]}" \
    --volume "$ROOT/results:/app/results" --volume "$ROOT/logs:/app/logs" \
    "$IMAGE" "${eval_args[@]}"
