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
        printf ' [%s]' "$default" >&2
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
    REPLY=${REPLY:-$default}
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
    while true; do
        prompt "$1" "${2:-}"
        if [[ -z "$REPLY" ]]; then
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

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
        printf 'Usage: %s\n' "$0"
        printf 'Interactively run a Docker model evaluation on Linux.\n'
        printf 'Asks for model, optional judge, test mode and language settings.\n'
        printf 'API keys are hidden and passed by environment variable name, never saved.\n'
        exit 0
    fi
    fail 'No arguments are accepted; use --help for usage.'
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
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

endpoint 'Model endpoint' 'http://127.0.0.1:8080/v1'
OPENAI_BASE_URL=$REPLY
prompt 'Model ID (blank = auto-discover)'
OPENAI_MODEL=$REPLY
prompt 'Model API key (blank = no authentication)' '' secret
OPENAI_API_KEY=$REPLY
while true; do
    prompt 'Context window in tokens (blank = auto-detect)'
    MAX_CONTEXT=$REPLY
    if [[ -z "$MAX_CONTEXT" || "$MAX_CONTEXT" =~ ^[1-9][0-9]*$ ]]; then
        break
    fi
    printf 'Enter a positive whole number or leave blank.\n' >&2
done

endpoint 'Independent judge endpoint (blank = objective grading)'
JUDGE_ENDPOINT=$REPLY
JUDGE_MODEL=
JUDGE_API_KEY=
if [[ -n "$JUDGE_ENDPOINT" ]]; then
    prompt 'Judge model ID (blank = auto-discover; must differ from tested model)'
    JUDGE_MODEL=$REPLY
    prompt 'Judge API key (blank = no authentication)' '' secret
    JUDGE_API_KEY=$REPLY
fi

choose 'Test mode: all / core / quick' all all core quick
MODE=$REPLY
choose 'Coding language: polyglot / python / typescript / go / java / rust / csharp' \
    polyglot polyglot python typescript go java rust csharp
LANGUAGE=$REPLY
prompt 'GPU hardware description (optional, e.g. 1x NVIDIA RTX PRO 6000)'
HARDWARE=$REPLY

printf '\nEndpoint: %s\nModel: %s\nContext: %s\nJudge: %s\nMode: %s\nLanguage: %s\n' \
    "$OPENAI_BASE_URL" "${OPENAI_MODEL:-auto-discover}" "${MAX_CONTEXT:-auto-detect}" \
    "${JUDGE_ENDPOINT:-none (objective grading)}" "$MODE" "$LANGUAGE"
printf 'Reports: %s/results\nLogs: %s/logs\n' "$ROOT" "$ROOT"
printf 'A full run can take several hours and stress the server.\n'
printf 'Model code has container network access. Use trusted models or a dedicated host/VM.\n'
if ! confirm 'Start evaluation?' n; then
    printf 'Cancelled; no evaluation started.\n'
    exit 0
fi

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

mkdir -p "$ROOT/results" "$ROOT/logs"
exec docker "${docker_args[@]}" \
    --volume "$ROOT/results:/app/results" --volume "$ROOT/logs:/app/logs" \
    "$IMAGE" "${eval_args[@]}"
