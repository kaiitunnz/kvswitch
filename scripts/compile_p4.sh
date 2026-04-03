#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

PROGRAM_PATH="${1:-p4/kvswitch.p4}"
OUTPUT_PATH="${2:-build/p4/kvswitch.json}"
P4C_IMAGE="${P4C_IMAGE:-p4c:latest}"

usage() {
    echo "Usage: bash scripts/compile_p4.sh [program.p4] [output.json]"
    echo ""
    echo "Defaults:"
    echo "  program.p4  = p4/kvswitch.p4"
    echo "  output.json = build/p4/kvswitch.json"
}

if [[ "${PROGRAM_PATH}" == "-h" || "${PROGRAM_PATH}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$ROOT/$PROGRAM_PATH" ]]; then
    echo "P4 program not found: $PROGRAM_PATH" >&2
    exit 1
fi

mkdir -p "$ROOT/$(dirname "$OUTPUT_PATH")"

compile_with_local_p4c() {
    echo "Compiling $PROGRAM_PATH with local p4c..."
    p4c --target bmv2 --arch v1model -o "$ROOT/$OUTPUT_PATH" "$ROOT/$PROGRAM_PATH"
}

compile_with_docker_p4c() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Neither p4c nor docker is available." >&2
        exit 1
    fi

    echo "Compiling $PROGRAM_PATH with Docker image $P4C_IMAGE..."
    docker run --rm \
        -v "$ROOT:/work" \
        -w /work \
        "$P4C_IMAGE" \
        p4c --target bmv2 --arch v1model -o "/work/$OUTPUT_PATH" "/work/$PROGRAM_PATH"
}

if command -v p4c >/dev/null 2>&1; then
    compile_with_local_p4c
else
    compile_with_docker_p4c
fi

echo "Wrote compiled P4 artifact to $OUTPUT_PATH"
