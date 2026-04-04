#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

PROGRAM_PATH="${1:-p4/kvswitch.p4}"
OUTPUT_DIR="${2:-build/p4/kvswitch}"
P4C_IMAGE="${P4C_IMAGE:-p4c:latest}"

usage() {
    echo "Usage: bash scripts/compile_p4.sh [program.p4] [output_dir]"
    echo ""
    echo "Defaults:"
    echo "  program.p4  = p4/kvswitch.p4"
    echo "  output_dir  = build/p4/kvswitch"
    echo ""
    echo "Produces both the BMv2 JSON and the P4Runtime .p4info.txtpb."
}

if [[ "${PROGRAM_PATH}" == "-h" || "${PROGRAM_PATH}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$ROOT/$PROGRAM_PATH" ]]; then
    echo "P4 program not found: $PROGRAM_PATH" >&2
    exit 1
fi

mkdir -p "$ROOT/$OUTPUT_DIR"

# Derive the p4info path from the output directory and program name.
BASENAME="$(basename "$PROGRAM_PATH" .p4)"
P4INFO_PATH="$OUTPUT_DIR/$BASENAME.p4info.txtpb"

P4C_ARGS=(
    --target bmv2 --arch v1model
    --p4runtime-files "$P4INFO_PATH"
    -o "$OUTPUT_DIR"
    "$PROGRAM_PATH"
)

compile_with_local_p4c() {
    echo "Compiling $PROGRAM_PATH with local p4c..."
    (cd "$ROOT" && p4c "${P4C_ARGS[@]}")
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
        p4c "${P4C_ARGS[@]}"
}

if command -v p4c >/dev/null 2>&1; then
    compile_with_local_p4c
else
    compile_with_docker_p4c
fi

echo "Wrote compiled P4 artifacts to $OUTPUT_DIR/"
echo "  JSON:   $OUTPUT_DIR/$BASENAME.json"
echo "  P4Info: $P4INFO_PATH"
