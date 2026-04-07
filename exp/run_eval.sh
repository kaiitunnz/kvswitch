#!/usr/bin/env bash
# Run the KVSwitch evaluation experiment inside Docker.
#
# Usage:
#   bash exp/run_eval.sh [extra args...]
#
# Examples:
#   bash exp/run_eval.sh --baselines l4_ecmp --num-requests 20
#   bash exp/run_eval.sh --baselines l4_ecmp,l7_rr,l7_pa,kvswitch --request-rate 10
#   bash exp/run_eval.sh --prefix-sharing-ratio 0.75

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="kvswitch-mininet"

# Build the Docker image (layer-cached after first run).
echo "Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" -f "$ROOT/docker/Dockerfile.mininet" "$ROOT"

echo ""
echo "Running evaluation experiment in Docker..."
echo "  Args: $*"
echo ""

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
P4_BUILD_DIR="$ROOT/build/p4/kvswitch"

if [[ ! -f "$P4_BUILD_DIR/kvswitch.json" || ! -f "$P4_BUILD_DIR/kvswitch.p4info.txtpb" ]]; then
    echo "Compiling P4 artifacts into $P4_BUILD_DIR ..."
    bash "$ROOT/scripts/compile_p4.sh" p4/kvswitch.p4 build/p4/kvswitch
fi

HOST_UID=$(id -u)
HOST_GID=$(id -g)

docker run --rm --privileged \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e HOST_UID="$HOST_UID" \
    -e HOST_GID="$HOST_GID" \
    -v /lib/modules:/lib/modules:ro \
    -v "$ROOT/kvswitch:/kvswitch/kvswitch:ro" \
    -v "$ROOT/exp:/kvswitch/exp:ro" \
    -v "$ROOT/results:/kvswitch/results" \
    -v "$ROOT/data:/kvswitch/data:ro" \
    -v "$ROOT/build:/kvswitch/build:ro" \
    -v "$ROOT/p4:/kvswitch/p4:ro" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "$IMAGE_NAME" \
    exp/run_eval.py "$@"
