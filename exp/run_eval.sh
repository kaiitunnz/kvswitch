#!/usr/bin/env bash
# Run the KVSwitch evaluation experiment inside Docker.
#
# Usage:
#   bash exp/run_eval.sh [extra args...]
#
# Examples:
#   bash exp/run_eval.sh --baselines l4_rr --num-requests 20 --n-workers 2
#   bash exp/run_eval.sh --baselines l4_rr,l7,kvswitch --request-rate 10
#   bash exp/run_eval.sh --prefix-sharing-ratio 0.75 --delay 0.1ms

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

docker run --rm --privileged \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
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
