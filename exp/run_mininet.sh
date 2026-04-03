#!/usr/bin/env bash
# Run the Mininet experiment inside Docker.
#
# First run builds the image (~2-3 min); subsequent runs use cache.
# Source code is bind-mounted so code changes don't need a rebuild.
# Results are written to results/mininet/comparison.json on the host.
#
# Usage:
#   bash exp/run_mininet.sh [extra args...]
#
# Examples:
#   bash exp/run_mininet.sh
#   bash exp/run_mininet.sh --n-requests 50 --ttft-ms 15
#   bash exp/run_mininet.sh --output results/mininet/custom.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="kvswitch-mininet"

# Build the Docker image (layer-cached after first run).
echo "Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" -f "$ROOT/docker/Dockerfile.mininet" "$ROOT"

echo ""
echo "Running Mininet experiment in Docker..."
echo "  Args: $*"
echo ""

# Resolve HuggingFace cache directory from environment.
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

# --privileged: Mininet needs full access to network namespaces.
# -v /lib/modules: OVS kernel modules.
# -v $ROOT/kvswitch: bind-mount source so code edits take effect immediately.
# -v $ROOT/exp: bind-mount experiment scripts.
# -v $ROOT/results: persist output on host.
# -v $HF_CACHE: share host's HuggingFace model cache to avoid re-downloads.
docker run --rm --privileged \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -v /lib/modules:/lib/modules:ro \
    -v "$ROOT/kvswitch:/kvswitch/kvswitch:ro" \
    -v "$ROOT/exp:/kvswitch/exp:ro" \
    -v "$ROOT/results:/kvswitch/results" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "$IMAGE_NAME" \
    exp/run_mininet.py "$@"
