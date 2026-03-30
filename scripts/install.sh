#!/bin/bash

set -eou pipefail

# Check if .venv exists and exit if it does
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Please remove .venv before running this script."
    exit 1
fi

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync --all-extras

# Install vllm
bash scripts/install_vllm.sh
