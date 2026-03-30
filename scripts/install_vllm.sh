#!/bin/bash

set -eou pipefail

git submodule update --init
cd 3rdparty/vllm
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=89a77b10846fd96273cce78d86d2556ea582d26e # Upstream v0.16.0 wheel commit
uv pip install -e .
cd -
