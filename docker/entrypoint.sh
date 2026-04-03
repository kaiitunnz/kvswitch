#!/usr/bin/env bash
# Docker entrypoint for the Mininet container.
# Starts OVS, then execs the given command under the project venv.
set -euo pipefail

# OVS needs its daemon running for OVSBridge to work.
service openvswitch-switch start >/dev/null 2>&1

PYTHON="/opt/kvswitch-venv/bin/python3"

exec "$PYTHON" "$@"
