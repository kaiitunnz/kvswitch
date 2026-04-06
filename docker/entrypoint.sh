#!/usr/bin/env bash
# Docker entrypoint for the Mininet container.
# Starts OVS, runs the command under the project venv, then fixes
# output file ownership so results are readable by the host user.
set -euo pipefail

# OVS needs its daemon running for OVSBridge to work.
service openvswitch-switch start >/dev/null 2>&1

PYTHON="/opt/kvswitch-venv/bin/python3"

"$PYTHON" "$@"
exit_code=$?

# Fix ownership of results so the host user can read/write them.
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "$HOST_UID:$HOST_GID" /kvswitch/results 2>/dev/null || true
fi

exit "$exit_code"
