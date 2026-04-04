"""Async workload client CLI for Mininet experiment hosts.

Reads a pre-generated workload JSON file and fires requests at Poisson-
scheduled times.  Unlike ``measure_client.py`` (which sends sequentially),
this client sends requests concurrently to model realistic arrival
patterns.

When ``--kvswitch`` is given, the client prepends a binary KVSwitch
shim header (20 bytes) to each UDP packet and sends to port 4789 so
that BMv2 switches can perform in-network prefix matching.

Each request's response is augmented with client-side timing and printed
as a JSON array to stdout.

Usage::

    python -m kvswitch.network.cli.workload_client \\
        --workload /tmp/workload.json \\
        --host 10.0.0.1 --port 8000

    # With KVSwitch shim header (for BMv2 switch routing):
    python -m kvswitch.network.cli.workload_client \\
        --workload /tmp/workload.json \\
        --host 10.0.0.1 --port 4789 --kvswitch
"""

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

from kvswitch.sdk.client import KVSwitchUDPClient
from kvswitch.utils.udp import UDPClient


async def _send_one(
    request: dict,
    host: str,
    port: int,
    timeout: float,
    t0: float,
    kvswitch: bool = False,
) -> dict:
    """Send a single request and return a metric dict."""
    actual_send = time.perf_counter() - t0

    payload: dict[str, Any] = {
        "endpoint": "generate",
        "prompt_token_ids": request["prompt_token_ids"],
        "max_tokens": request.get("max_tokens", 16),
    }
    if not kvswitch and "prefix_hashes" in request:
        payload["prefix_hashes"] = request["prefix_hashes"]

    start = time.perf_counter()
    try:
        if kvswitch:
            kv_client = KVSwitchUDPClient(host=host, port=port, timeout=timeout)
            resp = await kv_client.send(
                payload,
                prefix_hashes=request.get("prefix_hashes"),
                req_id=request.get("request_id", 0),
            )
        else:
            udp_client = UDPClient(host=host, port=port, timeout=timeout)
            resp = await udp_client.send(payload)
    except Exception as exc:
        resp = {"error": str(exc)}
    e2e_ms = (time.perf_counter() - start) * 1000

    return {
        "request_id": request["request_id"],
        "e2e_latency_ms": e2e_ms,
        "actual_send_time_s": actual_send,
        "scheduled_time_s": request.get("scheduled_time", 0.0),
        "prefix_group": request.get("prefix_group", "none"),
        "prompt_tokens": len(request.get("prompt_token_ids", [])),
        "max_tokens": request.get("max_tokens", 16),
        **{
            k: resp.get(k)
            for k in (
                "simulated_ttft_ms",
                "simulated_tpot_ms",
                "simulated_e2e_ms",
                "matched_blocks",
                "worker_id",
                "worker_port",
                "output_tokens",
                "num_cached_tokens",
                "error",
            )
        },
        "routing": resp.get("routing"),
    }


async def run_workload(
    workload_path: str,
    host: str,
    port: int,
    timeout: float,
    kvswitch: bool = False,
) -> list[dict]:
    """Load workload JSON, fire requests at scheduled times, collect results."""
    with open(workload_path) as f:
        requests = json.load(f)

    t0 = time.perf_counter()
    tasks: list[asyncio.Task] = []

    for req in requests:
        delay = req["scheduled_time"] - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(
            asyncio.create_task(
                _send_one(req, host, port, timeout, t0, kvswitch=kvswitch)
            )
        )

    return list(await asyncio.gather(*tasks))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Async workload client for evaluation experiments"
    )
    parser.add_argument(
        "--workload", type=str, required=True, help="Path to workload JSON"
    )
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--kvswitch",
        action="store_true",
        help="Send with KVSwitch shim header on port 4789 for BMv2 switch routing",
    )
    args = parser.parse_args(argv)

    results = asyncio.run(
        run_workload(
            workload_path=args.workload,
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            kvswitch=args.kvswitch,
        )
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
