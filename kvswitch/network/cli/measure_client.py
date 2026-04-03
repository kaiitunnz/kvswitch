"""Latency measurement CLI for Mininet experiment hosts."""

import argparse
import asyncio
import json
import time
from collections.abc import Sequence

from kvswitch.utils.udp import UDPClient


async def measure_latencies(
    host: str,
    port: int,
    n: int,
    prompt_tokens: int,
    timeout: float,
) -> list[float]:
    """Return per-request RTTs in milliseconds for UDP generate requests."""
    results: list[float] = []
    for _ in range(n):
        client = UDPClient(host=host, port=port, timeout=timeout)
        start = time.perf_counter()
        await client.send(
            {"endpoint": "generate", "prompt_token_ids": list(range(prompt_tokens))}
        )
        results.append((time.perf_counter() - start) * 1000)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure UDP request latencies")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    results = asyncio.run(
        measure_latencies(
            host=args.host,
            port=args.port,
            n=args.n,
            prompt_tokens=args.prompt_tokens,
            timeout=args.timeout,
        )
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
