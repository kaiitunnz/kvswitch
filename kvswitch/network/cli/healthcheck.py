"""UDP healthcheck CLI for Mininet experiment hosts."""

import argparse
import asyncio
from collections.abc import Sequence

from kvswitch.utils.udp import UDPClient


async def check_health(host: str, port: int, timeout: float) -> str:
    """Return the service status reported by the UDP health endpoint."""
    client = UDPClient(host=host, port=port, timeout=timeout)
    response = await client.send({"endpoint": "health"})
    return str(response.get("status", "unknown"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="UDP service healthcheck")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args(argv)

    print(asyncio.run(check_health(args.host, args.port, args.timeout)))


if __name__ == "__main__":
    main()
