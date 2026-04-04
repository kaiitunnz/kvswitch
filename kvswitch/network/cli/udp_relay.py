"""UDP request/response relay used by Mininet evaluation helpers."""

import argparse
import asyncio
from dataclasses import dataclass, field

from kvswitch.utils.logger import setup_logging
from kvswitch.utils.udp import UDPClient, UDPRequest, UDPResponse, UDPServer


@dataclass
class UDPRelay:
    """Forward UDP JSON requests to an upstream endpoint and relay the reply."""

    upstream_host: str
    upstream_port: int
    host: str = "0.0.0.0"
    port: int = 9100
    timeout: float = 5.0
    _server: UDPServer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._server = UDPServer(host=self.host, port=self.port, handler=self._handle)

    @property
    def bound_port(self) -> int:
        return self._server.port

    async def start(self) -> None:
        await self._server.start()

    def close(self) -> None:
        self._server.close()

    async def _handle(self, request: UDPRequest) -> UDPResponse:
        client = UDPClient(
            host=self.upstream_host,
            port=self.upstream_port,
            timeout=self.timeout,
        )
        response = await client.send(request.data)
        return UDPResponse(data=response)


async def _run(args: argparse.Namespace) -> None:
    relay = UDPRelay(
        host=args.host,
        port=args.port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
        timeout=args.timeout,
    )
    await relay.start()
    try:
        await asyncio.Event().wait()
    finally:
        relay.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="UDP relay for cache-event forwarding")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--upstream-host", type=str, required=True)
    parser.add_argument("--upstream-port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
