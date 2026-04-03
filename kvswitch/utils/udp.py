import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_UDP_PAYLOAD = 65535


@dataclass
class UDPRequest:
    """Represents an incoming UDP request."""

    data: dict[str, Any]
    addr: tuple[str, int]


@dataclass
class UDPResponse:
    """Represents a UDP response to send back."""

    data: dict[str, Any]

    def encode(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")


RequestHandler = Callable[[UDPRequest], Coroutine[Any, Any, UDPResponse]]


class UDPServerProtocol(asyncio.DatagramProtocol):
    """asyncio datagram protocol that dispatches JSON requests to a handler."""

    def __init__(self, handler: RequestHandler) -> None:
        self.handler = handler
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        asyncio.ensure_future(self._handle(data, addr))

    async def _handle(self, data: bytes, addr: tuple[str, int]) -> None:
        assert self.transport is not None
        try:
            request_dict = json.loads(data.decode("utf-8"))
            request = UDPRequest(data=request_dict, addr=addr)
            response = await self.handler(request)
            self.transport.sendto(response.encode(), addr)
        except json.JSONDecodeError:
            err = UDPResponse(data={"error": "invalid JSON"})
            self.transport.sendto(err.encode(), addr)
        except Exception as e:
            logger.exception("Error handling UDP request from %s", addr)
            err = UDPResponse(data={"error": str(e)})
            self.transport.sendto(err.encode(), addr)

    def error_received(self, exc: Exception) -> None:
        logger.error("UDP protocol error: %s", exc)


@dataclass
class UDPServer:
    """Simple async UDP server."""

    host: str = "0.0.0.0"
    port: int = 8000
    handler: RequestHandler | None = None
    _transport: asyncio.DatagramTransport | None = field(
        default=None, init=False, repr=False
    )
    _protocol: UDPServerProtocol | None = field(default=None, init=False, repr=False)

    async def start(self) -> None:
        assert self.handler is not None, "handler must be set before starting"
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: UDPServerProtocol(self.handler),  # type: ignore[arg-type]
            local_addr=(self.host, self.port),
        )
        logger.info("UDP server listening on %s:%d", self.host, self.port)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            logger.info("UDP server closed")


@dataclass
class UDPClient:
    """Simple async UDP client."""

    host: str = "127.0.0.1"
    port: int = 8000
    timeout: float = 30.0

    async def send(self, data: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON request and wait for a JSON response."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()

        class _ClientProtocol(asyncio.DatagramProtocol):
            def __init__(self) -> None:
                self.transport: asyncio.DatagramTransport | None = None

            def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
                self.transport = transport

            def datagram_received(  # type: ignore[override]
                self, recv_data: bytes, addr: tuple[str, int]
            ) -> None:
                if not future.done():
                    future.set_result(recv_data)

            def error_received(self, exc: Exception) -> None:
                if not future.done():
                    future.set_exception(exc)

        transport, protocol = await loop.create_datagram_endpoint(
            _ClientProtocol,
            remote_addr=(self.host, self.port),
        )
        try:
            payload = json.dumps(data).encode("utf-8")
            transport.sendto(payload)
            response_bytes = await asyncio.wait_for(future, timeout=self.timeout)
            return json.loads(response_bytes.decode("utf-8"))
        finally:
            transport.close()
