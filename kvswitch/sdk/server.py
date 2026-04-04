"""KVSwitch-aware UDP server protocol.

Provides ``KVSwitchUDPServerProtocol`` which strips the 20-byte binary
shim header from incoming packets before dispatching to a request
handler.  Used by the mock worker to listen on port 4789.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

from kvswitch.sdk.header import HEADER_SIZE, KVSwitchShimHeader
from kvswitch.utils.udp import UDPRequest, UDPResponse

logger = logging.getLogger(__name__)


class KVSwitchUDPServerProtocol(asyncio.DatagramProtocol):
    """UDP server protocol that strips the shim header before dispatching."""

    def __init__(self, handler: Callable[[UDPRequest], Awaitable[UDPResponse]]) -> None:
        self.handler = handler
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        asyncio.ensure_future(self._handle(data, addr))

    async def _handle(self, data: bytes, addr: tuple[str, int]) -> None:
        assert self.transport is not None
        try:
            if len(data) > HEADER_SIZE:
                shim = KVSwitchShimHeader.decode(data[:HEADER_SIZE])
                json_bytes = data[HEADER_SIZE:]
                request_dict = json.loads(json_bytes.decode("utf-8"))
                request_dict["_kvswitch_shim"] = shim.to_dict()
            else:
                request_dict = json.loads(data.decode("utf-8"))

            request = UDPRequest(data=request_dict, addr=addr)
            response = await self.handler(request)
            self.transport.sendto(response.encode(), addr)
        except json.JSONDecodeError:
            err = UDPResponse(data={"error": "invalid JSON"})
            self.transport.sendto(err.encode(), addr)
        except Exception as e:
            logger.exception("Error handling KVSwitch UDP request from %s", addr)
            err = UDPResponse(data={"error": str(e)})
            self.transport.sendto(err.encode(), addr)

    def error_received(self, exc: Exception) -> None:
        logger.error("KVSwitch UDP protocol error: %s", exc)
