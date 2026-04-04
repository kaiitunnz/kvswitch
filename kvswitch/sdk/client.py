"""KVSwitch client — sends UDP requests with the binary shim header.

Wire format (what the BMv2 switch sees)::

    [UDP dstPort=4789][shim header (20 bytes)][JSON payload]

The shim header carries prefix hashes (h0–h3) that the switch matches
against its TCAM tables.  The JSON payload after the shim is the same
request/response format used by the regular ``UDPClient``.
"""

import asyncio
import json
import logging
from typing import Any

from kvswitch.sdk.header import HEADER_SIZE, KVSwitchShimHeader

logger = logging.getLogger(__name__)

KVSWITCH_UDP_PORT = 4789


class KVSwitchUDPClient:
    """UDP client that prepends a KVSwitch shim header to each request.

    The shim header is constructed from *prefix_hashes* passed to
    :meth:`send`.  The destination port defaults to 4789 so that the
    BMv2 switch activates the KVSwitch parser.
    """

    def __init__(
        self,
        host: str,
        port: int = KVSWITCH_UDP_PORT,
        timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    async def send(
        self,
        data: dict[str, Any],
        prefix_hashes: list[int] | None = None,
        req_id: int = 0,
    ) -> dict[str, Any]:
        """Send a JSON request with a shim header and wait for a response."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()

        class _Protocol(asyncio.DatagramProtocol):
            def __init__(self) -> None:
                self.transport: asyncio.DatagramTransport | None = None

            def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
                self.transport = transport

            def datagram_received(self, recv_data: bytes, addr: tuple[str, int]) -> None:  # type: ignore[override]
                if not future.done():
                    future.set_result(recv_data)

            def error_received(self, exc: Exception) -> None:
                if not future.done():
                    future.set_exception(exc)

        transport, _ = await loop.create_datagram_endpoint(
            _Protocol,
            remote_addr=(self.host, self.port),
        )
        try:
            hashes = prefix_hashes or []
            shim = KVSwitchShimHeader.from_hashes(hashes, req_id=req_id % 0xFFFF)
            shim_bytes = shim.encode()

            json_bytes = json.dumps(data).encode("utf-8")
            transport.sendto(shim_bytes + json_bytes)

            response_bytes = await asyncio.wait_for(future, timeout=self.timeout)

            # The response may have the shim header prepended (if the
            # switch re-emits it on the return path).  Try to strip it;
            # fall back to parsing as plain JSON.
            if len(response_bytes) > HEADER_SIZE:
                try:
                    return json.loads(response_bytes[HEADER_SIZE:].decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            return json.loads(response_bytes.decode("utf-8"))
        finally:
            transport.close()
