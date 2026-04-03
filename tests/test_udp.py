"""Tests for kvswitch.utils.udp — async UDP server and client."""

import asyncio

import pytest

from kvswitch.utils.udp import UDPClient, UDPRequest, UDPResponse, UDPServer

# ---------------------------------------------------------------------------
# UDPRequest / UDPResponse
# ---------------------------------------------------------------------------


class TestUDPResponse:
    def test_encode(self) -> None:
        resp = UDPResponse(data={"status": "ok"})
        encoded = resp.encode()
        assert isinstance(encoded, bytes)
        assert b'"status"' in encoded
        assert b'"ok"' in encoded

    def test_encode_nested(self) -> None:
        resp = UDPResponse(data={"a": [1, 2], "b": {"c": 3}})
        encoded = resp.encode()
        assert b'"a"' in encoded
        assert b'"b"' in encoded


class TestUDPRequest:
    def test_fields(self) -> None:
        req = UDPRequest(data={"key": "value"}, addr=("127.0.0.1", 9999))
        assert req.data["key"] == "value"
        assert req.addr == ("127.0.0.1", 9999)


# ---------------------------------------------------------------------------
# Server + Client integration (loopback)
# ---------------------------------------------------------------------------


async def _echo_handler(request: UDPRequest) -> UDPResponse:
    """Echo the request data back with an 'echo' wrapper."""
    return UDPResponse(data={"echo": request.data})


async def _error_handler(request: UDPRequest) -> UDPResponse:
    raise ValueError("intentional test error")


class TestUDPServerClient:
    def test_echo_roundtrip(self) -> None:
        async def _run() -> None:
            server = UDPServer(host="127.0.0.1", port=0, handler=_echo_handler)
            await server.start()
            port = server.bound_port()
            assert server.port == port

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            response = await client.send({"hello": "world"})

            assert response == {"echo": {"hello": "world"}}
            server.close()

        asyncio.run(_run())

    def test_multiple_requests(self) -> None:
        async def _run() -> None:
            server = UDPServer(host="127.0.0.1", port=0, handler=_echo_handler)
            await server.start()
            port = server.bound_port()

            for i in range(5):
                client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                resp = await client.send({"i": i})
                assert resp == {"echo": {"i": i}}

            server.close()

        asyncio.run(_run())

    def test_server_handler_error(self) -> None:
        async def _run() -> None:
            server = UDPServer(host="127.0.0.1", port=0, handler=_error_handler)
            await server.start()
            port = server.bound_port()

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send({"test": 1})
            assert "error" in resp

            server.close()

        asyncio.run(_run())

    def test_client_timeout(self) -> None:
        async def _run() -> None:
            # Connect to a port with no server — should timeout or get refused.
            client = UDPClient(host="127.0.0.1", port=19999, timeout=0.1)
            with pytest.raises((asyncio.TimeoutError, ConnectionRefusedError, OSError)):
                await client.send({"test": 1})

        asyncio.run(_run())

    def test_invalid_json(self) -> None:
        async def _run() -> None:
            server = UDPServer(host="127.0.0.1", port=0, handler=_echo_handler)
            await server.start()
            port = server.bound_port()

            # Send raw invalid JSON via low-level socket.
            loop = asyncio.get_running_loop()
            transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", port),
            )

            future: asyncio.Future[bytes] = loop.create_future()

            class _Recv(asyncio.DatagramProtocol):
                def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:  # type: ignore[override]
                    if not future.done():
                        future.set_result(data)

            transport2, _ = await loop.create_datagram_endpoint(
                _Recv,
                remote_addr=("127.0.0.1", port),
            )
            transport2.sendto(b"not json{{{")

            resp = await asyncio.wait_for(future, timeout=5.0)
            assert b"error" in resp
            assert b"invalid JSON" in resp

            transport.close()
            transport2.close()
            server.close()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# UDPServer start/close lifecycle
# ---------------------------------------------------------------------------


class TestUDPServerLifecycle:
    def test_close_without_start(self) -> None:
        server = UDPServer(host="127.0.0.1", port=0, handler=_echo_handler)
        # Should not raise.
        server.close()

    def test_port_property_returns_requested_port_before_start(self) -> None:
        server = UDPServer(host="127.0.0.1", port=43210, handler=_echo_handler)
        assert server.port == 43210

    def test_bound_port_requires_start(self) -> None:
        server = UDPServer(host="127.0.0.1", port=0, handler=_echo_handler)
        with pytest.raises(RuntimeError, match="not started"):
            server.bound_port()

    def test_start_requires_handler(self) -> None:
        async def _run() -> None:
            server = UDPServer(host="127.0.0.1", port=0)
            with pytest.raises(AssertionError, match="handler must be set"):
                await server.start()

        asyncio.run(_run())
