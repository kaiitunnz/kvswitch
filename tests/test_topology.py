"""Tests for kvswitch.network.topology — Topo graph structure only (no sudo)."""

from kvswitch.network.topology import (
    CLIENT_IP,
    ROUTER_IP,
    SpineLeafTopo,
    worker_ip,
)

# ---------------------------------------------------------------------------
# worker_ip helper
# ---------------------------------------------------------------------------


class TestWorkerIp:
    def test_first(self) -> None:
        assert worker_ip(0) == "10.0.0.1"

    def test_nth(self) -> None:
        assert worker_ip(4) == "10.0.0.5"


# ---------------------------------------------------------------------------
# SpineLeafTopo graph structure
# ---------------------------------------------------------------------------


class TestSpineLeafTopo:
    def test_default_hosts(self) -> None:
        topo = SpineLeafTopo(n_workers=1)
        hosts = sorted(topo.hosts())
        assert hosts == ["client", "worker0"]

    def test_default_switches(self) -> None:
        topo = SpineLeafTopo(n_workers=1)
        switches = sorted(topo.switches())
        assert switches == ["s1", "s2"]

    def test_multiple_workers(self) -> None:
        topo = SpineLeafTopo(n_workers=3)
        hosts = sorted(topo.hosts())
        assert hosts == ["client", "worker0", "worker1", "worker2"]

    def test_link_count_no_router(self) -> None:
        topo = SpineLeafTopo(n_workers=2)
        # spine-leaf + client-spine + 2x worker-leaf = 4 links
        assert len(topo.links()) == 4

    def test_with_router(self) -> None:
        topo = SpineLeafTopo(n_workers=1, with_router=True)
        hosts = sorted(topo.hosts())
        assert "router" in hosts
        assert hosts == ["client", "router", "worker0"]

    def test_with_router_link_count(self) -> None:
        topo = SpineLeafTopo(n_workers=1, with_router=True)
        # spine-leaf + client-spine + router-spine + worker-leaf = 4 links
        assert len(topo.links()) == 4

    def test_without_router(self) -> None:
        topo = SpineLeafTopo(n_workers=1, with_router=False)
        hosts = sorted(topo.hosts())
        assert "router" not in hosts

    def test_constants(self) -> None:
        assert CLIENT_IP == "10.0.0.100"
        assert ROUTER_IP == "10.0.0.200"
