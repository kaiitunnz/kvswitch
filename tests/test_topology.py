"""Tests for kvswitch.network.topology — Topo graph structure only (no sudo)."""

from kvswitch.network.topology import (
    CLOS_CLIENT_IP,
    CLOS_ROUTER_IP,
    ClosTopology,
    clos_worker_ip,
)


class TestClosWorkerIp:
    def test_first_leaf_first_worker(self) -> None:
        assert clos_worker_ip(0, 0) == "10.2.0.1"

    def test_second_leaf(self) -> None:
        assert clos_worker_ip(1, 2) == "10.2.1.3"


class TestClosTopology:
    def test_default_hosts(self) -> None:
        topo = ClosTopology(n_spines=2, n_worker_leaves=2, workers_per_leaf=2)
        hosts = sorted(topo.hosts())
        assert hosts == ["client", "worker0", "worker1", "worker2", "worker3"]

    def test_switches(self) -> None:
        topo = ClosTopology(n_spines=2, n_worker_leaves=2, workers_per_leaf=2)
        switches = sorted(topo.switches())
        assert switches == ["ingress0", "leaf0", "leaf1", "spine0", "spine1"]

    def test_with_router(self) -> None:
        topo = ClosTopology(
            n_spines=1, n_worker_leaves=1, workers_per_leaf=1, with_router=True
        )
        assert "router" in topo.hosts()

    def test_without_router(self) -> None:
        topo = ClosTopology(
            n_spines=1, n_worker_leaves=1, workers_per_leaf=1, with_router=False
        )
        assert "router" not in topo.hosts()

    def test_worker_count(self) -> None:
        topo = ClosTopology(n_spines=2, n_worker_leaves=3, workers_per_leaf=4)
        workers = [h for h in topo.hosts() if h.startswith("worker")]
        assert len(workers) == 12

    def test_full_mesh_links(self) -> None:
        topo = ClosTopology(n_spines=2, n_worker_leaves=2, workers_per_leaf=1)
        # Each worker leaf links to 2 spines = 4 spine-leaf links
        # Ingress links to 2 spines = 2 ingress-spine links
        # 1 client link + 2 worker links = 3 host links
        # Total: 4 + 2 + 3 = 9
        assert len(topo.links()) == 9

    def test_constants(self) -> None:
        assert CLOS_CLIENT_IP == "10.1.0.100"
        assert CLOS_ROUTER_IP == "10.1.0.200"
