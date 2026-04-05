"""Mininet topologies for KVSwitch evaluation.

**Legacy single-spine/single-leaf**::

              [router?]
                 |
    [client] --- [spine] --- [leaf] --- [worker0..N]

**Two-tier Clos (``ClosTopology``)**::

                   spine0     spine1
                  / | \\       / | \\
    ingress0 ---   |  \\----/  |   --- leaf0 --- w0, w1
                   |   \\     |       leaf1 --- w2, w3
    (router?) -----+    \\----+

Run with: ``sudo python -m kvswitch.network.topology``
"""

import logging

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.topo import Topo

logger = logging.getLogger(__name__)

# --- Legacy flat addressing (single spine/leaf) ---

CLIENT_IP = "10.0.0.100"
ROUTER_IP = "10.0.0.200"
KVSWITCH_SERVICE_IP = "10.0.0.250"
KVSWITCH_SERVICE_MAC = "02:00:00:00:00:fa"


def worker_ip(idx: int) -> str:
    """Worker IP in legacy flat ``10.0.0.0/24`` addressing."""
    return f"10.0.0.{idx + 1}"


# --- Clos per-leaf subnet addressing ---

CLOS_INGRESS_SUBNET = "10.1.0"
CLOS_WORKER_LEAF_SUBNET_PREFIX = "10.2"
CLOS_CLIENT_IP = f"{CLOS_INGRESS_SUBNET}.100"
CLOS_ROUTER_IP = f"{CLOS_INGRESS_SUBNET}.200"


def clos_worker_ip(leaf_idx: int, worker_idx: int) -> str:
    """Worker IP on a Clos worker leaf: ``10.2.{leaf_idx}.{worker_idx+1}``."""
    return f"{CLOS_WORKER_LEAF_SUBNET_PREFIX}.{leaf_idx}.{worker_idx + 1}"


def clos_worker_subnet(leaf_idx: int) -> str:
    """Subnet prefix for a Clos worker leaf: ``10.2.{leaf_idx}``."""
    return f"{CLOS_WORKER_LEAF_SUBNET_PREFIX}.{leaf_idx}"


class SpineLeafTopo(Topo):
    """Single-spine, single-leaf topology with optional L7 router host.

    Parameters (passed via ``build``):
        n_workers:    Number of worker hosts attached to the leaf switch.
        with_router:  If True, add an L7 router host attached to the spine.
    """

    def build(  # type: ignore[override]
        self,
        n_workers: int = 1,
        with_router: bool = False,
        delay: str | None = None,
    ) -> None:
        link_opts: dict = {}
        if delay is not None:
            link_opts["delay"] = delay  # e.g. "1ms", "10ms"

        spine = self.addSwitch("s1")
        leaf = self.addSwitch("s2")
        self.addLink(spine, leaf, **link_opts)

        # Client host attached to the spine.
        self.addHost("client", ip=f"{CLIENT_IP}/24")
        self.addLink("client", spine, **link_opts)

        # Optional L7 router host attached to the spine.
        if with_router:
            self.addHost("router", ip=f"{ROUTER_IP}/24")
            self.addLink("router", spine, **link_opts)

        # Worker hosts attached to the leaf.
        for i in range(n_workers):
            self.addHost(f"worker{i}", ip=f"{worker_ip(i)}/24")
            self.addLink(f"worker{i}", leaf, **link_opts)


class BMv2SpineLeafTopo(Topo):
    """Spine-leaf topology using BMv2 switches running ``kvswitch.p4``.

    All baselines share this topology.  For L4 RR the TCAM tables are
    empty so traffic hits ECMP fallback.  For KVSwitch the SDN controller
    populates TCAM rules and uses a dedicated control-plane network
    (see :class:`kvswitch.network.control_plane.ControlPlane`).

    Parameters (passed via ``build``):
        n_workers:       Number of worker hosts attached to the leaf.
        with_router:     Add an L7 router host (for the L7 baseline).
        delay:           Per-link latency string, e.g. ``"0.1ms"``.
    """

    def build(  # type: ignore[override]
        self,
        n_workers: int = 4,
        with_router: bool = False,
        delay: str | None = None,
    ) -> None:
        link_opts: dict = {}
        if delay is not None:
            link_opts["delay"] = delay

        spine = self.addSwitch("s1")
        leaf = self.addSwitch("s2")
        self.addLink(spine, leaf, **link_opts)

        self.addHost("client", ip=f"{CLIENT_IP}/24")
        self.addLink("client", spine, **link_opts)

        if with_router:
            self.addHost("router", ip=f"{ROUTER_IP}/24")
            self.addLink("router", spine, **link_opts)

        for i in range(n_workers):
            self.addHost(f"worker{i}", ip=f"{worker_ip(i)}/24")
            self.addLink(f"worker{i}", leaf, **link_opts)


class ClosTopology(Topo):
    """Parameterized two-tier Clos (leaf-spine) topology for BMv2.

    Models a datacenter fabric with:

    - *n_spines* spine switches (full-mesh to all leaves)
    - *n_worker_leaves* worker-leaf switches, each with *workers_per_leaf* workers
    - *n_ingress_leaves* ingress-leaf switches carrying client/router hosts
    - Per-leaf /24 subnets for clean L3 routing

    Switch naming:
        ``spine0..N``, ``leaf0..M``, ``ingress0..K``

    Host naming:
        ``worker0..W`` (global index), ``client``, ``router``
    """

    def build(  # type: ignore[override]
        self,
        n_spines: int = 2,
        n_worker_leaves: int = 2,
        workers_per_leaf: int = 2,
        n_ingress_leaves: int = 1,
        with_router: bool = False,
        delay: str | None = None,
    ) -> None:
        link_opts: dict = {}
        if delay is not None:
            link_opts["delay"] = delay

        # --- Spine switches ---
        spines = [self.addSwitch(f"spine{i}") for i in range(n_spines)]

        # --- Ingress leaves ---
        for i in range(n_ingress_leaves):
            ingress = self.addSwitch(f"ingress{i}")
            for sp in spines:
                self.addLink(ingress, sp, **link_opts)

        # Client on ingress0.
        self.addHost("client", ip=f"{CLOS_CLIENT_IP}/24")
        self.addLink("client", "ingress0", **link_opts)

        if with_router:
            self.addHost("router", ip=f"{CLOS_ROUTER_IP}/24")
            self.addLink("router", "ingress0", **link_opts)

        # --- Worker leaves ---
        global_worker_idx = 0
        for leaf_idx in range(n_worker_leaves):
            leaf = self.addSwitch(f"leaf{leaf_idx}")
            # Full mesh: every worker leaf connects to every spine.
            for sp in spines:
                self.addLink(leaf, sp, **link_opts)
            # Workers under this leaf.
            for w in range(workers_per_leaf):
                name = f"worker{global_worker_idx}"
                ip = clos_worker_ip(leaf_idx, w)
                self.addHost(name, ip=f"{ip}/24")
                self.addLink(name, leaf, **link_opts)
                global_worker_idx += 1


def run_mininet(
    n_workers: int = 1,
    with_router: bool = False,
    delay: str | None = None,
) -> None:
    """Start the Mininet network and drop into the CLI."""
    topo = SpineLeafTopo(n_workers=n_workers, with_router=with_router, delay=delay)
    net = Mininet(topo=topo, switch=OVSBridge, link=TCLink, controller=None)
    net.start()
    logger.info(
        "Topology started: 1 spine, 1 leaf, %d worker(s), router=%s",
        n_workers,
        with_router,
    )
    logger.info("Client IP: %s", CLIENT_IP)
    if with_router:
        logger.info("Router IP: %s", ROUTER_IP)
    for i in range(n_workers):
        logger.info("Worker%d IP: %s", i, worker_ip(i))

    CLI(net)
    net.stop()


if __name__ == "__main__":
    import argparse

    from kvswitch.utils.logger import setup_logging

    parser = argparse.ArgumentParser(description="KVSwitch Mininet topology")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--with-router", action="store_true")
    parser.add_argument(
        "--delay",
        type=str,
        default=None,
        help="Per-link latency, e.g. '1ms', '10ms'",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_mininet(
        n_workers=args.n_workers, with_router=args.with_router, delay=args.delay
    )
