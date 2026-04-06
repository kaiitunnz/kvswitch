"""Mininet Clos topology for KVSwitch evaluation.

Two-tier leaf-spine fabric::

                   spine0     spine1
                  / | \\       / | \\
    ingress0 ---   |  \\----/  |   --- leaf0 --- w0, w1
                   |   \\     |       leaf1 --- w2, w3
    (router?) -----+    \\----+

Run interactively: ``sudo python -m kvswitch.network.topology``
"""

import logging

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.topo import Topo

logger = logging.getLogger(__name__)

# --- Clos per-leaf subnet addressing ---

KVSWITCH_SERVICE_IP = "10.0.0.250"
KVSWITCH_SERVICE_MAC = "02:00:00:00:00:fa"

CLOS_INGRESS_SUBNET = "10.1.0"
CLOS_WORKER_LEAF_SUBNET_PREFIX = "10.2"
CLOS_CLIENT_IP = f"{CLOS_INGRESS_SUBNET}.100"
CLOS_ROUTER_IP = f"{CLOS_INGRESS_SUBNET}.200"


def clos_worker_ip(leaf_idx: int, worker_idx: int) -> str:
    """Worker IP on a Clos worker leaf: ``10.2.{leaf_idx}.{worker_idx+1}``."""
    return f"{CLOS_WORKER_LEAF_SUBNET_PREFIX}.{leaf_idx}.{worker_idx + 1}"


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


def run_cli(
    n_spines: int = 2,
    n_worker_leaves: int = 2,
    workers_per_leaf: int = 4,
    with_router: bool = False,
    delay: str | None = None,
) -> None:
    """Start a Clos topology and drop into the Mininet CLI."""
    topo = ClosTopology(
        n_spines=n_spines,
        n_worker_leaves=n_worker_leaves,
        workers_per_leaf=workers_per_leaf,
        with_router=with_router,
        delay=delay,
    )
    net = Mininet(topo=topo, switch=OVSBridge, link=TCLink, controller=None)
    net.start()
    logger.info(
        "Clos topology: %d spines, %d worker leaves x %d workers, router=%s",
        n_spines,
        n_worker_leaves,
        workers_per_leaf,
        with_router,
    )
    logger.info("Client IP: %s", CLOS_CLIENT_IP)
    if with_router:
        logger.info("Router IP: %s", CLOS_ROUTER_IP)
    for leaf_idx in range(n_worker_leaves):
        for w in range(workers_per_leaf):
            global_idx = leaf_idx * workers_per_leaf + w
            logger.info("Worker%d IP: %s", global_idx, clos_worker_ip(leaf_idx, w))

    CLI(net)
    net.stop()


if __name__ == "__main__":
    import argparse

    from kvswitch.utils.logger import setup_logging

    parser = argparse.ArgumentParser(description="KVSwitch Mininet Clos topology")
    parser.add_argument("--n-spines", type=int, default=2)
    parser.add_argument("--n-worker-leaves", type=int, default=2)
    parser.add_argument("--workers-per-leaf", type=int, default=4)
    parser.add_argument("--with-router", action="store_true")
    parser.add_argument("--delay", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_cli(
        n_spines=args.n_spines,
        n_worker_leaves=args.n_worker_leaves,
        workers_per_leaf=args.workers_per_leaf,
        with_router=args.with_router,
        delay=args.delay,
    )
