"""Mininet topologies for KVSwitch evaluation.

**OVS topology (Direct / L7 router)**::

    [client] --- [spine] --- [leaf] --- [worker0]

**BMv2 topology (all baselines — P4 switches)**::

              [router?]
                 |
    [client] --- [spine] --- [leaf] --- [worker0..N]

Run with: ``sudo python -m kvswitch.network.topology``
"""

import logging

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.topo import Topo

logger = logging.getLogger(__name__)

CLIENT_IP = "10.0.0.100"
ROUTER_IP = "10.0.0.200"
KVSWITCH_SERVICE_IP = "10.0.0.250"
KVSWITCH_SERVICE_MAC = "02:00:00:00:00:fa"


def worker_ip(idx: int) -> str:
    return f"10.0.0.{idx + 1}"


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
