"""Dedicated control-plane management network for KVSwitch evaluation.

Creates a Linux bridge in root namespace with direct veth pairs to each
Mininet worker namespace, modelling a separate management fabric that
bypasses the BMv2 data-plane switches.  In a real deployment this
corresponds to a dedicated management ToR or BMC network used for
P4Runtime / OpenFlow control traffic.

Usage::

    from kvswitch.network.control_plane import ControlPlane

    with ControlPlane(net, n_workers=4) as cp:
        controller = SDNController(host=cp.controller_ip, ...)
        ...
"""

import logging
import subprocess
from typing import Any, cast

from mininet.net import Mininet
from mininet.node import Node

logger = logging.getLogger(__name__)

DEFAULT_SUBNET = "192.168.100"
DEFAULT_CONTROLLER_IP = f"{DEFAULT_SUBNET}.1"
DEFAULT_BRIDGE = "oob-br0"


class ControlPlane:
    """Context manager for the control-plane management network.

    On enter, creates a Linux bridge in root namespace with a single
    controller IP and one veth pair per worker connecting each worker's
    Mininet namespace to the bridge.  On exit — including on error — the
    bridge and all attached interfaces are cleaned up.

    Parameters
    ----------
    net:
        A running Mininet instance whose worker hosts have been started
        (their PIDs must be available for namespace assignment).
    n_workers:
        Number of workers to connect to the control plane.
    subnet:
        The ``/24`` subnet prefix (e.g. ``"192.168.100"``).  The
        controller gets ``.1`` and workers get ``.10``, ``.11``, etc.
    controller_ip:
        IP address for the controller on the bridge.  Defaults to
        ``{subnet}.1``.
    bridge:
        Name of the Linux bridge device in root namespace.
    """

    def __init__(
        self,
        net: Mininet,
        n_workers: int,
        subnet: str = DEFAULT_SUBNET,
        controller_ip: str | None = None,
        bridge: str = DEFAULT_BRIDGE,
    ) -> None:
        self._net = net
        self._n_workers = n_workers
        self._subnet = subnet
        self._controller_ip = controller_ip or f"{subnet}.1"
        self._bridge = bridge

    @property
    def controller_ip(self) -> str:
        """IP address the SDN controller should bind to."""
        return self._controller_ip

    def worker_ip(self, idx: int) -> str:
        """Return the control-plane IP for worker *idx*."""
        return f"{self._subnet}.{10 + idx}"

    def __enter__(self) -> "ControlPlane":
        self._setup()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove a bridge and sweep any orphaned ``oob-r-*`` veths."""
        subprocess.run(["ip", "link", "del", self._bridge], capture_output=True)
        result = subprocess.run(
            ["ip", "-o", "link", "show"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            for token in line.split():
                name = token.rstrip(":")
                if name.startswith("oob-r-"):
                    subprocess.run(["ip", "link", "del", name], capture_output=True)
                    break

    def _setup(self) -> None:
        created_root_intfs: list[str] = []
        bridge = self._bridge
        controller_ip = self._controller_ip
        try:
            self._cleanup()
            subprocess.run(
                ["ip", "link", "add", bridge, "type", "bridge"],
                check=True,
            )
            subprocess.run(
                [
                    "ip",
                    "addr",
                    "add",
                    f"{controller_ip}/24",
                    "dev",
                    bridge,
                ],
                check=True,
            )
            subprocess.run(
                ["ip", "link", "set", bridge, "up"],
                check=True,
            )
            bridge_state = subprocess.run(
                ["ip", "addr", "show", "dev", bridge],
                check=True,
                capture_output=True,
                text=True,
            )
            if controller_ip not in bridge_state.stdout:
                raise RuntimeError(
                    f"failed to assign control-plane IP {controller_ip}" f" to {bridge}"
                )

            for i in range(self._n_workers):
                worker_host = cast(Node, self._net.get(f"worker{i}"))
                root_intf = f"oob-r-{i}"
                worker_intf = f"oob-w-{i}"
                ip_addr = self.worker_ip(i)

                subprocess.run(
                    [
                        "ip",
                        "link",
                        "add",
                        root_intf,
                        "type",
                        "veth",
                        "peer",
                        "name",
                        worker_intf,
                        "netns",
                        str(worker_host.pid),
                    ],
                    check=True,
                )
                created_root_intfs.append(root_intf)
                subprocess.run(
                    ["ip", "link", "set", root_intf, "master", bridge],
                    check=True,
                )
                subprocess.run(["ip", "link", "set", root_intf, "up"], check=True)
                worker_host.cmd(f"ip addr add {ip_addr}/24 dev {worker_intf}")
                worker_host.cmd(f"ip link set {worker_intf} up")
                ip_addr_info = worker_host.cmd(f"ip addr show dev {worker_intf}")
                assert isinstance(ip_addr_info, str)
                if ip_addr_info.find(ip_addr) == -1:
                    raise RuntimeError(
                        f"failed to configure control-plane interface"
                        f" {worker_intf} on worker{i}"
                    )
        except Exception:
            for intf in created_root_intfs:
                subprocess.run(["ip", "link", "del", intf], capture_output=True)
            self._cleanup()
            raise

        logger.info(
            "Created control-plane network %s (%s) with %d worker veths",
            bridge,
            controller_ip,
            self._n_workers,
        )
