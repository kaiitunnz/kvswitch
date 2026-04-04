"""BMv2 simple_switch integration for Mininet.

Provides a ``BMv2Switch`` node class that starts ``simple_switch`` with
a compiled P4 JSON program.  Each switch instance gets a unique device
ID and thrift port for runtime table management via ``simple_switch_CLI``.
"""

import logging
import time

from mininet.node import Switch

logger = logging.getLogger(__name__)

# Base thrift port; each switch gets base + device_id.
_THRIFT_PORT_BASE = 9090
_next_device_id = 0


def _allocate_device_id() -> int:
    global _next_device_id
    did = _next_device_id
    _next_device_id += 1
    return did


def reset_device_ids() -> None:
    """Reset the global device ID counter (call between test runs)."""
    global _next_device_id
    _next_device_id = 0


class BMv2Switch(Switch):
    """Mininet switch backed by BMv2 ``simple_switch``.

    Parameters
    ----------
    name:
        Switch name (e.g. ``"s1"``).
    json_path:
        Path to the compiled P4 JSON program.
    thrift_port:
        If ``None``, auto-assigned as ``9090 + device_id``.
    log_console:
        If ``True``, simple_switch logs to the console.
    **kwargs:
        Passed to ``mininet.node.Switch``.
    """

    def __init__(
        self,
        name: str,
        json_path: str = "build/p4/kvswitch.json/kvswitch.json",
        thrift_port: int | None = None,
        log_console: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.json_path = json_path
        self.device_id = _allocate_device_id()
        self.thrift_port = thrift_port or (_THRIFT_PORT_BASE + self.device_id)
        self.log_console = log_console

    def start(self, controllers) -> None:  # type: ignore[override]
        """Start simple_switch for this node."""
        # Build interface mapping: -i port_num@intf_name
        intf_args = []
        for port_num, intf in enumerate(self.intfList()):
            if intf.name == "lo":
                continue
            intf_args.extend(["-i", f"{port_num}@{intf.name}"])

        log_arg = "--log-console" if self.log_console else ""
        cmd = (
            f"simple_switch"
            f" --device-id {self.device_id}"
            f" --thrift-port {self.thrift_port}"
            f" {log_arg}"
            f" {' '.join(intf_args)}"
            f" {self.json_path}"
            f" &"
        )
        logger.info(
            "Starting BMv2 switch %s (device %d, thrift %d)",
            self.name,
            self.device_id,
            self.thrift_port,
        )
        self.cmd(cmd)
        # Give simple_switch time to bind its thrift port.
        time.sleep(0.5)

    def stop(self, deleteIntfs: bool = True) -> None:  # type: ignore[override]
        """Stop simple_switch."""
        self.cmd("kill %simple_switch 2>/dev/null")
        super().stop(deleteIntfs)

    @property
    def cli_cmd(self) -> str:
        """Return the simple_switch_CLI command prefix for this switch."""
        return f"simple_switch_CLI --thrift-port {self.thrift_port}"

    def table_add(
        self, table: str, action: str, match: list[str], params: list[str]
    ) -> str:
        """Insert a table entry via simple_switch_CLI."""
        match_str = " ".join(match)
        params_str = " ".join(params)
        entry = f"table_add {table} {action} {match_str} => {params_str}"
        out = self.cmd(f'echo "{entry}" | {self.cli_cmd}')
        logger.debug("table_add on %s: %s -> %s", self.name, entry, out.strip())
        return out
