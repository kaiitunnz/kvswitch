"""BMv2 simple_switch_grpc integration for Mininet.

Provides a ``BMv2Switch`` node class that starts ``simple_switch_grpc``
with a compiled P4 JSON program.  Each switch instance gets a unique
device ID, thrift port, and **gRPC port** for P4Runtime table management
via Finsy.
"""

import logging
import shlex
import time
from pathlib import Path

from mininet.node import Switch

logger = logging.getLogger(__name__)

_THRIFT_PORT_BASE = 9090
_GRPC_PORT_BASE = 50000
_BMV2_STARTUP_TIMEOUT = 10.0
_BMV2_STARTUP_POLL = 0.1
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
    """Mininet switch backed by BMv2 ``simple_switch_grpc``.

    Parameters
    ----------
    name:
        Switch name (e.g. ``"s1"``).
    json_path:
        Path to the compiled P4 JSON program.
    thrift_port:
        If ``None``, auto-assigned as ``9090 + device_id``.
    grpc_port:
        If ``None``, auto-assigned as ``50000 + device_id``.
    log_console:
        If ``True``, simple_switch_grpc logs to the console.
    **kwargs:
        Passed to ``mininet.node.Switch``.
    """

    def __init__(
        self,
        name: str,
        json_path: str = "build/p4/kvswitch/kvswitch.json",
        thrift_port: int | None = None,
        grpc_port: int | None = None,
        log_console: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.json_path = str(Path(json_path).resolve())
        self.device_id = _allocate_device_id()
        self.thrift_port = thrift_port or (_THRIFT_PORT_BASE + self.device_id)
        self.grpc_port = grpc_port or (_GRPC_PORT_BASE + self.device_id)
        self.log_console = log_console
        self.log_path = f"/tmp/{self.name}-simple_switch_grpc.log"
        self.process_pid: int | None = None

    @property
    def grpc_addr(self) -> str:
        """Return the gRPC address for P4Runtime (Finsy) connections."""
        return f"127.0.0.1:{self.grpc_port}"

    def start(self, controllers) -> None:  # type: ignore[override]
        """Start simple_switch_grpc for this node."""
        intf_args = []
        for port_num, intf in enumerate(self.intfList()):
            if intf.name == "lo":
                continue
            intf_args.extend(["-i", f"{port_num}@{intf.name}"])

        log_arg = "--log-console" if self.log_console else ""
        switch_cmd = (
            f"simple_switch_grpc"
            f" --device-id {self.device_id}"
            f" --thrift-port {self.thrift_port}"
            f" {log_arg}"
            f" {' '.join(intf_args)}"
            f" {self.json_path}"
            f" -- --grpc-server-addr 0.0.0.0:{self.grpc_port}"
        )
        launcher = (
            f"rm -f {shlex.quote(self.log_path)}; "
            f"{switch_cmd} > {shlex.quote(self.log_path)} 2>&1 < /dev/null & echo $!"
        )
        logger.info(
            "Starting BMv2 switch %s (device %d, thrift %d, grpc %d)",
            self.name,
            self.device_id,
            self.thrift_port,
            self.grpc_port,
        )
        pid_output = self.cmd(f"bash -lc {shlex.quote(launcher)}")
        assert isinstance(pid_output, str)
        pid_output = pid_output.strip()
        pid_line = pid_output.splitlines()[-1] if pid_output else ""
        self.process_pid = int(pid_line) if pid_line.isdigit() else None
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _BMV2_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._grpc_ready():
                return
            if not self._process_alive():
                raise RuntimeError(
                    f"BMv2 switch {self.name} exited during startup. "
                    f"Log output:\n{self._read_log_excerpt()}"
                )
            time.sleep(_BMV2_STARTUP_POLL)

        raise RuntimeError(
            f"BMv2 switch {self.name} did not expose gRPC {self.grpc_port} "
            f"within {_BMV2_STARTUP_TIMEOUT:.1f}s. "
            f"Log output:\n{self._read_log_excerpt()}"
        )

    def _process_alive(self) -> bool:
        if self.process_pid is None:
            return False
        check = self.cmd(
            f"bash -lc {shlex.quote(f'kill -0 {self.process_pid} 2>/dev/null && echo alive || echo dead')}"
        )
        assert isinstance(check, str)
        return check.strip() == "alive"

    def _grpc_ready(self) -> bool:
        script = (
            f"kill -0 {self.process_pid} 2>/dev/null && "
            f"exec 3<>/dev/tcp/127.0.0.1/{self.grpc_port} && "
            "echo ready || echo waiting"
        )
        ready = self.cmd(f"bash -lc {shlex.quote(script)}")
        assert isinstance(ready, str)
        return ready.strip() == "ready"

    def _read_log_excerpt(self) -> str:
        log = self.cmd(
            "bash -lc "
            + shlex.quote(
                f"if test -f {shlex.quote(self.log_path)}; "
                f"then tail -n 40 {shlex.quote(self.log_path)}; fi"
            )
        )
        assert isinstance(log, str)
        return log.strip()

    def stop(self, deleteIntfs: bool = True) -> None:  # type: ignore[override]
        """Stop simple_switch_grpc."""
        if self.process_pid is not None:
            self.cmd(f"kill {self.process_pid} 2>/dev/null || true")
        self.cmd("kill %simple_switch_grpc 2>/dev/null || true")
        super().stop(deleteIntfs)

    @property
    def cli_cmd(self) -> str:
        """Return the simple_switch_CLI command prefix for this switch."""
        return f"simple_switch_CLI --thrift-port {self.thrift_port}"

    def table_add(
        self, table: str, action: str, match: list[str], params: list[str]
    ) -> str:
        """Insert a table entry via simple_switch_CLI (legacy helper)."""
        match_str = " ".join(match)
        params_str = " ".join(params)
        entry = f"table_add {table} {action} {match_str} => {params_str}"
        out = self.cmd(f'echo "{entry}" | {self.cli_cmd}')
        assert isinstance(out, str)
        logger.debug("table_add on %s: %s -> %s", self.name, entry, out.strip())
        return out
