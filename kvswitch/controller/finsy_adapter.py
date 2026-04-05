"""Finsy-based P4Runtime switch adapter for BMv2 switches.

Receives structured :class:`SwitchOp` objects from the SDN controller
and translates them directly into Finsy ``P4TableEntry`` operations —
no regex, no string parsing.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import finsy

from kvswitch.controller.switch_adapter import (
    SwitchOp,
    TableAddOp,
    TableClearOp,
    TableDeleteOp,
)

logger = logging.getLogger(__name__)


def _normalize_match(match: dict[str, str | int]) -> dict[str, str | int]:
    """Translate controller match values into the format expected by Finsy.

    The controller still emits BMv2-style ternary strings (``0xVAL&&&0xMASK``)
    because the in-memory adapter and earlier BMv2 CLI path used that form.
    Finsy expects ternary masks in ``value/&mask`` form instead, so normalize
    them here at the P4Runtime boundary.
    """
    normalized: dict[str, str | int] = {}
    for field_name, value in match.items():
        if isinstance(value, str) and "&&&" in value:
            ternary_value, ternary_mask = value.split("&&&", maxsplit=1)
            normalized[field_name] = f"{ternary_value}/&{ternary_mask}"
        else:
            normalized[field_name] = value
    return normalized


@dataclass
class FinsyAdapter:
    """Applies structured table operations to BMv2 switches via Finsy P4Runtime.

    Parameters
    ----------
    switches:
        Mapping of canonical switch names (e.g. ``"s1"``, ``"s2"``) to
        their gRPC addresses (e.g. ``"127.0.0.1:50000"``).
    p4info_path:
        Path to the P4Info text protobuf file.
    p4blob_path:
        Path to the compiled P4 JSON blob.
    device_ids:
        Optional mapping of canonical switch names to BMv2 device IDs.
    """

    switches: dict[str, str]  # canonical name → grpc_addr
    p4info_path: Path
    p4blob_path: Path
    device_ids: dict[str, int] = field(default_factory=dict)
    _finsy_switches: dict[str, finsy.Switch] = field(
        default_factory=dict, init=False, repr=False
    )
    _controller: finsy.Controller | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(
        default=None, init=False, repr=False
    )
    _ready: dict[str, asyncio.Event] = field(
        default_factory=dict, init=False, repr=False
    )
    _write_tails: dict[str, asyncio.Task[None]] = field(
        default_factory=dict, init=False, repr=False
    )
    _background_tasks: set[asyncio.Task[None]] = field(
        default_factory=set, init=False, repr=False
    )

    async def start(self) -> None:
        """Connect to all switches and push the P4 pipeline."""
        self._loop = asyncio.get_running_loop()

        for name, addr in self.switches.items():
            ready_event = asyncio.Event()
            self._ready[name] = ready_event

            async def _ready_handler(
                sw: finsy.Switch, _evt: asyncio.Event = ready_event
            ) -> None:
                _evt.set()
                logger.info("Finsy: switch %s ready", sw.name)
                await asyncio.Event().wait()

            sw = finsy.Switch(
                name,
                addr,
                finsy.SwitchOptions(
                    p4info=self.p4info_path,
                    p4blob=self.p4blob_path,
                    device_id=self.device_ids.get(name, 0),
                    ready_handler=_ready_handler,
                ),
            )
            self._finsy_switches[name] = sw

        self._controller = finsy.Controller(list(self._finsy_switches.values()))
        asyncio.ensure_future(self._controller.run())

        for name, evt in self._ready.items():
            try:
                await asyncio.wait_for(evt.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Finsy: switch %s not ready after 10s", name)

        logger.info("FinsyAdapter connected to %d switches", len(self.switches))

    def close(self) -> None:
        """Disconnect from all switches."""
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        if self._controller is not None:
            self._controller.stop()

    def apply_ops(self, ops: Sequence[SwitchOp]) -> None:
        """Apply structured operations to the appropriate switches.

        When called from outside the Finsy event loop, submits the write
        to that loop and waits for completion.
        """
        if not ops or self._loop is None:
            return

        # Group ops by switch.
        by_switch: dict[str, list[SwitchOp]] = defaultdict(list)
        for op in ops:
            by_switch[op.switch].append(op)

        for switch_name, switch_ops in by_switch.items():
            sw = self._finsy_switches.get(switch_name)
            if sw is None:
                continue
            entities = [self._to_finsy(op) for op in switch_ops]

            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is self._loop:
                task = asyncio.create_task(self._write_serialized(sw, entities))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    self._write_serialized(sw, entities), self._loop
                )
                future.result(timeout=5.0)

    async def _write_serialized(self, sw: finsy.Switch, entities: list[Any]) -> None:
        current_task = asyncio.current_task()
        if current_task is None:
            await self._write(sw, entities)
            return

        previous_task = self._write_tails.get(sw.name)
        self._write_tails[sw.name] = current_task

        if previous_task is not None and previous_task is not current_task:
            try:
                await asyncio.shield(previous_task)
            except Exception:
                logger.debug(
                    "Previous write task for %s failed before queued update applied",
                    sw.name,
                    exc_info=True,
                )

        try:
            await self._write(sw, entities)
        finally:
            if self._write_tails.get(sw.name) is current_task:
                self._write_tails.pop(sw.name, None)

    async def _write(self, sw: finsy.Switch, entities: list[Any]) -> None:
        try:
            pending: list[Any] = []
            for entity in entities:
                if isinstance(entity, _TableClearSentinel):
                    if pending:
                        await sw.write(pending)
                        pending = []
                    await sw.delete_many([finsy.P4TableEntry(entity.table_id)])
                    continue
                pending.append(entity)

            if pending:
                await sw.write(pending)
        except Exception:
            logger.warning("Finsy write to %s failed", sw.name, exc_info=True)

    def _to_finsy(self, op: SwitchOp) -> Any:
        """Convert a structured SwitchOp to a Finsy entity."""
        match op:
            case TableClearOp():
                return _TableClearSentinel(op.table)
            case TableAddOp():
                return +finsy.P4TableEntry(
                    table_id=op.table,
                    match=finsy.P4TableMatch(_normalize_match(op.match)),
                    action=finsy.P4TableAction(op.action, **op.action_params),
                    priority=op.priority,
                )
            case TableDeleteOp():
                return -finsy.P4TableEntry(
                    table_id=op.table,
                    match=finsy.P4TableMatch(_normalize_match(op.match)),
                    priority=op.priority,
                )
            case _:
                raise ValueError(f"Unknown op type: {type(op)}")


@dataclass(frozen=True)
class _TableClearSentinel:
    """Internal sentinel for table_clear — handled specially in _write."""

    table_id: str
