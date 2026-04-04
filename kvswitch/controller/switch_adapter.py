"""Structured switch adapter protocol and data types.

Replaces the old string-based protocol with typed
operations so that adapters never need to parse command strings.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class TableAddOp:
    """A table_add operation with structured fields."""

    switch: str
    table: str
    action: str
    match: dict[str, str | int]
    """field_name → value (ternary "0xVAL&&&0xMASK", exact int, LPM "ip/prefix")"""
    action_params: dict[str, int]
    priority: int = 0


@dataclass(frozen=True)
class TableDeleteOp:
    """A table_delete operation with structured fields."""

    switch: str
    table: str
    match: dict[str, str | int]
    priority: int = 0


@dataclass(frozen=True)
class TableClearOp:
    """A table_clear operation."""

    switch: str
    table: str


# Union of all operation types.
type SwitchOp = TableAddOp | TableDeleteOp | TableClearOp


class SwitchAdapter(Protocol):
    """Applies structured table operations to switches."""

    def apply_ops(self, ops: list[SwitchOp]) -> None: ...


@dataclass
class InMemorySwitchAdapter:
    """Test adapter that records all operations."""

    ops: list[SwitchOp] = field(default_factory=list)

    def apply_ops(self, ops: list[SwitchOp]) -> None:
        self.ops.extend(ops)

    def ops_for_switch(self, switch: str) -> list[SwitchOp]:
        return [op for op in self.ops if op.switch == switch]

    def table_adds(
        self, switch: str | None = None, table: str | None = None
    ) -> list[TableAddOp]:
        return [
            op
            for op in self.ops
            if isinstance(op, TableAddOp)
            and (switch is None or op.switch == switch)
            and (table is None or op.table == table)
        ]

    def table_deletes(
        self, switch: str | None = None, table: str | None = None
    ) -> list[TableDeleteOp]:
        return [
            op
            for op in self.ops
            if isinstance(op, TableDeleteOp)
            and (switch is None or op.switch == switch)
            and (table is None or op.table == table)
        ]

    def table_clears(
        self, switch: str | None = None, table: str | None = None
    ) -> list[TableClearOp]:
        return [
            op
            for op in self.ops
            if isinstance(op, TableClearOp)
            and (switch is None or op.switch == switch)
            and (table is None or op.table == table)
        ]
