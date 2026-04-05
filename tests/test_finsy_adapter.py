"""Tests for kvswitch.controller.finsy_adapter — structured op to Finsy entity conversion."""

import asyncio
from pathlib import Path

from kvswitch.controller.finsy_adapter import (
    FinsyAdapter,
    _normalize_match,
    _TableClearSentinel,
)
from kvswitch.controller.switch_adapter import TableAddOp, TableClearOp, TableDeleteOp


def _make_adapter() -> FinsyAdapter:
    return FinsyAdapter(
        switches={},
        p4info_path=Path("dummy.p4info.txtpb"),
        p4blob_path=Path("dummy.json"),
    )


class TestToFinsy:
    def test_normalize_match_converts_bmv2_ternary_to_finsy_format(self) -> None:
        assert _normalize_match({"hdr.kvswitch.h0": "0xABCD1234&&&0xffffffff"}) == {
            "hdr.kvswitch.h0": "0xABCD1234/&0xffffffff"
        }

    def test_table_add_ternary(self) -> None:
        adapter = _make_adapter()
        op = TableAddOp(
            switch="s1",
            table="spine_prefix_route",
            action="route_to_pod",
            match={"hdr.kvswitch.h0": "0xABCD1234&&&0xffffffff"},
            action_params={"port": 3},
            priority=1,
        )
        entity = adapter._to_finsy(op)
        assert entity is not None
        assert entity.match["hdr.kvswitch.h0"] == "0xABCD1234/&0xffffffff"

    def test_table_add_leaf(self) -> None:
        adapter = _make_adapter()
        op = TableAddOp(
            switch="s2",
            table="leaf_prefix_route",
            action="route_to_worker",
            match={
                "hdr.kvswitch.h0": "0x00000001&&&0xffffffff",
                "hdr.kvswitch.h1": "0x00000002&&&0xffffffff",
                "hdr.kvswitch.h2": "0x00000003&&&0xffffffff",
            },
            action_params={"port": 5, "dst_mac": 0x020000000005},
            priority=1,
        )
        entity = adapter._to_finsy(op)
        assert entity is not None
        assert entity.match == {
            "hdr.kvswitch.h0": "0x00000001/&0xffffffff",
            "hdr.kvswitch.h1": "0x00000002/&0xffffffff",
            "hdr.kvswitch.h2": "0x00000003/&0xffffffff",
        }

    def test_table_add_ecmp(self) -> None:
        adapter = _make_adapter()
        op = TableAddOp(
            switch="s1",
            table="spine_ecmp_select",
            action="route_to_pod",
            match={"meta.ecmp_bucket": 7},
            action_params={"port": 2},
        )
        entity = adapter._to_finsy(op)
        assert entity is not None

    def test_table_add_lpm(self) -> None:
        adapter = _make_adapter()
        op = TableAddOp(
            switch="s1",
            table="ipv4_lpm",
            action="forward",
            match={"hdr.ipv4.dstAddr": "10.0.0.1/32"},
            action_params={"port": 1},
        )
        entity = adapter._to_finsy(op)
        assert entity is not None

    def test_table_delete(self) -> None:
        adapter = _make_adapter()
        op = TableDeleteOp(
            switch="s1",
            table="spine_prefix_route",
            match={"hdr.kvswitch.h0": "0xABCD1234&&&0xffffffff"},
            priority=1,
        )
        entity = adapter._to_finsy(op)
        assert entity is not None

    def test_table_clear(self) -> None:
        adapter = _make_adapter()
        op = TableClearOp(switch="s1", table="spine_ecmp_select")
        entity = adapter._to_finsy(op)
        assert isinstance(entity, _TableClearSentinel)
        assert entity.table_id == "spine_ecmp_select"


class TestWrite:
    def test_write_clears_then_adds(self) -> None:
        adapter = _make_adapter()
        clear = adapter._to_finsy(TableClearOp(switch="s2", table="leaf_ecmp_select"))
        add = adapter._to_finsy(
            TableAddOp(
                switch="s2",
                table="leaf_ecmp_select",
                action="route_to_worker",
                match={"meta.ecmp_bucket": 7},
                action_params={"port": 2, "dst_mac": 0x020000000002},
            )
        )

        class FakeSwitch:
            name = "s2"

            def __init__(self) -> None:
                self.calls: list[tuple[str, list]] = []

            async def delete_many(self, entities) -> None:
                self.calls.append(("delete_many", list(entities)))

            async def write(self, updates) -> None:
                self.calls.append(("write", list(updates)))

        fake = FakeSwitch()
        asyncio.run(adapter._write(fake, [clear, add]))  # type: ignore[arg-type]

        assert [call[0] for call in fake.calls] == ["delete_many", "write"]
        assert len(fake.calls[0][1]) == 1
        assert fake.calls[0][1][0].table_id == "leaf_ecmp_select"
        assert len(fake.calls[1][1]) == 1

    def test_write_serialized_per_switch(self) -> None:
        adapter = _make_adapter()
        first = ["first"]
        second = ["second"]

        class FakeSwitch:
            name = "s2"

            def __init__(self) -> None:
                self.events: list[str] = []

            async def delete_many(self, entities) -> None:
                raise AssertionError("delete_many should not be called")

            async def write(self, updates) -> None:
                label = list(updates)[0]
                self.events.append(f"start-{label}")
                await asyncio.sleep(0.01)
                self.events.append(f"end-{label}")

        async def _run() -> list[str]:
            fake = FakeSwitch()
            await asyncio.gather(
                adapter._write_serialized(fake, first),
                adapter._write_serialized(fake, second),
            )
            return fake.events

        events = asyncio.run(_run())  # type: ignore[arg-type]
        assert events == ["start-first", "end-first", "start-second", "end-second"]


class TestApplyOps:
    def test_apply_ops_no_switches(self) -> None:
        adapter = _make_adapter()
        adapter.apply_ops(
            [
                TableAddOp(
                    switch="nonexistent",
                    table="ipv4_lpm",
                    action="forward",
                    match={"hdr.ipv4.dstAddr": "10.0.0.1/32"},
                    action_params={"port": 1},
                )
            ]
        )
