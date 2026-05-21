"""Tests for collective_ai.core — RoomAddress, TMinusEvent, GapSeverity, GapSignal, FocusQueue, SimulationRoom."""

import time
import pytest
from collective_ai.core import (
    RoomKind, RoomAddress, TMinusEvent, GapSeverity, GapSignal, FocusQueue, SimulationRoom,
)


class TestRoomAddress:
    def test_room_id(self):
        addr = RoomAddress(kind=RoomKind.SENSOR, domain="drift", name="main")
        assert addr.room_id == "sensor:drift:main"

    def test_full_address(self):
        addr = RoomAddress(kind=RoomKind.PREDICTOR, domain="intent", name="router")
        assert addr.full_address == "predictor:intent:router"

    def test_parent(self):
        addr = RoomAddress(kind=RoomKind.SENSOR, domain="test", name="child")
        parent = addr.parent()
        assert parent.name == "test"

    def test_child(self):
        addr = RoomAddress(kind=RoomKind.SENSOR, domain="test", name="parent")
        child = addr.child("sub")
        assert child.name == "sub"

    def test_parse(self):
        addr = RoomAddress.parse("sensor:drift:main")
        assert addr.kind == RoomKind.SENSOR
        assert addr.domain == "drift"
        assert addr.name == "main"

    def test_str(self):
        addr = RoomAddress(kind=RoomKind.GAP, domain="x", name="y")
        assert str(addr) == "gap:x:y"


class TestTMinusEvent:
    def test_creation(self):
        event = TMinusEvent(room="test", event_time=time.time() + 60, data={"key": "val"})
        assert event.room == "test"

    def test_time_until_event(self):
        future = time.time() + 100
        event = TMinusEvent(room="test", event_time=future)
        assert event.time_until_event() > 50

    def test_is_expired_future(self):
        event = TMinusEvent(room="test", event_time=time.time() + 1000)
        assert not event.is_expired()

    def test_is_expired_past(self):
        event = TMinusEvent(room="test", event_time=time.time() - 10)
        assert event.is_expired()

    def test_to_dict_from_dict(self):
        event = TMinusEvent(room="test", event_time=1000.0, data={"x": 1})
        d = event.to_dict()
        restored = TMinusEvent.from_dict(d)
        assert restored.room == "test"


class TestGapSignal:
    def test_create(self):
        pred = TMinusEvent(room="test", event_time=time.time() + 60)
        gap = GapSignal.create(room="test", prediction=pred, actual={"x": 1}, delta=0.5)
        assert gap.room == "test"
        assert gap.delta == 0.5

    def test_to_dict(self):
        pred = TMinusEvent(room="test", event_time=1000.0)
        gap = GapSignal.create(room="test", prediction=pred, actual=None, delta=0.3)
        d = gap.to_dict()
        assert "room" in d


class TestFocusQueue:
    def test_add_and_top(self):
        fq = FocusQueue()
        pred = TMinusEvent(room="a", event_time=1000.0)
        gap = GapSignal.create(room="a", prediction=pred, actual=None, delta=0.5)
        fq.add(gap)
        top = fq.top(1)
        assert len(top) == 1

    def test_by_room(self):
        fq = FocusQueue()
        for room_name in ["a", "a", "b"]:
            pred = TMinusEvent(room=room_name, event_time=1000.0)
            gap = GapSignal.create(room=room_name, prediction=pred, actual=None, delta=0.5)
            fq.add(gap)
        assert len(fq.by_room("a")) == 2

    def test_clear_resolved(self):
        fq = FocusQueue()
        pred = TMinusEvent(room="test", event_time=1000.0)
        gap = GapSignal.create(room="test", prediction=pred, actual=None, delta=0.5)
        fq.add(gap)
        fq.clear_resolved({gap.gap_id})
        assert len(fq) == 0

    def test_summary(self):
        fq = FocusQueue()
        s = fq.summary()
        assert isinstance(s, dict)
        assert "total_gaps" in s

    def test_len(self):
        fq = FocusQueue()
        assert len(fq) == 0


class TestSimulationRoom:
    def test_creation(self):
        room = SimulationRoom(name="test-room", kind=RoomKind.SENSOR)
        assert room.name == "test-room"

    def test_repr(self):
        room = SimulationRoom(name="test", kind=RoomKind.SENSOR)
        r = repr(room)
        assert "test" in r
