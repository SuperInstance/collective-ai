"""Tests for collective-ai."""

import time
import pytest
from collective_ai import (
    RoomKind, RoomAddress, TMinusEvent, GapSeverity, GapSignal, FocusQueue, SimulationRoom,
)


class TestRoomAddress:
    def test_parse_and_str(self):
        addr = RoomAddress.parse("forgemaster@eileen/drift-detect/predictor")
        assert addr.instance == "forgemaster@eileen"
        assert addr.path == ["drift-detect", "predictor"]
        assert str(addr) == "forgemaster@eileen/drift-detect/predictor"

    def test_parent(self):
        addr = RoomAddress.parse("inst/a/b/c")
        parent = addr.parent()
        assert parent is not None
        assert parent.room_id == "a/b"
        assert parent.parent().parent() is None  # root has no parent

    def test_child(self):
        addr = RoomAddress(instance="inst", path=["root"])
        child = addr.child("leaf")
        assert child.full_address == "inst/root/leaf"


class TestTMinusEvent:
    def test_expired_event(self):
        event = TMinusEvent(
            predictor="test", event_type="test", predicted_value=1.0,
            confidence=0.9, predicted_at=time.time(), event_time=time.time() - 1,
        )
        assert event.is_expired

    def test_future_event_not_expired(self):
        event = TMinusEvent(
            predictor="test", event_type="test", predicted_value=1.0,
            confidence=0.9, predicted_at=time.time(), event_time=time.time() + 60,
        )
        assert not event.is_expired

    def test_roundtrip_dict(self):
        event = TMinusEvent(
            predictor="test", event_type="test", predicted_value=42,
            confidence=0.8, predicted_at=1000.0, event_time=1060.0, context={"foo": "bar"},
        )
        d = event.to_dict()
        restored = TMinusEvent.from_dict(d)
        assert restored.predicted_value == 42
        assert restored.confidence == 0.8


class TestGapSignal:
    def test_low_severity(self):
        pred = TMinusEvent(
            predictor="r", event_type="e", predicted_value=1.0,
            confidence=0.9, predicted_at=time.time(), event_time=time.time() + 60,
        )
        gap = GapSignal.create(room="r", prediction=pred, actual=1.05, delta=0.05)
        assert gap.severity == GapSeverity.LOW

    def test_critical_severity(self):
        pred = TMinusEvent(
            predictor="r", event_type="e", predicted_value=1.0,
            confidence=0.9, predicted_at=time.time(), event_time=time.time() + 60,
        )
        gap = GapSignal.create(room="r", prediction=pred, actual=5.0, delta=0.96)
        assert gap.severity == GapSeverity.CRITICAL

    def test_focus_score_is_confidence_times_delta(self):
        pred = TMinusEvent(
            predictor="r", event_type="e", predicted_value=1.0,
            confidence=0.8, predicted_at=time.time(), event_time=time.time() + 60,
        )
        gap = GapSignal.create(room="r", prediction=pred, actual=2.0, delta=0.5)
        assert abs(gap.focus_score - 0.4) < 1e-6


class TestFocusQueue:
    def test_sorted_by_focus(self):
        pred = TMinusEvent(
            predictor="r", event_type="e", predicted_value=1.0,
            confidence=0.5, predicted_at=time.time(), event_time=time.time() + 60,
        )
        q = FocusQueue()
        q.add(GapSignal.create("r1", pred, 2.0, 0.3))
        pred2 = TMinusEvent(
            predictor="r", event_type="e", predicted_value=1.0,
            confidence=0.9, predicted_at=time.time(), event_time=time.time() + 60,
        )
        q.add(GapSignal.create("r2", pred2, 5.0, 0.8))
        assert q.top(1)[0].room == "r2"  # higher focus

    def test_clear_resolved(self):
        pred = TMinusEvent(
            predictor="r", event_type="e", predicted_value=1.0,
            confidence=0.9, predicted_at=time.time(), event_time=time.time() + 60,
        )
        q = FocusQueue()
        g = GapSignal.create("r1", pred, 2.0, 0.3)
        q.add(g)
        q.clear_resolved({g.gap_id})
        assert len(q.gaps) == 0


class TestSimulationRoom:
    def test_predict_and_observe_match(self):
        room = SimulationRoom(RoomAddress(instance="test", path=["room1"]), tolerance=0.2)
        room.predict("temp", predicted_value=100.0, confidence=0.9, horizon_seconds=60)
        gap = room.observe("temp", actual_value=105.0)
        assert gap is None  # within tolerance

    def test_predict_and_observe_mismatch(self):
        room = SimulationRoom(RoomAddress(instance="test", path=["room1"]), tolerance=0.1)
        room.predict("temp", predicted_value=100.0, confidence=0.9, horizon_seconds=60)
        gap = room.observe("temp", actual_value=500.0)  # delta = 0.8
        assert gap is not None
        assert gap.severity in (GapSeverity.MEDIUM, GapSeverity.HIGH, GapSeverity.CRITICAL)

    def test_observe_without_prediction(self):
        room = SimulationRoom(RoomAddress(instance="test", path=["room1"]))
        gap = room.observe("unknown-event", actual_value=42)
        assert gap is None  # no prediction to compare against

    def test_nested_rooms(self):
        room = SimulationRoom(RoomAddress(instance="test", path=["root"]))
        child = room.add_child("sub", kind=RoomKind.SENSOR)
        assert child.address.full_address == "test/root/sub"
        assert room.get_child(["sub"]) is child
        assert room.get_child(["nonexistent"]) is None

    def test_focus_report(self):
        room = SimulationRoom(RoomAddress(instance="test", path=["room1"]), tolerance=0.01)
        room.predict("x", predicted_value=0.0, confidence=0.95, horizon_seconds=60)
        room.observe("x", actual_value=1.0)
        report = room.focus_report()
        assert "Focus Queue" in report
