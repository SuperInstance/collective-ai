"""
Collective Inference — rooms predict, sensors listen, gaps surface.

Architecture:
  Every instance is a peer. Everything that's not a level is a room.
  Some rooms have levels (nested rooms). Connections are automated.

  Simulation-first: each room predicts what SHOULD happen (t-minus-event).
  Sensors listen for glitches — mismatches between prediction and reality.
  Gaps in understanding become the focus queue. Sound out the rocks.

  The glitches ARE the research agenda. The gaps ARE the work.

Core loop (per room, per instance):
  1. PREDICT: "At t-minus-event, I expect X to happen"
  2. LISTEN: sensors watch for what actually happens
  3. COMPARE: prediction vs reality
  4. GAP: if mismatch → gap signal → focus queue
  5. LEARN: focus on the gap, update the room's model
  6. SHARE: broadcast updated understanding to peers via I2I

This is how a fleet of instances collectively discovers what they don't know.
"""

import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


class RoomKind(Enum):
    SENSOR = "sensor"
    MODEL = "model"
    PREDICTOR = "predictor"
    COMPARATOR = "comparator"
    GLUE = "glue"
    LEVEL = "level"
    BRIDGE = "bridge"
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class RoomAddress:
    """How to find a room in the fleet. Format: instance/room/path/..."""
    instance: str
    path: List[str]

    @property
    def room_id(self) -> str:
        return "/".join(self.path)

    @property
    def full_address(self) -> str:
        return f"{self.instance}/{self.room_id}"

    def parent(self) -> Optional["RoomAddress"]:
        if len(self.path) <= 1:
            return None
        return RoomAddress(instance=self.instance, path=self.path[:-1])

    def child(self, name: str) -> "RoomAddress":
        return RoomAddress(instance=self.instance, path=self.path + [name])

    @classmethod
    def parse(cls, address: str) -> "RoomAddress":
        if not address or not isinstance(address, str):
            raise ValueError("address must be a non-empty string")
        parts = address.split("/")
        if len(parts) < 2:
            raise ValueError(f"address must contain at least instance/path, got: {address!r}")
        return cls(instance=parts[0], path=parts[1:])

    def __str__(self) -> str:
        return self.full_address

    def __repr__(self) -> str:
        return f"RoomAddress(instance={self.instance!r}, path={self.path!r})"


@dataclass
class TMinusEvent:
    """A prediction about what should happen at a future time."""
    predictor: str
    event_type: str
    predicted_value: Any
    confidence: float
    predicted_at: float
    event_time: float
    context: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")


    @property
    def time_until_event(self) -> float:
        return max(0.0, self.event_time - time.time())

    @property
    def is_expired(self) -> bool:
        return time.time() > self.event_time

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TMinusEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return (
            f"TMinusEvent(predictor={self.predictor!r}, event_type={self.event_type!r}, "
            f"confidence={self.confidence:.2f})"
        )


class GapSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GapSignal:
    """A mismatch between prediction and reality — the research agenda."""
    gap_id: str
    room: str
    prediction: TMinusEvent
    actual: Any
    severity: GapSeverity
    detected_at: float
    delta: float
    focus_score: float = 0.0

    @classmethod
    def create(cls, room: str, prediction: TMinusEvent, actual: Any, delta: float) -> "GapSignal":
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        severity = GapSeverity.LOW
        if delta > 0.5:
            severity = GapSeverity.MEDIUM
        if delta > 0.8:
            severity = GapSeverity.HIGH
        if delta > 0.95:
            severity = GapSeverity.CRITICAL

        focus_score = prediction.confidence * delta
        gap_id = hashlib.sha256(
            f"{room}:{prediction.event_type}:{prediction.predicted_at}:{actual}".encode()
        ).hexdigest()[:12]

        return cls(
            gap_id=gap_id, room=room, prediction=prediction, actual=actual,
            severity=severity, detected_at=time.time(), delta=delta, focus_score=focus_score,
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["prediction"] = self.prediction.to_dict()
        d["severity"] = self.severity.value
        return d

    def __repr__(self) -> str:
        return (
            f"GapSignal(gap_id={self.gap_id!r}, room={self.room!r}, "
            f"severity={self.severity.value!r}, delta={self.delta:.3f})"
        )


class FocusQueue:
    """Priority queue of gaps sorted by focus_score. Thread-safe."""

    def __init__(self) -> None:
        self.gaps: List[GapSignal] = []
        self._lock: threading.Lock = threading.Lock()

    def add(self, gap: GapSignal) -> None:
        with self._lock:
            self.gaps.append(gap)
            self.gaps.sort(key=lambda g: g.focus_score, reverse=True)

    def top(self, n: int = 5) -> List[GapSignal]:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        with self._lock:
            return list(self.gaps[:n])

    def by_room(self, room: str) -> List[GapSignal]:
        with self._lock:
            return [g for g in self.gaps if room in g.room]

    def by_severity(self, min_severity: GapSeverity = GapSeverity.MEDIUM) -> List[GapSignal]:
        order: Dict[GapSeverity, int] = {
            GapSeverity.LOW: 0, GapSeverity.MEDIUM: 1,
            GapSeverity.HIGH: 2, GapSeverity.CRITICAL: 3,
        }
        min_level = order[min_severity]
        with self._lock:
            return [g for g in self.gaps if order[g.severity] >= min_level]

    def clear_resolved(self, gap_ids: Set[str]) -> None:
        with self._lock:
            self.gaps = [g for g in self.gaps if g.gap_id not in gap_ids]

    def summary(self) -> Dict:
        with self._lock:
            return {
                "total_gaps": len(self.gaps),
                "critical": len([g for g in self.gaps if g.severity == GapSeverity.CRITICAL]),
                "high": len([g for g in self.gaps if g.severity == GapSeverity.HIGH]),
                "top_focus": [
                    {"room": g.room, "delta": round(g.delta, 3), "focus": round(g.focus_score, 3)}
                    for g in self.top(3)
                ],
            }

    def __repr__(self) -> str:
        return f"FocusQueue(gaps={len(self.gaps)})"

    def __len__(self) -> int:
        with self._lock:
            return len(self.gaps)


class SimulationRoom:
    """A room that predicts before it observes. Thread-safe."""

    def __init__(
        self,
        address: RoomAddress,
        kind: RoomKind = RoomKind.PREDICTOR,
        tolerance: float = 0.1,
    ) -> None:
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        self.address = address
        self.kind = kind
        self.tolerance = tolerance
        self.predictions: List[TMinusEvent] = []
        self.gaps: FocusQueue = FocusQueue()
        self.observations: List[Dict] = []
        self.lamport: int = 0
        self.child_rooms: Dict[str, "SimulationRoom"] = {}
        self._lock: threading.Lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SimulationRoom({self.address}, kind={self.kind.value}, lamport={self.lamport})"

    def predict(
        self,
        event_type: str,
        predicted_value: Any,
        confidence: float,
        horizon_seconds: float = 60.0,
        context: Optional[Dict] = None,
    ) -> TMinusEvent:
        if not event_type:
            raise ValueError("event_type must be a non-empty string")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")
        if horizon_seconds <= 0.0:
            raise ValueError(f"horizon_seconds must be positive, got {horizon_seconds}")
        with self._lock:
            self.lamport += 1
            event = TMinusEvent(
                predictor=str(self.address), event_type=event_type,
                predicted_value=predicted_value, confidence=confidence,
                predicted_at=time.time(), event_time=time.time() + horizon_seconds,
                context=context or {},
            )
            self.predictions.append(event)
            return event

    def observe(
        self,
        event_type: str,
        actual_value: Any,
        timestamp: Optional[float] = None,
    ) -> Optional[GapSignal]:
        if not event_type:
            raise ValueError("event_type must be a non-empty string")
        ts = timestamp or time.time()
        with self._lock:
            matching = [p for p in self.predictions if p.event_type == event_type and not p.is_expired]

            if not matching:
                self.observations.append({
                    "event_type": event_type, "value": actual_value,
                    "timestamp": ts, "predicted": False,
                })
                return None

            closest = min(matching, key=lambda p: abs(p.event_time - ts))
            delta = self._compute_delta(closest.predicted_value, actual_value)
            self.observations.append({
                "event_type": event_type, "value": actual_value, "timestamp": ts,
                "predicted": True, "predicted_value": closest.predicted_value, "delta": delta,
            })

            if delta > self.tolerance:
                gap = GapSignal.create(
                    room=str(self.address), prediction=closest,
                    actual=actual_value, delta=delta,
                )
                self.gaps.add(gap)
                return gap
            return None

    def _compute_delta(self, predicted: Any, actual: Any) -> float:
        if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
            if predicted == 0 and actual == 0:
                return 0.0
            return abs(predicted - actual) / max(abs(predicted), abs(actual), 1e-8)
        if isinstance(predicted, str) and isinstance(actual, str):
            return 0.0 if predicted == actual else 1.0
        if isinstance(predicted, bool) and isinstance(actual, bool):
            return 0.0 if predicted == actual else 1.0
        if isinstance(predicted, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(predicted) != len(actual):
                return 1.0
            deltas = [self._compute_delta(p, a) for p, a in zip(predicted, actual)]
            return sum(deltas) / len(deltas)
        return 0.0 if predicted == actual else 1.0

    def add_child(
        self, name: str, kind: RoomKind = RoomKind.PREDICTOR
    ) -> "SimulationRoom":
        if not name:
            raise ValueError("name must be a non-empty string")
        with self._lock:
            child_address = self.address.child(name)
            child = SimulationRoom(child_address, kind=kind, tolerance=self.tolerance)
            self.child_rooms[name] = child
            return child

    def get_child(self, path: List[str]) -> Optional["SimulationRoom"]:
        if not path:
            return self
        first, rest = path[0], path[1:]
        with self._lock:
            child = self.child_rooms.get(first)
        if child is None:
            return None
        return child.get_child(rest)

    def summary(self) -> Dict:
        with self._lock:
            return {
                "address": str(self.address), "kind": self.kind.value,
                "lamport": self.lamport, "predictions": len(self.predictions),
                "observations": len(self.observations), "gaps": self.gaps.summary(),
                "children": {n: c.summary() for n, c in self.child_rooms.items()},
            }

    def focus_report(self) -> str:
        top = self.gaps.top(5)
        if not top:
            return f"[{self.address}] No gaps — understanding is sound."
        lines = [f"[{self.address}] Focus Queue (sound out the rocks):"]
        for i, gap in enumerate(top):
            lines.append(
                f"  {i+1}. {gap.severity.value:8s} δ={gap.delta:.2f} "
                f"focus={gap.focus_score:.2f} "
                f"predicted={gap.prediction.predicted_value} actual={gap.actual}"
            )
        return "\n".join(lines)
