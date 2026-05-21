# collective-ai

**Simulation-first collective inference: predict, observe, gap, learn.**

A zero-dependency Python library for building systems where agents predict what should happen, observe reality, and surface gaps (mismatches) as the research agenda.

## Core Concept

Every room in a collective AI system follows the same loop:

1. **PREDICT** — "At time T, I expect X with confidence Y"
2. **OBSERVE** — sensors watch for what actually happens
3. **COMPARE** — prediction vs reality
4. **GAP** — if mismatch → gap signal → focus queue
5. **LEARN** — focus on the gap, update the room's model

"The glitches ARE the research agenda. The gaps ARE the work."

## Installation

```bash
pip install collective-ai
```

Zero hard dependencies. Python 3.10+.

## Quick Start

```python
from collective_ai import SimulationRoom, RoomAddress, RoomKind

# Create a room
addr = RoomAddress(instance="agent@host", path=["drift-detect", "predictor"])
room = SimulationRoom(addr, kind=RoomKind.PREDICTOR, tolerance=0.1)

# Predict
room.predict("drift-exceeds-threshold", predicted_value=0.3, confidence=0.9, horizon_seconds=60)

# Observe reality
gap = room.observe("drift-exceeds-threshold", actual_value=0.8)

if gap:
    print(gap.severity)  # HIGH
    print(gap.focus_score)  # confidence × delta

# Focus report
print(room.focus_report())
```

## Key Types

- **SimulationRoom** — predicts, observes, and surfaces gaps
- **RoomAddress** — fleet-wide addressing (instance/room/path)
- **TMinusEvent** — a timestamped prediction with confidence
- **GapSignal** — prediction vs reality mismatch with severity
- **FocusQueue** — priority queue of gaps by focus score

## License

MIT
