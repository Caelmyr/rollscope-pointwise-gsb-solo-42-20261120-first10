from dataclasses import dataclass
from enum import Enum

import numpy as np

from .calibration import scale_summary


class EventKind(str, Enum):
    PAUSE = "clock_pause"
    JUMP = "encoder_jump"
    REPEATED = "repeated_position"
    COINCIDENT = "pause_and_jump"


@dataclass(frozen=True)
class AcquisitionEvent:
    kind: EventKind
    previous_line: int
    next_line: int
    station_before_mm: float
    station_after_mm: float
    delta_station_mm: float
    delta_time_ms: float
    speed_mm_s: float

    def as_dict(self):
        return {
            "kind": self.kind.value,
            "previous_line": self.previous_line,
            "next_line": self.next_line,
            "station_before_mm": self.station_before_mm,
            "station_after_mm": self.station_after_mm,
            "delta_station_mm": self.delta_station_mm,
            "delta_time_ms": self.delta_time_ms,
            "speed_mm_s": self.speed_mm_s,
        }


@dataclass(frozen=True)
class AcquisitionReport:
    summary: dict
    events: tuple
    step_mm: np.ndarray
    delay_ms: np.ndarray
    speeds: np.ndarray

    def table(self):
        return [event.as_dict() for event in self.events]

    def series(self, stations, max_points):
        from .presentation import extrema_indices

        indices = extrema_indices([self.step_mm, self.delay_ms], max_points)
        return [
            {
                "arrival_line": int(index + 1),
                "station_mm": float(stations[index + 1]),
                "step_mm": float(self.step_mm[index]),
                "delay_ms": float(self.delay_ms[index]),
                "speed_mm_s": float(self.speeds[index]),
            }
            for index in indices
        ]


def measure_acquisition(capture, gap_limit_mm):
    step_mm = np.diff(capture.ticks) * capture.metadata.geometry.tick_mm
    delta_time = np.diff(capture.seconds)
    delay_ms = delta_time * 1000
    median_ms = float(np.median(delay_ms))
    pause_limit = median_ms * 3.5
    speeds = step_mm / delta_time
    jumps = step_mm > gap_limit_mm
    pauses = delay_ms > pause_limit
    repeats = step_mm == 0
    stations = capture.stations
    events = classify_events(stations, step_mm, delay_ms, speeds, jumps, pauses, repeats)

    continuous = (step_mm > 0) & ~(jumps | pauses)
    summary = {
        "line_step_mm": scale_summary(step_mm[step_mm > 0]),
        "clock_delay_ms": scale_summary(delay_ms),
        "continuous_speed_mm_s": scale_summary(speeds[continuous]),
        "pause_limit_ms": pause_limit,
        "gap_limit_mm": float(gap_limit_mm),
        "coincident_events": int(np.count_nonzero(jumps & pauses)),
        "encoder_jumps": int(np.count_nonzero(jumps)),
        "clock_pauses": int(np.count_nonzero(pauses)),
        "repeated_steps": int(np.count_nonzero(repeats)),
    }
    return AcquisitionReport(summary, tuple(events), step_mm, delay_ms, speeds)


def classify_events(stations, steps, delays, speeds, jumps, pauses, repeats):
    for index in np.flatnonzero(jumps | pauses | repeats):
        kind = event_kind(jumps[index], pauses[index])
        yield AcquisitionEvent(
            kind=kind,
            previous_line=int(index),
            next_line=int(index + 1),
            station_before_mm=float(stations[index]),
            station_after_mm=float(stations[index + 1]),
            delta_station_mm=float(steps[index]),
            delta_time_ms=float(delays[index]),
            speed_mm_s=float(speeds[index]),
        )


def event_kind(jump, pause):
    if jump and pause:
        return EventKind.COINCIDENT
    if jump:
        return EventKind.JUMP
    if pause:
        return EventKind.PAUSE
    return EventKind.REPEATED
