from dataclasses import dataclass

import numpy as np

from .calibration import scale_summary
from .numerics import true_runs
from .presentation import extrema_indices


@dataclass(frozen=True)
class MotionSegment:
    first_line: int
    last_line: int
    period: float
    distance: float
    observed_speed: np.ndarray
    instants: np.ndarray
    stations: np.ndarray

    def summary(self):
        return {
            "first_line": self.first_line,
            "last_line": self.last_line,
            "from_mm": float(self.stations[0]),
            "to_mm": float(self.stations[-1]),
            "duration_s": self.period,
            "distance_mm": self.distance,
            "average_mm_s": self.distance / self.period,
            "local_speed_mm_s": scale_summary(self.observed_speed),
        }

    def points(self, indices):
        return [
            {
                "line": int(self.first_line + index),
                "elapsed_s": float(self.instants[index]),
                "station_mm": float(self.stations[index]),
                "speed_mm_s": float(self.observed_speed[index]),
            }
            for index in indices
        ]


def estimate_speed(capture, index_from, index_to):
    times = capture.seconds[index_from:index_to]
    ticks = capture.ticks[index_from:index_to]
    pitch = capture.metadata.geometry.tick_mm
    coordinates = capture.metadata.geometry.station(ticks)
    span = len(times)
    local = np.arange(span)
    before = np.maximum(0, local - 2)
    after = np.minimum(span - 1, local + 2)
    delta_station = (ticks[after] - ticks[before]) * pitch
    delta_seconds = times[after] - times[before]
    return MotionSegment(
        int(index_from),
        int(index_to - 1),
        float(times[-1] - times[0]),
        float((ticks[-1] - ticks[0]) * pitch),
        delta_station / delta_seconds,
        times - capture.seconds[0],
        coordinates,
    )


def infer_motion(capture, report):
    uninterrupted = ~(
        (report.delay_ms > report.summary["pause_limit_ms"])
        | (report.step_mm > report.summary["gap_limit_mm"])
    )
    return [
        estimate_speed(capture, int(first), int(stop + 1))
        for first, stop in true_runs(uninterrupted)
        if stop - first >= 3
    ]


def motion_values(segments, max_points):
    if not segments:
        return []
    if len(segments) > max_points // 4:
        rank = sorted(segments, key=lambda item: item.period, reverse=True)
        report = sorted(rank[:max_points // 4], key=lambda item: item.first_line)
    else:
        report = segments
    lengths = [len(segment.observed_speed) for segment in report]
    total = sum(lengths)
    result = []
    for index, segment in enumerate(report):
        if index:
            result.append(
                {
                    "line": segment.first_line - 0.5,
                    "elapsed_s": float(segment.instants[0]),
                    "station_mm": None,
                    "speed_mm_s": None,
                }
            )
        share = max(3, int((max_points - len(report)) * lengths[index] / total))
        sample = extrema_indices([segment.observed_speed], share)
        result.extend(segment.points(sample))
    return result
