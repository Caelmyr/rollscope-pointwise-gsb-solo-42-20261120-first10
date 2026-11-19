from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .calibration import safe_value, scale_summary
from .numerics import select_peaks, true_runs


@dataclass(frozen=True)
class Mark:
    id: int
    row: int
    station_mm: float
    prominence: float
    support_fraction: float
    segment: int

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MarkInterval:
    first_mark: int
    second_mark: int
    from_mm: float
    to_mm: float
    measured_mm: float
    classification: str
    expected_mm: float
    error_mm: float | None
    inferred_missing_marks: int

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MarkReport:
    stations: np.ndarray
    darkness: np.ndarray
    prominence: np.ndarray
    support: np.ndarray
    marks: tuple
    intervals: tuple

    def summary(self):
        accepted = [interval for interval in self.intervals if interval.classification == "regular"]
        return {
            "mark_count": len(self.marks),
            "interval_count": len(self.intervals),
            "regular_intervals": len(accepted),
            "sampled_rows": int(np.count_nonzero(np.isfinite(self.darkness))),
            "spacing_mm": scale_summary([interval.measured_mm for interval in accepted]),
            "missed_candidates": sum(item.inferred_missing_marks for item in self.intervals),
            "unknown_intervals": sum(
                item.classification == "unknown_across_gap" for item in self.intervals
            ),
        }

    def series(self, indices):
        return [
            {
                "station_mm": float(self.stations[index]),
                "darkness": safe_value(self.darkness[index]),
                "prominence": safe_value(self.prominence[index]),
                "support_fraction": float(self.support[index]),
            }
            for index in indices
        ]

    def table(self):
        return [mark.as_dict() for mark in self.marks]

    def intervals_table(self):
        return [interval.as_dict() for interval in self.intervals]


def band_measurements(grid, edges, start_fraction, end_fraction):
    length = grid.shape[0]
    darkness = np.full(length, np.nan)
    support = np.zeros(length, dtype=np.float32)
    for row in range(length):
        left = edges.left_mm[row]
        right = edges.right_mm[row]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        width = right - left
        endpoints = [left + width * start_fraction, left + width * end_fraction]
        start = int(np.searchsorted(grid.transverse, endpoints[0], side="left"))
        stop = int(np.searchsorted(grid.transverse, endpoints[1], side="right"))
        if stop - start < 3:
            continue
        patch = grid.values[row, start:stop]
        valid = edges.material[row, start:stop]
        support[row] = np.count_nonzero(valid) / len(patch)
        if support[row] < 0.65:
            continue
        darkness[row] = 1.0 - float(np.median(patch[valid]))
    return darkness, support


def prominence_by_segment(signal, relative_gap, pitch, expected_mm):
    prominent = np.full(signal.shape, np.nan)
    segments = np.full(signal.shape, -1, dtype=np.int32)
    radius = int(np.clip(round(0.17 * expected_mm / pitch), 2, 70))
    kernel = 2 * radius + 1
    valid = np.isfinite(signal) & ~relative_gap
    for segment_id, (start, stop) in enumerate(true_runs(valid)):
        if stop - start < 5:
            continue
        values = signal[start:stop]
        backdrop = ndimage.median_filter(values, size=kernel, mode="nearest")
        prominent[start:stop] = values - backdrop
        segments[start:stop] = segment_id
        shoulder = min(2, (stop - start - 1) // 2)
        prominent[start : start + shoulder] = np.nan
        prominent[stop - shoulder : stop] = np.nan
    return prominent, segments


def classify_interval(current, following, expected):
    distance = following.station_mm - current.station_mm
    if current.segment != following.segment:
        category = "unknown_across_gap"
        missing = 0
        error = None
    else:
        multiples = max(1, int(round(distance / expected)))
        error = float(distance - multiples * expected)
        missing = 0
        tolerance = expected * 0.17
        if abs(error) > tolerance:
            category = "irregular"
        elif multiples > 1:
            category = "multiple_spacing"
            missing = multiples - 1
        else:
            category = "regular"
    return MarkInterval(
        current.id,
        following.id,
        current.station_mm,
        following.station_mm,
        distance,
        category,
        expected,
        error,
        missing,
    )


def inspect_marks(grid, edges, settings):
    darkness, support = band_measurements(
        grid, edges, settings.band_start_fraction, settings.band_end_fraction
    )
    residual, segments = prominence_by_segment(
        darkness, ~grid.plan.observed, grid.row_pitch_mm, settings.expected_spacing_mm
    )
    separation = int(np.clip(round(0.55 * settings.expected_spacing_mm / grid.row_pitch_mm), 1, 2000))
    eligible = np.isfinite(residual) & (residual >= settings.minimum_prominence)
    peaks = select_peaks(residual, eligible, separation)
    marks = tuple(
        Mark(
            identifier,
            peak.index,
            float(grid.stations[peak.index]),
            peak.prominence,
            float(support[peak.index]),
            int(segments[peak.index]),
        )
        for identifier, peak in enumerate(peaks, 1)
    )
    intervals = tuple(
        classify_interval(current, following, settings.expected_spacing_mm)
        for current, following in zip(marks, marks[1:])
    )
    return MarkReport(grid.stations, darkness, residual, support, marks, intervals)
