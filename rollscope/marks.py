from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .calibration import safe_value, scale_summary
from .numerics import select_peaks, true_runs

PITCH_TOLERANCE_FRACTION = 0.17


@dataclass(frozen=True)
class Mark:
    id: int
    row: int
    station_mm: float
    prominence: float
    support_fraction: float
    segment: int
    pitch_group: str | None = None

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
    expected_mm: float | None
    error_mm: float | None
    inferred_missing_marks: int

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MarkShift:
    first_spacing_mm: float
    second_spacing_mm: float
    from_mm: float
    to_mm: float
    transition_intervals: int
    misfit_intervals: int
    minimum_segment_mm: float

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
    mode: str = "single_pitch"
    shift: MarkShift | None = None

    def summary(self):
        accepted = [interval for interval in self.intervals if interval.classification == "regular"]
        result = {
            "mode": self.mode,
            "mark_count": len(self.marks),
            "interval_count": len(self.intervals),
            "regular_intervals": len(accepted),
            "sampled_rows": int(np.count_nonzero(np.isfinite(self.darkness))),
            "missed_candidates": sum(item.inferred_missing_marks for item in self.intervals),
            "unknown_intervals": sum(
                item.classification == "unknown_across_gap" for item in self.intervals
            ),
        }
        if self.mode == "single_pitch":
            result["spacing_mm"] = scale_summary(
                [interval.measured_mm for interval in accepted]
            )
            return result
        result["shift_intervals"] = sum(
            item.classification == "shift_transition" for item in self.intervals
        )
        if self.shift is None:
            result["shift_status"] = "undetermined"
            return result
        shift = self.shift
        first = [
            interval.measured_mm
            for interval in accepted
            if interval.expected_mm == shift.first_spacing_mm
        ]
        second = [
            interval.measured_mm
            for interval in accepted
            if interval.expected_mm == shift.second_spacing_mm
        ]
        result.update(
            {
                "shift_status": "located",
                "first_spacing_mm": shift.first_spacing_mm,
                "second_spacing_mm": shift.second_spacing_mm,
                "shift_from_mm": shift.from_mm,
                "shift_to_mm": shift.to_mm,
                "misfit_intervals": shift.misfit_intervals,
                "minimum_segment_mm": shift.minimum_segment_mm,
                "first_segment_spacing_mm": scale_summary(first),
                "second_segment_spacing_mm": scale_summary(second),
            }
        )
        return result

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
        tolerance = expected * PITCH_TOLERANCE_FRACTION
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


def unjudged_interval(current, following, classification):
    return MarkInterval(
        current.id,
        following.id,
        current.station_mm,
        following.station_mm,
        following.station_mm - current.station_mm,
        classification,
        None,
        None,
        0,
    )


def pitch_fits(distances, pitch):
    tolerance = PITCH_TOLERANCE_FRACTION * pitch
    multiples = np.maximum(1, np.rint(distances / pitch).astype(np.int64))
    return np.abs(distances - multiples * pitch) <= tolerance


def locate_shift(stations, spacings, minimum_segment_mm):
    """Place the single pitch change that best explains the observed marks.

    Every candidate split treats one interval as the changeover and scores the
    remaining intervals against the segment pitch on their side. Splits that
    tie at the lowest misfit count are all kept, so the changeover stays a
    range whenever neighbouring explanations fit equally well. Returns the
    winning (first, second) pitch assignment, the plausible split positions
    and the misfit count, or None when no split leaves both segments at least
    minimum_segment_mm of observed marks.
    """
    stations = np.asarray(stations, dtype=np.float64)
    count = len(stations)
    if count < 2:
        return None
    distances = np.diff(stations)
    fits = {pitch: pitch_fits(distances, pitch) for pitch in set(spacings)}
    first_span = stations[:-1] - stations[0]
    second_span = stations[-1] - stations[1:]
    feasible = (first_span >= minimum_segment_mm - 1e-9) & (
        second_span >= minimum_segment_mm - 1e-9
    )
    best = None
    assignments = ((spacings[0], spacings[1]), (spacings[1], spacings[0]))
    for order, assignment in enumerate(assignments):
        first_miss = np.concatenate(([0], np.cumsum(~fits[assignment[0]])))
        second_miss = np.concatenate(([0], np.cumsum(~fits[assignment[1]])))
        costs = first_miss[:-1] + (second_miss[-1] - second_miss[1:])
        viable = np.flatnonzero(feasible)
        if not len(viable):
            continue
        low = int(np.min(costs[viable]))
        plausible = [int(split) for split in viable if costs[split] == low]
        candidate = (low, len(plausible), order, assignment, plausible)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        return None
    low, _, _, assignment, plausible = best
    return assignment, plausible, low


def segment_pitch_groups(mark_count, plausible):
    first_split = plausible[0]
    last_split = plausible[-1]
    groups = []
    for index in range(mark_count):
        if index <= first_split:
            groups.append("first")
        elif index > last_split:
            groups.append("second")
        else:
            groups.append("shift")
    return groups


def two_segment_intervals(marks, assignment, plausible):
    first_split = plausible[0]
    last_split = plausible[-1]
    intervals = []
    for position, (current, following) in enumerate(zip(marks, marks[1:])):
        if position < first_split:
            intervals.append(classify_interval(current, following, assignment[0]))
        elif position > last_split:
            intervals.append(classify_interval(current, following, assignment[1]))
        else:
            intervals.append(unjudged_interval(current, following, "shift_transition"))
    return tuple(intervals)


def detect_marks(grid, edges, settings, detection_pitch):
    darkness, support = band_measurements(
        grid, edges, settings.band_start_fraction, settings.band_end_fraction
    )
    residual, segments = prominence_by_segment(
        darkness, ~grid.plan.observed, grid.row_pitch_mm, detection_pitch
    )
    separation = int(np.clip(round(0.55 * detection_pitch / grid.row_pitch_mm), 1, 2000))
    eligible = np.isfinite(residual) & (residual >= settings.minimum_prominence)
    peaks = select_peaks(residual, eligible, separation)
    return darkness, support, residual, segments, peaks


def build_marks(grid, support, segments, peaks, pitch_groups=None):
    marks = []
    for position, peak in enumerate(peaks):
        marks.append(
            Mark(
                position + 1,
                peak.index,
                float(grid.stations[peak.index]),
                peak.prominence,
                float(support[peak.index]),
                int(segments[peak.index]),
                None if pitch_groups is None else pitch_groups[position],
            )
        )
    return tuple(marks)


def inspect_marks(grid, edges, settings):
    second = settings.second_spacing_mm
    detection_pitch = settings.expected_spacing_mm
    if second is not None:
        detection_pitch = min(detection_pitch, second)
    darkness, support, residual, segments, peaks = detect_marks(
        grid, edges, settings, detection_pitch
    )
    if second is None:
        marks = build_marks(grid, support, segments, peaks)
        intervals = tuple(
            classify_interval(current, following, settings.expected_spacing_mm)
            for current, following in zip(marks, marks[1:])
        )
        return MarkReport(grid.stations, darkness, residual, support, marks, intervals)

    stations = np.array([float(grid.stations[peak.index]) for peak in peaks])
    outcome = locate_shift(
        stations,
        (settings.expected_spacing_mm, second),
        settings.minimum_segment_mm,
    )
    if outcome is None:
        marks = build_marks(grid, support, segments, peaks)
        intervals = tuple(
            unjudged_interval(current, following, "undetermined")
            for current, following in zip(marks, marks[1:])
        )
        return MarkReport(
            grid.stations, darkness, residual, support, marks, intervals, mode="two_segment"
        )
    assignment, plausible, misfits = outcome
    groups = segment_pitch_groups(len(peaks), plausible)
    marks = build_marks(grid, support, segments, peaks, groups)
    intervals = two_segment_intervals(marks, assignment, plausible)
    shift = MarkShift(
        assignment[0],
        assignment[1],
        float(stations[plausible[0]]),
        float(stations[plausible[-1] + 1]),
        plausible[-1] - plausible[0] + 1,
        misfits,
        settings.minimum_segment_mm,
    )
    return MarkReport(
        grid.stations,
        darkness,
        residual,
        support,
        marks,
        intervals,
        mode="two_segment",
        shift=shift,
    )
