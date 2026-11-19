from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .calibration import safe_value, scale_summary
from .numerics import bridge_short_holes, true_runs


class EdgeState(IntEnum):
    MISSING = 0
    CLIPPED = 1
    COMPLETE = 2
    DISCONTINUOUS = 3


@dataclass(frozen=True)
class Span:
    start: int
    stop: int
    left: float
    right: float
    contrast: float

    @property
    def center(self):
        return 0.5 * (self.start + self.stop - 1)

    @property
    def width(self):
        return self.stop - self.start


@dataclass(frozen=True)
class StripEdges:
    stations: np.ndarray
    left_mm: np.ndarray
    right_mm: np.ndarray
    width_mm: np.ndarray
    states: np.ndarray
    material: np.ndarray
    intervals: np.ndarray

    @property
    def center_mm(self):
        return (self.left_mm + self.right_mm) / 2

    def summary(self):
        counts = {
            state.name.lower(): int(np.count_nonzero(self.states == state.value))
            for state in EdgeState
        }
        return {
            "rows": counts,
            "width_mm": scale_summary(self.width_mm),
            "center_mm": scale_summary(self.center_mm),
            "valid_material_pixels": int(np.count_nonzero(self.material)),
        }

    def row_table(self, indices):
        center = self.center_mm
        return [
            {
                "row": int(row),
                "station_mm": float(self.stations[row]),
                "left_mm": safe_value(self.left_mm[row]),
                "right_mm": safe_value(self.right_mm[row]),
                "width_mm": safe_value(self.width_mm[row]),
                "center_mm": safe_value(center[row]),
                "status": EdgeState(self.states[row]).name.lower(),
            }
            for row in indices
        ]

    def inner_mask(self, margin_cols):
        if margin_cols <= 0:
            return self.material.copy()
        columns = np.arange(self.material.shape[1])
        inside = (columns[None, :] >= self.intervals[:, :1] + margin_cols) & (
            columns[None, :] < self.intervals[:, 1:] - margin_cols
        )
        return self.material & inside


def crossing(row_values, valid, outer, inner, threshold):
    if not 0 <= outer < len(row_values) or not valid[outer]:
        return np.nan
    low = row_values[outer]
    high = row_values[inner]
    if not (low < threshold <= high):
        return np.nan
    fraction = (threshold - low) / (high - low)
    return float(outer + (inner - outer) * fraction)


def spans_for_row(values, valid, settings, min_width_px, bridge_px):
    bright = valid & (values >= settings.threshold)
    nearby = bridge_short_holes(bright, ~valid, bridge_px)
    spans = []
    for start, stop in true_runs(nearby):
        if stop - start < min_width_px:
            continue
        within = bright[start:stop]
        core = values[start:stop][within]
        spans.append(
            Span(
                int(start),
                int(stop),
                crossing(values, valid, start - 1, start, settings.threshold),
                crossing(values, valid, stop, stop - 1, settings.threshold),
                float(np.median(core)) - settings.threshold,
            )
        )
    return spans


def choose_span(spans, previous, continuity_px):
    if previous is None:
        return max(spans, key=lambda item: (item.width, item.contrast)), EdgeState.COMPLETE
    lower = previous.start - continuity_px
    upper = previous.stop + continuity_px
    nearby = [item for item in spans if lower <= item.center < upper]
    if not nearby:
        return None, EdgeState.DISCONTINUOUS
    return max(
        nearby,
        key=lambda item: (
            min(item.stop, previous.stop) - max(item.start, previous.start),
            item.width,
            item.contrast,
        ),
    ), EdgeState.COMPLETE


def track_edges(grid, settings):
    count, columns = grid.shape
    minimum_width = max(3, int(np.ceil(settings.minimum_width_mm / grid.column_pitch_mm)))
    bridge = int(np.floor(settings.bridge_mm / grid.column_pitch_mm))
    continuity = settings.continuity_mm / grid.column_pitch_mm
    left = np.full(count, np.nan, dtype=np.float64)
    right = np.full(count, np.nan, dtype=np.float64)
    states = np.full(count, EdgeState.MISSING, dtype=np.uint8)
    material = np.zeros(grid.shape, dtype=bool)
    intervals = np.zeros((count, 2), dtype=np.int32)
    previous = None
    missing_count = 0
    for index in range(count):
        if not grid.plan.observed[index]:
            previous = None
            missing_count = 0
            continue
        row = grid.values[index]
        valid = grid.valid[index]
        spans = spans_for_row(row, valid, settings, minimum_width, bridge)
        if not spans:
            missing_count += 1
            if missing_count >= 3:
                previous = None
            continue
        span, state = choose_span(spans, previous, continuity)
        if span is None:
            states[index] = state
            missing_count += 1
            if missing_count >= 3:
                previous = None
            continue
        left[index] = span.left
        right[index] = span.right
        if not np.isfinite(span.left) or not np.isfinite(span.right):
            state = EdgeState.CLIPPED
        material[index, span.start : span.stop] = valid[span.start : span.stop]
        intervals[index] = [span.start, span.stop]
        states[index] = state
        missing_count = 0
        previous = span

    cross_pitch = grid.column_pitch_mm
    start = float(grid.transverse[0])
    left = start + left * cross_pitch
    right = start + right * cross_pitch
    width = right - left
    return StripEdges(grid.stations, left, right, width, states, material, intervals)
