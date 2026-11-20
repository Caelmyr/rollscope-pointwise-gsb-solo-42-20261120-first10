from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .domain import MAX_GRID_PIXELS, UserInputError


class Support(IntEnum):
    GAP = 0
    NATIVE = 1
    BLEND = 2


@dataclass(frozen=True)
class EncoderGroups:
    ticks: np.ndarray
    first: np.ndarray
    counts: np.ndarray
    values: np.ndarray
    valid: np.ndarray

    def sources(self, index):
        first = int(self.first[index])
        return range(first, first + int(self.counts[index]))


@dataclass(frozen=True)
class GridPlan:
    stations: np.ndarray
    left: np.ndarray
    right: np.ndarray
    right_weight: np.ndarray
    support: np.ndarray
    pitch_mm: float
    gap_limit_mm: float

    @property
    def observed(self):
        return self.support != Support.GAP


@dataclass(frozen=True)
class StationGrid:
    values: np.ndarray
    valid: np.ndarray
    transverse: np.ndarray
    groups: EncoderGroups
    plan: GridPlan
    transverse_calibrated: np.ndarray | None = None

    @property
    def stations(self):
        return self.plan.stations

    @property
    def shape(self):
        return self.values.shape

    @property
    def row_pitch_mm(self):
        return self.plan.pitch_mm

    @property
    def column_pitch_mm(self):
        return float(np.median(np.diff(self.transverse)))

    @property
    def calibrated_mask(self):
        if self.transverse_calibrated is None:
            return np.ones(len(self.transverse), dtype=bool)
        return self.transverse_calibrated

    def summary(self):
        capabilities = {
            support.name.lower(): int(np.count_nonzero(self.plan.support == support.value))
            for support in Support
        }
        if self.transverse_calibrated is None:
            calibration = {
                "transverse_calibrated": False,
                "calibrated_columns": len(self.transverse),
                "uncalibrated_columns": 0,
            }
        else:
            calibration = {
                "transverse_calibrated": True,
                "calibrated_columns": int(np.count_nonzero(self.transverse_calibrated)),
                "uncalibrated_columns": int(np.count_nonzero(~self.transverse_calibrated)),
                "first_calibrated_column": int(np.flatnonzero(self.transverse_calibrated)[0]),
                "last_calibrated_column": int(np.flatnonzero(self.transverse_calibrated)[-1]),
            }
        return {
            "stations": len(self.stations),
            "columns": len(self.transverse),
            "pitch_mm": self.row_pitch_mm,
            "gap_limit_mm": self.plan.gap_limit_mm,
            "first_station_mm": float(self.stations[0]),
            "last_station_mm": float(self.stations[-1]),
            "support_counts": capabilities,
            "valid_fraction": float(np.count_nonzero(self.valid) / self.valid.size),
            "encoder_groups": len(self.groups.ticks),
            "repeated_acquisitions": int(np.sum(self.groups.counts - 1)),
            **calibration,
        }

    def provenance(self, row, column=None):
        coverage = Support(self.plan.support[row])
        result = {
            "row": int(row),
            "station_mm": float(self.stations[row]),
            "coverage": coverage.name.lower(),
            "sources": [],
        }
        if coverage == Support.GAP:
            return result
        left = int(self.plan.left[row])
        fraction = float(self.plan.right_weight[row])
        result["sources"].append(self.source_group(left, 1 - fraction, column))
        if coverage == Support.BLEND:
            right = int(self.plan.right[row])
            result["sources"].append(self.source_group(right, fraction, column))
        return result

    def source_group(self, index, weight, column):
        source_lines = self.groups.sources(index)
        result = {
            "tick": int(self.groups.ticks[index]),
            "weight": weight,
            "first_line": source_lines.start,
            "line_count": len(source_lines),
        }
        if column is not None:
            result["column_valid"] = bool(self.groups.valid[index, column])
        return result


def encoder_groups(ticks, corrected):
    values, first, counts = np.unique(ticks, return_index=True, return_counts=True)
    weighted = np.where(corrected.valid, corrected.values, 0)
    counts_by_col = np.add.reduceat(corrected.valid, first, axis=0)
    sums = np.add.reduceat(weighted, first, axis=0)
    known = counts_by_col > 0
    averaged = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts_by_col, where=known, out=averaged)
    return EncoderGroups(values, first, counts, averaged, known)


def grid_coordinates(capture, groups, settings):
    pitch = capture.positive_pitch if settings.pitch_mm is None else settings.pitch_mm
    gap_limit = 3.5 * capture.positive_pitch if settings.gap_limit_mm is None else (
        settings.gap_limit_mm
    )
    origin_tick = groups.ticks[0]
    coordinate = (groups.ticks - origin_tick) * capture.metadata.geometry.tick_mm
    limit = float(coordinate[-1])
    ratio = limit / pitch
    count = int(np.floor(ratio + 1e-9)) + 1
    if count < 2:
        raise UserInputError("The station pitch must be no larger than the captured length.")
    if count * capture.columns > MAX_GRID_PIXELS:
        raise UserInputError(
            f"The station grid would exceed {MAX_GRID_PIXELS:,} cells. Increase its pitch."
        )
    targets = np.arange(count, dtype=np.float64) * pitch
    targets[-1] = min(limit, targets[-1])
    first_station = float(capture.metadata.geometry.station(origin_tick))
    return coordinate, targets, first_station, pitch, gap_limit


def interpolation_plan(positions, targets, origin, pitch, gap_limit):
    insertion = np.searchsorted(positions, targets, side="left")
    right = np.clip(insertion, 0, len(positions) - 1)
    left = np.clip(insertion - 1, 0, len(positions) - 1)
    tolerance = max(1e-12, 8 * np.finfo(np.float64).eps * positions[-1])
    exact_left = np.abs(positions[left] - targets) <= tolerance
    exact_right = np.abs(positions[right] - targets) <= tolerance
    exact = exact_left | exact_right
    exact_index = np.where(exact_right, right, left)

    distance = positions[right] - positions[left]
    bracketed = (insertion > 0) & (insertion < len(positions))
    blended = ~exact & bracketed & (distance <= gap_limit + tolerance)
    fractions = np.zeros(len(targets), dtype=np.float64)
    np.divide(targets - positions[left], distance, out=fractions, where=blended)
    support = np.full(len(targets), Support.GAP, dtype=np.uint8)
    support[exact] = Support.NATIVE
    support[blended] = Support.BLEND
    left[exact] = exact_index[exact]
    right[exact] = exact_index[exact]
    return GridPlan(targets + origin, left, right, fractions, support, pitch, gap_limit)


def sample_grid(groups, plan):
    shape = (len(plan.stations), groups.values.shape[1])
    output = np.full(shape, np.nan, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)

    native_rows = np.flatnonzero(plan.support == Support.NATIVE)
    native_sources = plan.left[native_rows]
    output[native_rows] = groups.values[native_sources]
    valid[native_rows] = groups.valid[native_sources]

    blend_rows = np.flatnonzero(plan.support == Support.BLEND)
    l_source = plan.left[blend_rows]
    r_source = plan.right[blend_rows]
    jointly_valid = groups.valid[l_source] & groups.valid[r_source]
    weights = plan.right_weight[blend_rows, None].astype(np.float32)
    blend = (1 - weights) * groups.values[l_source] + weights * groups.values[r_source]
    output[blend_rows] = np.where(jointly_valid, blend, np.nan)
    valid[blend_rows] = jointly_valid
    return output, valid


def build_grid(capture, calibrated, settings, transverse_calibration=None):
    groups = encoder_groups(capture.ticks, calibrated)
    positions, targets, origin, pitch, limit = grid_coordinates(capture, groups, settings)
    plan = interpolation_plan(positions, targets, origin, pitch, limit)
    values, valid = sample_grid(groups, plan)
    if transverse_calibration is None:
        transverse = capture.transverse
        calibrated_mask = None
    else:
        if transverse_calibration.columns != capture.columns:
            raise UserInputError(
                "The transverse calibration was built for a detector with a different column count."
            )
        transverse = transverse_calibration.transverse
        calibrated_mask = transverse_calibration.calibrated
    return StationGrid(values, valid, transverse, groups, plan, calibrated_mask)


def index_near(values, query):
    right = int(np.searchsorted(values, query, side="left"))
    if right == 0:
        return 0
    if right >= len(values):
        return len(values) - 1
    left = right - 1
    return left if query - values[left] <= values[right] - query else right
