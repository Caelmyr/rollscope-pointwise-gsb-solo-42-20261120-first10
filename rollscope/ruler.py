from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np

from .calibration import safe_value
from .domain import UserInputError, exact_keys, expect_object, finite_number
from .grid import Support
from .presentation import extrema_indices


class PathSupport(IntEnum):
    GAP = 0
    BAD_PIXEL = 1
    BACKGROUND = 2
    MATERIAL = 3


@dataclass(frozen=True)
class PhysicalPoint:
    station_mm: float
    transverse_mm: float

    @classmethod
    def parse(cls, payload):
        payload = expect_object(payload, "Ruler endpoint")
        exact_keys(payload, ("station_mm", "transverse_mm"), "Ruler endpoint")
        return cls(
            finite_number(payload["station_mm"], "Endpoint station"),
            finite_number(payload["transverse_mm"], "Endpoint transverse position"),
        )

    def inside(self, grid):
        return (
            grid.stations[0] <= self.station_mm <= grid.stations[-1]
            and grid.transverse[0] <= self.transverse_mm <= grid.transverse[-1]
        )


@dataclass(frozen=True)
class Ruler:
    start: PhysicalPoint
    end: PhysicalPoint

    @classmethod
    def parse(cls, payload, grid):
        payload = expect_object(payload, "Ruler")
        exact_keys(payload, ("start", "end"), "Ruler")
        result = cls(PhysicalPoint.parse(payload["start"]), PhysicalPoint.parse(payload["end"]))
        if not result.start.inside(grid) or not result.end.inside(grid):
            raise UserInputError("Ruler endpoints must lie within the current station grid.")
        if result.length < 1e-6:
            raise UserInputError("The ruler must have two different endpoints.")
        return result

    @property
    def length(self):
        return float(np.hypot(self.dy, self.dx))

    @property
    def dy(self):
        return self.end.station_mm - self.start.station_mm

    @property
    def dx(self):
        return self.end.transverse_mm - self.start.transverse_mm

    def interpolate(self, ratio):
        return (
            self.start.station_mm + self.dy * ratio,
            self.start.transverse_mm + self.dx * ratio,
        )

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MeasuredPath:
    ruler: Ruler
    enter: np.ndarray
    leave: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    states: np.ndarray
    record_values: np.ndarray
    spot_ids: np.ndarray
    calibrated_cells: np.ndarray | None = None

    @property
    def widths(self):
        return (self.leave - self.enter) * self.ruler.length

    def summary(self):
        widths = self.widths
        state_counts = {
            state.name.lower(): float(np.sum(widths[self.states == state]))
            for state in PathSupport
        }
        chip = self.record_values[self.states == PathSupport.MATERIAL]
        weights = widths[self.states == PathSupport.MATERIAL]
        reflectance = float(np.average(chip, weights=weights)) if weights.size else None
        uncalibrated_width = (
            float(np.sum(widths[~self.calibrated_cells]))
            if self.calibrated_cells is not None
            else 0.0
        )
        return {
            "length_mm": self.ruler.length,
            "travel_component_mm": self.ruler.dy,
            "transverse_component_mm": self.ruler.dx,
            "travel_angle_deg": float(np.degrees(np.arctan2(self.ruler.dx, self.ruler.dy))),
            "cell_count": len(widths),
            "distance_by_support_mm": state_counts,
            "weighted_material_reflectance": reflectance,
            "intersected_spots": len(np.unique(self.spot_ids[self.spot_ids > 0])),
            "uncalibrated_length_mm": uncalibrated_width,
            "fully_calibrated": bool(
                self.calibrated_cells is None or bool(np.all(self.calibrated_cells))
            ),
        }

    def profile(self, maximum_points=None):
        count = len(self.enter)
        if maximum_points is None:
            indices = np.arange(count)
        else:
            indices = extrema_indices([self.record_values, self.states], maximum_points)
        middle = 0.5 * (self.enter[indices] + self.leave[indices])
        positions, transverse = self.ruler.interpolate(middle)
        return [
            {
                "distance_mm": float(middle[local] * self.ruler.length),
                "station_mm": float(positions[local]),
                "transverse_mm": float(transverse[local]),
                "row": int(self.rows[index]),
                "column": int(self.columns[index]),
                "state": PathSupport(self.states[index]).name.lower(),
                "reflectance": safe_value(self.record_values[index]),
                "spot_id": int(self.spot_ids[index]) or None,
                "calibrated": (
                    True
                    if self.calibrated_cells is None
                    else bool(self.calibrated_cells[index])
                ),
            }
            for local, index in enumerate(indices)
        ]

    def intervals(self):
        boundaries = np.r_[0, np.flatnonzero(np.diff(self.states)) + 1, len(self.states)]
        return [
            {
                "from_mm": float(self.enter[first] * self.ruler.length),
                "to_mm": float(self.leave[stop - 1] * self.ruler.length),
                "support": PathSupport(self.states[first]).name.lower(),
                "cells": int(stop - first),
            }
            for first, stop in zip(boundaries[:-1], boundaries[1:])
        ]

    def spot_intersections(self):
        records = []
        widths = self.widths
        for identifier in np.unique(self.spot_ids[self.spot_ids > 0]):
            indices = np.flatnonzero(self.spot_ids == identifier)
            records.append(
                {
                    "spot_id": int(identifier),
                    "first_mm": float(self.enter[indices[0]] * self.ruler.length),
                    "last_mm": float(self.leave[indices[-1]] * self.ruler.length),
                    "observed_length_mm": float(np.sum(widths[indices])),
                    "pieces": int(np.count_nonzero(np.diff(indices) != 1) + 1),
                }
            )
        return sorted(records, key=lambda item: item["first_mm"])

    def as_dict(self, points=None):
        return {
            "ruler": self.ruler.as_dict(),
            "summary": self.summary(),
            "profile": self.profile(points),
            "intervals": self.intervals(),
            "spots": self.spot_intersections(),
        }


def cell_boundaries(centers):
    return centers[:-1] + np.diff(centers) * 0.5


def direction_events(boundaries, start, end):
    if start == end:
        return np.zeros(0, dtype=np.float64)
    lower, upper = sorted((start, end))
    first = int(np.searchsorted(boundaries, lower, side="right"))
    last = int(np.searchsorted(boundaries, upper, side="left"))
    return (boundaries[first:last] - start) / (end - start)


def crossing_order(ruler, row_edges, column_edges):
    intersections = np.sort(
        np.concatenate(
            (
                [0.0],
                direction_events(row_edges, ruler.start.station_mm, ruler.end.station_mm),
                direction_events(
                    column_edges, ruler.start.transverse_mm, ruler.end.transverse_mm
                ),
                [1.0],
            )
        )
    )
    distinct = np.r_[True, np.diff(intersections) > 1e-12]
    coordinates = intersections[distinct]
    coordinates[-1] = 1
    enter = coordinates[:-1]
    leave = coordinates[1:]
    s, x = ruler.interpolate((enter + leave) * 0.5)
    rows = np.searchsorted(row_edges, s, side="right")
    columns = np.searchsorted(column_edges, x, side="right")
    return enter, leave, rows, columns


def evaluate_ruler(context, ruler):
    grid = context.grid
    row_edges = cell_boundaries(grid.stations)
    column_edges = cell_boundaries(grid.transverse)
    enter, leave, rows, columns = crossing_order(ruler, row_edges, column_edges)
    area_valid = grid.valid[rows, columns]
    states = np.where(area_valid, PathSupport.BACKGROUND, PathSupport.BAD_PIXEL)
    on_material = context.edges.material[rows, columns]
    states[on_material] = PathSupport.MATERIAL
    states[grid.plan.support[rows] == Support.GAP] = PathSupport.GAP
    values = grid.values[rows, columns]
    spot_ids = context.spots.labels[rows, columns]
    if grid.transverse_calibrated is None:
        calibrated_cells = None
    else:
        calibrated_cells = grid.calibrated_mask[columns]
    return MeasuredPath(
        ruler, enter, leave, rows, columns, states, values, spot_ids, calibrated_cells
    )
