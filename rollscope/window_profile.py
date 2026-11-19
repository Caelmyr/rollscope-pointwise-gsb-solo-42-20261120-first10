from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from .grid import Support


MAX_PROFILE_POINTS = 90
MIN_MATERIAL_SHARE = 0.34


class BandStatus(str, Enum):
    OBSERVED = "observed"
    SPARSE = "sparse"
    EMPTY = "empty"
    GAP = "encoder_gap"


@dataclass(frozen=True)
class ProfileBand:
    start: float
    end: float
    center: float
    sample_count: int
    usable_fraction: float
    material_fraction: float
    median: float | None
    lower_quartile: float | None
    upper_quartile: float | None
    status: BandStatus

    def as_dict(self):
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class WindowProfiles:
    along: tuple
    across: tuple

    def as_dict(self):
        return {
            "station_bands": [item.as_dict() for item in self.along],
            "transverse_bands": [item.as_dict() for item in self.across],
        }


def transitions(size, maximum, mandatory=()):
    edges = np.linspace(0, size, min(size, maximum) + 1, dtype=np.int64)
    all_edges = np.unique(np.r_[edges, mandatory]).astype(np.int64)
    return zip(all_edges[:-1], all_edges[1:])


def unknown_transitions(status):
    gap = status == Support.GAP
    return np.flatnonzero(gap[:-1] != gap[1:]) + 1


def profile_bands(array, observed, material, positions, intervals, row_support=None):
    for start, stop in intervals:
        patch = array[start:stop]
        known = observed[start:stop]
        relevant = material[start:stop]
        geometry = positions[start:stop]
        count = int(np.count_nonzero(relevant))
        valid = int(np.count_nonzero(known))
        usage = count / patch.size
        center = (geometry[0] + geometry[-1]) / 2
        median = lower = upper = None
        if row_support is not None and np.all(row_support[start:stop] == Support.GAP):
            state = BandStatus.GAP
        elif valid == 0:
            state = BandStatus.EMPTY
        elif usage < MIN_MATERIAL_SHARE:
            state = BandStatus.SPARSE
        else:
            state = BandStatus.OBSERVED
            lower, median, upper = map(float, np.percentile(patch[relevant], [25, 50, 75]))
        yield ProfileBand(
            float(geometry[0]),
            float(geometry[-1]),
            float(center),
            count,
            valid / patch.size,
            usage,
            median,
            lower,
            upper,
            state,
        )


def profile_window(cells):
    along_bounds = transitions(
        len(cells.stations), MAX_PROFILE_POINTS, unknown_transitions(cells.row_support)
    )
    along = profile_bands(
        cells.values,
        cells.valid,
        cells.material,
        cells.stations,
        along_bounds,
        cells.row_support,
    )
    across = profile_bands(
        cells.values.T,
        cells.valid.T,
        cells.material.T,
        cells.transverse,
        transitions(len(cells.transverse), MAX_PROFILE_POINTS),
    )
    return WindowProfiles(tuple(along), tuple(across))


def within_line_contrast(cells):
    material = cells.material
    rows = np.count_nonzero(material, axis=1)
    treated = np.flatnonzero(rows >= max(4, int(cells.values.shape[1] * MIN_MATERIAL_SHARE)))
    if not len(treated):
        return None
    cell_weights = []
    values = []
    for index in treated:
        candidate = cells.values[index, material[index]]
        amplitude = float(np.percentile(candidate, 90) - np.percentile(candidate, 10))
        if np.isfinite(amplitude):
            cell_weights.append(rows[index])
            values.append(amplitude)
    if not values:
        return None
    values = np.array(values)
    weights = np.array(cell_weights, dtype=np.float64)
    sort_index = np.argsort(values, kind="stable")
    ordered = values[sort_index]
    fraction = np.cumsum(weights[sort_index]) / np.sum(weights)
    return float(ordered[np.searchsorted(fraction, 0.9, side="left")])
