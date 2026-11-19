from dataclasses import asdict, dataclass

import numpy as np

from .calibration import scale_summary
from .grid import Support
from .numerics import true_runs


@dataclass(frozen=True)
class Gap:
    id: int
    first_index: int
    last_index: int
    from_mm: float
    to_mm: float
    grid_cells: int
    width_mm: float
    upstream_tick: int
    downstream_tick: int
    upstream_line: int
    downstream_line: int

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CoverageReport:
    shape: tuple
    covered_columns: np.ndarray
    observed_pixels: np.ndarray
    support: np.ndarray
    stations: np.ndarray
    gaps: tuple

    def summary(self):
        mixed_rows = (self.observed_pixels > 0) & (self.observed_pixels < self.shape[1])
        return {
            "image_coverage": float(np.sum(self.observed_pixels) / np.prod(self.shape)),
            "encoder_gap_count": len(self.gaps),
            "longest_gap_mm": max((gap.width_mm for gap in self.gaps), default=0),
            "partially_valid_rows": int(np.count_nonzero(mixed_rows)),
            "zero_pixel_rows": int(np.count_nonzero(self.observed_pixels == 0)),
            "imaging_columns": int(np.count_nonzero(self.covered_columns)),
            "column_support_fraction": scale_summary(self.covered_columns / self.shape[0]),
        }

    def series(self, indices):
        return [
            {
                "row": int(index),
                "station_mm": float(self.stations[index]),
                "support": Support(self.support[index]).name.lower(),
                "usable_pixels": int(self.observed_pixels[index]),
                "usable_fraction": float(self.observed_pixels[index] / self.shape[1]),
            }
            for index in indices
        ]

    def table(self):
        return [gap.as_dict() for gap in self.gaps]


def derive_gaps(grid):
    for identifier, (start, stop) in enumerate(true_runs(~grid.plan.observed), 1):
        left_idx = grid.plan.left[start]
        right_idx = grid.plan.right[stop - 1]
        upstream = grid.groups.sources(left_idx)
        downstream = grid.groups.sources(right_idx)
        yield Gap(
            identifier,
            int(start),
            int(stop - 1),
            float(grid.stations[start]),
            float(grid.stations[stop - 1]),
            int(stop - start),
            float((stop - start) * grid.row_pitch_mm),
            int(grid.groups.ticks[left_idx]),
            int(grid.groups.ticks[right_idx]),
            upstream.stop - 1,
            downstream.start,
        )


def inspect_coverage(grid):
    return CoverageReport(
        grid.shape,
        np.count_nonzero(grid.valid, axis=0),
        np.count_nonzero(grid.valid, axis=1),
        grid.plan.support,
        grid.stations,
        tuple(derive_gaps(grid)),
    )
