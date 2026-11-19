from dataclasses import dataclass

import numpy as np

from .calibration import scale_summary
from .grid import Support
from .numerics import true_runs
from .window_profile import profile_window, within_line_contrast


@dataclass(frozen=True)
class WindowCells:
    rows: slice
    columns: slice
    stations: np.ndarray
    transverse: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    material: np.ndarray
    row_support: np.ndarray

    @classmethod
    def take(cls, context, region):
        grid = context.grid
        rows, columns = region.indices(grid.stations, grid.transverse)
        return cls(
            rows,
            columns,
            grid.stations[rows],
            grid.transverse[columns],
            grid.values[rows, columns],
            grid.valid[rows, columns],
            context.edges.material[rows, columns],
            grid.plan.support[rows],
        )

    @property
    def size(self):
        return self.values.size

    def sampled_extent(self):
        return {
            "row_start": self.rows.start,
            "row_end": self.rows.stop - 1,
            "column_start": self.columns.start,
            "column_end": self.columns.stop - 1,
            "station_start_mm": float(self.stations[0]),
            "station_end_mm": float(self.stations[-1]),
            "transverse_start_mm": float(self.transverse[0]),
            "transverse_end_mm": float(self.transverse[-1]),
        }


def window_gaps(window):
    for start, stop in true_runs(window.row_support == Support.GAP):
        yield {
            "first_row": int(start + window.rows.start),
            "last_row": int(stop + window.rows.start - 1),
            "from_mm": float(window.stations[start]),
            "to_mm": float(window.stations[stop - 1]),
            "cells": int(stop - start),
        }


def contours_in_window(context, cells):
    labels = context.spots.labels[cells.rows, cells.columns]
    chosen, count = np.unique(labels[labels > 0], return_counts=True)
    cell_area = context.grid.row_pitch_mm * context.grid.column_pitch_mm
    observations = []
    for identifier, hits in zip(chosen, count):
        spot = context.spots.spots[int(identifier - 1)]
        observations.append(
            {
                "spot_id": int(identifier),
                "affected_cells": int(hits),
                "observed_area_mm2": float(hits * cell_area),
                "partial_in_window": bool(hits < spot.pixel_count),
                "partial_in_capture": spot.partial,
                "polarity": spot.polarity,
            }
        )
    return observations


def support_summary(cells, row_pitch, column_pitch):
    known = int(np.count_nonzero(cells.valid))
    material = int(np.count_nonzero(cells.material))
    gaps = list(window_gaps(cells))
    return {
        "rows": len(cells.stations),
        "columns": len(cells.transverse),
        "known_cells": known,
        "material_cells": material,
        "known_fraction": known / cells.size,
        "material_fraction": material / cells.size,
        "gap_count": len(gaps),
        "sampled_gap_length_mm": sum(gap["cells"] for gap in gaps) * row_pitch,
        "observed_material_area_mm2": material * row_pitch * column_pitch,
        "material_reflectance": scale_summary(cells.values[cells.material]),
    }


def source_line_set(grid, rows):
    group_indices = set()
    for row in range(rows.start, rows.stop):
        support = Support(grid.plan.support[row])
        if support == Support.GAP:
            continue
        group_indices.add(int(grid.plan.left[row]))
        if support == Support.BLEND:
            group_indices.add(int(grid.plan.right[row]))
    return sorted(
        line for group in group_indices for line in grid.groups.sources(group)
    )


def source_rows(context, rows):
    capture = context.capture
    lines = source_line_set(context.grid, rows)
    stations = capture.stations
    return [
        {
            "source_line": line,
            "tick": int(capture.ticks[line]),
            "station_mm": float(stations[line]),
            "capture_time_s": float(capture.seconds[line]),
        }
        for line in lines
    ]


def row_provenance(grid, rows):
    for row in range(rows.start, rows.stop):
        record = grid.provenance(row)
        sources = record.pop("sources")
        if not sources:
            yield record | {
                "tick": None,
                "weight": None,
                "first_source_line": None,
                "source_line_count": 0,
            }
        for source in sources:
            yield record | {
                "tick": source["tick"],
                "weight": source["weight"],
                "first_source_line": source["first_line"],
                "source_line_count": source["line_count"],
            }


def review_window(context, window):
    cells = WindowCells.take(context, window.region)
    numbers = support_summary(cells, context.grid.row_pitch_mm, context.grid.column_pitch_mm)
    spots = contours_in_window(context, cells)
    lines = source_line_set(context.grid, cells.rows)
    profiles = profile_window(cells)
    metrics = {
        "contributing_source_lines": len(lines),
        "p90_within_line_contrast": within_line_contrast(cells),
    }
    return {
        "window": window.as_dict(),
        "sampled_extent": cells.sampled_extent(),
        "summary": numbers | metrics,
        "profiles": profiles.as_dict(),
        "gaps": list(window_gaps(cells)),
        "spots": spots,
    }
