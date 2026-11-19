from dataclasses import dataclass

import numpy as np

from .calibration import safe_value, scale_summary
from .numerics import bin_quantiles


@dataclass(frozen=True)
class IlluminationReport:
    station: np.ndarray
    transverse: np.ndarray
    longitudinal: np.ndarray
    cross_web: np.ndarray
    support_per_line: np.ndarray
    support_per_column: np.ndarray
    hist_edges: np.ndarray
    hist_counts: np.ndarray
    trend: tuple
    drift_per_meter: float | None

    def summary(self):
        return {
            "material_line_median": scale_summary(self.longitudinal),
            "cross_web_median": scale_summary(self.cross_web),
            "evaluated_rows": int(np.count_nonzero(np.isfinite(self.longitudinal))),
            "evaluated_columns": int(np.count_nonzero(np.isfinite(self.cross_web))),
            "change_per_meter": self.drift_per_meter,
            "histogram_samples": int(np.sum(self.hist_counts)),
        }

    def row_series(self, indices):
        return [
            {
                "station_mm": float(self.station[index]),
                "reflectance": safe_value(self.longitudinal[index]),
                "eligible_pixels": int(self.support_per_line[index]),
            }
            for index in indices
        ]

    def column_series(self):
        return [
            {
                "transverse_mm": float(position),
                "reflectance": safe_value(self.cross_web[index]),
                "eligible_pixels": int(self.support_per_column[index]),
            }
            for index, position in enumerate(self.transverse)
        ]

    def histogram(self):
        return [
            {
                "from": float(start),
                "to": float(stop),
                "count": int(count),
            }
            for start, stop, count in zip(self.hist_edges[:-1], self.hist_edges[1:], self.hist_counts)
        ]


def axis_medians(values, valid, min_support, orientation):
    if orientation == "column":
        values = values.T
        valid = valid.T
    observations = np.count_nonzero(valid, axis=1)
    medians = np.full(len(values), np.nan, dtype=np.float32)
    for index in np.flatnonzero(observations >= min_support):
        medians[index] = np.median(values[index, valid[index]])
    return medians, observations


def robust_endpoint_drift(trend):
    valid = [section for section in trend if section["median"] is not None]
    if len(valid) < 4:
        return None
    first = valid[:2]
    last = valid[-2:]
    from_mm = np.mean([(item["from_mm"] + item["to_mm"]) / 2 for item in first])
    to_mm = np.mean([(item["from_mm"] + item["to_mm"]) / 2 for item in last])
    starting = np.mean([item["median"] for item in first])
    ending = np.mean([item["median"] for item in last])
    return float(1000 * (ending - starting) / (to_mm - from_mm))


def reflectance_histogram(values, mask, bins=48):
    finite = values[mask]
    if finite.size:
        lower = min(0, float(np.min(finite)))
        upper = max(1, float(np.max(finite)))
    else:
        lower, upper = 0, 1
    return np.histogram(finite, bins=bins, range=(lower, upper))


def measure_illumination(grid, edges):
    mask = edges.inner_mask(max(2, int(np.ceil(2.5 / grid.column_pitch_mm))))
    width = grid.shape[1]
    along, along_counts = axis_medians(grid.values, mask, max(6, width // 8), "line")
    across, across_counts = axis_medians(
        grid.values, mask, max(8, grid.shape[0] // 8), "column"
    )
    bins = max(2, min(500, int(np.ceil(23 / grid.row_pitch_mm))))
    trend = tuple(bin_quantiles(along, np.isfinite(along), grid.stations, bins))
    hist_counts, hist_edges = reflectance_histogram(grid.values, mask)
    return IlluminationReport(
        grid.stations,
        grid.transverse,
        along,
        across,
        along_counts,
        across_counts,
        hist_edges,
        hist_counts,
        trend,
        robust_endpoint_drift(trend),
    )
