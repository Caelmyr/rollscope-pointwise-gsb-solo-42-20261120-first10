from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import safe_value, scale_summary
from .numerics import segmented_reference


MAX_SPOTS = 500


@dataclass(frozen=True)
class Spot:
    id: int
    polarity: str
    pixel_count: int
    area_mm2: float
    peak_contrast: float
    mean_contrast: float
    station_mm: float
    transverse_mm: float
    from_mm: float
    to_mm: float
    left_mm: float
    right_mm: float
    partial: bool
    row_slice: slice
    column_slice: slice

    def as_dict(self):
        return {
            "id": self.id,
            "polarity": self.polarity,
            "pixel_count": self.pixel_count,
            "area_mm2": self.area_mm2,
            "peak_contrast": self.peak_contrast,
            "mean_contrast": self.mean_contrast,
            "station_mm": self.station_mm,
            "transverse_mm": self.transverse_mm,
            "from_mm": self.from_mm,
            "to_mm": self.to_mm,
            "left_mm": self.left_mm,
            "right_mm": self.right_mm,
            "partial": self.partial,
        }


@dataclass(frozen=True)
class SpotReport:
    residual: np.ndarray
    detectable: np.ndarray
    labels: np.ndarray
    spots: tuple
    candidate_count: int

    def summary(self):
        return {
            "dark_spots": sum(spot.polarity == "dark" for spot in self.spots),
            "bright_spots": sum(spot.polarity == "bright" for spot in self.spots),
            "partial_spots": sum(spot.partial for spot in self.spots),
            "candidate_count": self.candidate_count,
            "truncated": self.candidate_count > len(self.spots),
            "inspection_pixels": int(np.count_nonzero(self.detectable)),
            "area_mm2": scale_summary([spot.area_mm2 for spot in self.spots]),
        }

    def table(self):
        return [spot.as_dict() for spot in self.spots]

    def spot_at(self, row, column):
        label = int(self.labels[row, column])
        return self.spots[label - 1] if label else None

    def pixel(self, row, column):
        spot = self.spot_at(row, column)
        return {
            "detectable": bool(self.detectable[row, column]),
            "signed_contrast": safe_value(self.residual[row, column]),
            "spot_id": spot.id if spot else None,
        }


@dataclass(frozen=True)
class Component:
    polarity: str
    source: int
    peak: float
    rows: slice
    columns: slice


def component_candidates(residual, mask, contrast, minimum_pixels):
    components = []
    maps = {}
    for sign, polarity in ((1, "bright"), (-1, "dark")):
        target = mask & (residual * sign >= contrast)
        labels, count = ndimage.label(target)
        maps[polarity] = labels
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        boxes = ndimage.find_objects(labels)
        for index in np.flatnonzero(sizes[1:] >= minimum_pixels):
            source = int(index + 1)
            rows, columns = boxes[index]
            signed_patch = residual[rows, columns] * sign
            peak = max(signed_patch[labels[rows, columns] == source])
            components.append(Component(polarity, source, float(peak), rows, columns))
    return components, maps


def describe_component(component, masks, residual, unknown_adjacent, grid, output, identifier):
    rows = component.rows
    columns = component.columns
    selected = masks[component.polarity][rows, columns] == component.source
    local_y, local_x = np.nonzero(selected)
    y = local_y + rows.start
    x = local_x + columns.start

    output[rows, columns][selected] = identifier
    contrast = np.abs(residual[y, x]).astype(np.float64)
    size = len(y)
    height = grid.row_pitch_mm
    width = grid.column_pitch_mm
    station_mm = float(np.average(grid.stations[y], weights=contrast))
    transverse_mm = float(np.average(grid.transverse[x], weights=contrast))
    return Spot(
        id=identifier,
        polarity=component.polarity,
        pixel_count=size,
        area_mm2=float(size * height * width),
        peak_contrast=component.peak,
        mean_contrast=float(np.mean(contrast)),
        station_mm=station_mm,
        transverse_mm=transverse_mm,
        from_mm=float(grid.stations[rows.start]),
        to_mm=float(grid.stations[rows.stop - 1]),
        left_mm=float(grid.transverse[columns.start]),
        right_mm=float(grid.transverse[columns.stop - 1]),
        partial=bool(np.any(unknown_adjacent[y, x])),
        row_slice=rows,
        column_slice=columns,
    )


def detect_spots(grid, edges, settings):
    margin = int(np.ceil(settings.edge_guard_mm / grid.column_pitch_mm))
    material = edges.inner_mask(margin)
    sigma = (
        settings.background_mm / grid.row_pitch_mm,
        settings.background_mm / grid.column_pitch_mm,
    )
    smooth, support = segmented_reference(grid.values, material, sigma, 0.45)
    detectable = material & (support >= 0.45)
    residual = np.where(detectable, grid.values - smooth, np.nan)
    minimum_pixels = max(
        1, int(np.ceil(settings.minimum_area_mm2 / (grid.column_pitch_mm * grid.row_pitch_mm)))
    )
    candidates, labels_by_polarity = component_candidates(
        residual, detectable, settings.contrast, minimum_pixels
    )
    selected = sorted(candidates, key=lambda item: (-item.peak, item.rows.start, item.columns.start))
    selected = sorted(
        selected[:MAX_SPOTS],
        key=lambda item: (item.rows.start, item.columns.start, -item.peak),
    )
    adjacency = ndimage.binary_dilation(~detectable, border_value=1)
    labels = np.zeros(grid.shape, dtype=np.int32)
    spots = tuple(
        describe_component(
            component, labels_by_polarity, residual, adjacency, grid, labels, identifier
        )
        for identifier, component in enumerate(selected, 1)
    )
    return SpotReport(residual, detectable, labels, spots, len(candidates))
