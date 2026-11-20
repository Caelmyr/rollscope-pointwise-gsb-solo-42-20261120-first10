import io
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw

from .domain import DISPLAY_HEIGHT, DISPLAY_WIDTH, StationWindow, UserInputError
from .grid import Support, index_near
from .numerics import true_runs


class Layer(str, Enum):
    SENSOR = "sensor"
    FLAT = "flat"
    STATION = "station"
    COVERAGE = "coverage"
    EDGES = "edges"
    SPOTS = "spots"
    STREAKS = "streaks"
    MARKS = "marks"
    SPLICES = "splices"
    TEXTURE = "texture"
    ILLUMINATION = "illumination"
    GRAIN = "grain"


ACQUISITION_LAYERS = {Layer.SENSOR, Layer.FLAT}
CATEGORY_STYLES = {
    "edge": (85, 244, 212),
    "dark": (255, 169, 88),
    "bright": (91, 237, 250),
    "mark": (239, 182, 92),
    "splice": (219, 137, 255),
    "texture": (138, 240, 205),
}


@dataclass(frozen=True)
class RasterSpec:
    layer: Layer
    max_width: int = DISPLAY_WIDTH
    max_height: int = DISPLAY_HEIGHT
    show_overlays: bool = True
    window: StationWindow | None = None


@dataclass(frozen=True)
class RasterMapping:
    rows: slice
    columns: slice
    height: int
    width: int

    @property
    def row_count(self):
        return self.rows.stop - self.rows.start

    @property
    def column_count(self):
        return self.columns.stop - self.columns.start

    @classmethod
    def make(cls, shape, max_width, max_height, rows=None, cols=None):
        rows = slice(0, shape[0]) if rows is None else rows
        cols = slice(0, shape[1]) if cols is None else cols
        in_rows = rows.stop - rows.start
        in_cols = cols.stop - cols.start
        if in_rows <= 0 or in_cols <= 0:
            raise ValueError("The view has no cells.")
        return cls(rows, cols, min(in_rows, max_height), min(in_cols, max_width))

    def pixel(self, row, column):
        x = ((column - self.columns.start + 0.5) * self.width / self.column_count) - 0.5
        y = ((row - self.rows.start + 0.5) * self.height / self.row_count) - 0.5
        return float(x), float(y)

    def cell(self, x, y):
        out_col = min(self.width - 1, int(x * self.width))
        out_row = min(self.height - 1, int(y * self.height))
        col = self.columns.start + (2 * out_col + 1) * self.column_count // (2 * self.width)
        row = self.rows.start + (2 * out_row + 1) * self.row_count // (2 * self.height)
        return int(row), int(col)

    def as_dict(self):
        return {
            "first_line": self.rows.start,
            "last_line": self.rows.stop - 1,
            "first_column": self.columns.start,
            "last_column": self.columns.stop - 1,
            "display_width": self.width,
            "display_height": self.height,
            "source_width": self.column_count,
            "source_height": self.row_count,
        }


def scaling_limits(array):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [2, 98])
    if high - low < 1e-6:
        return float(low), float(low + 1)
    return float(low), float(high)


def sample_indices(length, count):
    return (2 * np.arange(count) + 1) * length // (2 * count)


def map_limits(mapping):
    rows = sample_indices(mapping.row_count, mapping.height) + mapping.rows.start
    columns = sample_indices(mapping.column_count, mapping.width) + mapping.columns.start
    return rows, columns


def normalize(array, low, high):
    scaled = np.where(np.isfinite(array), (array - low) / (high - low), 0)
    return np.clip(scaled, 0, 1)


def gray_palette(data, low, high):
    scaled = normalize(data, low, high)
    color = np.stack(
        (20 + 211 * scaled, 24 + 212 * scaled, 31 + 208 * scaled), axis=2
    )
    return np.rint(color).astype(np.uint8)


def diverging_palette(data, extent):
    extent = max(0.001, extent)
    normalized = np.clip(np.nan_to_num(data) / extent, -1, 1)
    strength = np.sqrt(np.abs(normalized))[..., None]
    middle = np.array([35, 36, 44], dtype=np.float32)
    dark = np.array([249, 143, 73], dtype=np.float32)
    bright = np.array([70, 224, 223], dtype=np.float32)
    extreme = np.where(normalized[..., None] < 0, dark, bright)
    return np.rint((1 - strength) * middle + strength * extreme).astype(np.uint8)


def color_unobserved(picture, valid, row_gaps):
    y, x = np.ogrid[: picture.shape[0], : picture.shape[1]]
    patterned = ((x + y) % 8) < 3
    unknown = ~valid
    picture[unknown] = (47, 31, 43)
    picture[unknown & patterned] = (77, 42, 55)
    if row_gaps is not None:
        gap = np.broadcast_to(row_gaps[:, None], valid.shape)
        picture[gap] = (29, 32, 55)
        picture[gap & patterned] = (63, 68, 106)


def coverage_palette(support, valid):
    color_map = np.array(
        [[36, 39, 71], [64, 146, 127], [78, 101, 145]], dtype=np.uint8
    )
    image = np.broadcast_to(color_map[support][:, None, :], valid.shape + (3,)).copy()
    color_unobserved(image, valid, support == Support.GAP)
    return image


def position_mapping(context, spec):
    if spec.layer in ACQUISITION_LAYERS:
        return RasterMapping.make(
            context.capture.raw.shape, spec.max_width, spec.max_height
        )
    grid = context.grid
    if spec.window is None:
        rows = columns = None
    else:
        rows, columns = spec.window.indices(grid.stations, grid.transverse)
    return RasterMapping.make(grid.shape, spec.max_width, spec.max_height, rows, columns)


def raw_layer(context, indices, calibrated):
    if calibrated:
        surface = context.calibrated
        values = surface.values[indices]
        valid = surface.valid[indices]
        image = gray_palette(values, 0.02, 0.83)
        color_unobserved(image, valid, None)
        return image
    values = context.capture.raw[indices]
    low, high = scaling_limits(context.capture.raw)
    return gray_palette(values, low, high)


def grain_layer(context, comparison, rows, columns, spec):
    """Paint each target tile with its conformance class.

    Tiles are coloured by their comparison row. Cells that were not evaluated
    or that have unavailable samples keep the regular unknown hatching, so a
    thick band of "neither" can be told apart from an unmeasured area.
    """
    from .grain import CLASS_COLORS

    grid = context.grid
    colors = {
        "both_references": CLASS_COLORS["both_references"],
        "only_reference_a": CLASS_COLORS["only_reference_a"],
        "only_reference_b": CLASS_COLORS["only_reference_b"],
        "neither_reference": CLASS_COLORS["neither_reference"],
        "unknown": CLASS_COLORS["unknown"],
    }
    paint = np.empty(grid.shape + (3,), dtype=np.uint8)
    paint[:, :] = (26, 33, 46)
    if comparison is None:
        color_unobserved(paint, np.zeros(grid.shape, dtype=bool), None)
        return paint[rows[:, None], columns[None, :]].copy()
    window = comparison.window("target")
    for row in comparison.rows:
        tile = window.tiles[row["tile"] - 1]
        paint[
            tile.row_start : tile.row_stop, tile.column_start : tile.column_stop
        ] = colors[row["classification"]]
    picture = paint[rows[:, None], columns[None, :]].copy()
    support = grid.plan.support[rows]
    valid = grid.valid[rows[:, None], columns[None, :]]
    color_unobserved(picture, valid, support == Support.GAP)
    return picture


def grid_layer(context, indices, row_indices, layer):
    grid = context.grid
    support = grid.plan.support[row_indices]
    valid = grid.valid[indices]
    if layer == Layer.COVERAGE:
        return coverage_palette(support, valid)
    if layer == Layer.SPOTS:
        report = context.spots
        valid = report.detectable[indices]
        extent = context.settings.spots.contrast * 2
        pixels = diverging_palette(report.residual[indices], extent)
    elif layer == Layer.STREAKS:
        report = context.streaks
        valid = report.evidence[indices] & np.isfinite(report.residual[indices])
        extent = context.settings.streaks.contrast * 3
        pixels = diverging_palette(report.residual[indices], extent)
    else:
        pixels = gray_palette(grid.values[indices], 0.02, 0.83)
    color_unobserved(pixels, valid, support == Support.GAP)
    return pixels


def preview_image(context, spec, grain=None):
    mapping = position_mapping(context, spec)
    rows, columns = map_limits(mapping)
    indices = np.ix_(rows, columns)
    if spec.layer in ACQUISITION_LAYERS:
        pixels = raw_layer(context, indices, calibrated=spec.layer == Layer.FLAT)
    elif spec.layer == Layer.GRAIN:
        if grain is None:
            raise UserInputError("Choose two signed-off grain references first.")
        pixels = grain_layer(context, grain, rows, columns, spec)
    else:
        pixels = grid_layer(context, indices, rows, spec.layer)
    image = Image.fromarray(pixels)
    if spec.show_overlays and spec.layer not in ACQUISITION_LAYERS:
        ReviewDrawing(image, mapping, context).render(spec.layer)
    return image, mapping


def png_bytes(image):
    result = io.BytesIO()
    image.save(result, format="PNG", compress_level=4)
    return result.getvalue()


class ReviewDrawing:
    def __init__(self, image, mapping, context):
        self.mapping = mapping
        self.context = context
        self.draw = ImageDraw.Draw(image)

    def render(self, layer):
        handlers = {
            Layer.EDGES: self.edges,
            Layer.SPOTS: self.spots,
            Layer.STREAKS: self.streaks,
            Layer.MARKS: self.marks,
            Layer.SPLICES: self.splices,
            Layer.TEXTURE: self.texture,
        }
        painter = handlers.get(layer)
        if painter is not None:
            painter()

    def strip_box(self, first, last, left, right, color, padding=0):
        grid = self.context.grid
        x0 = index_near(grid.transverse, left)
        x1 = index_near(grid.transverse, right)
        y0 = index_near(grid.stations, first)
        y1 = index_near(grid.stations, last)
        if y1 < self.mapping.rows.start or y0 >= self.mapping.rows.stop:
            return
        if x1 < self.mapping.columns.start or x0 >= self.mapping.columns.stop:
            return
        a, b = self.mapping.pixel(y0, x0)
        c, d = self.mapping.pixel(y1, x1)
        self.draw.rectangle((a - padding, b - padding, c + padding, d + padding), outline=color)

    def transverse_mark(self, station, left, right, color):
        self.strip_box(station, station, left, right, color)

    def edges(self):
        edges = self.context.edges
        grid = self.context.grid
        start = self.mapping.rows.start
        stop = self.mapping.rows.stop
        for side in (edges.left_mm, edges.right_mm):
            columns = (side[start:stop] - grid.transverse[0]) / grid.column_pitch_mm
            for lower, upper in true_runs(np.isfinite(columns)):
                points = [
                    self.mapping.pixel(row + start, columns[row]) for row in range(lower, upper)
                ]
                if len(points) > 1:
                    self.draw.line(points, fill=CATEGORY_STYLES["edge"], width=2)
                else:
                    self.draw.point(points[0], fill=CATEGORY_STYLES["edge"])

    def spots(self):
        for spot in self.context.spots.spots:
            self.strip_box(
                spot.from_mm,
                spot.to_mm,
                spot.left_mm,
                spot.right_mm,
                CATEGORY_STYLES[spot.polarity],
                padding=1,
            )

    def streaks(self):
        for streak in self.context.streaks.streaks:
            self.strip_box(
                streak.from_mm,
                streak.to_mm,
                streak.left_mm,
                streak.right_mm,
                CATEGORY_STYLES[streak.polarity],
                padding=2,
            )

    def marks(self):
        grid = self.context.grid
        edges = self.context.edges
        settings = self.context.settings.marks
        for mark in self.context.marks.marks:
            left = edges.left_mm[mark.row]
            width = edges.width_mm[mark.row]
            x0 = left + settings.band_start_fraction * width
            x1 = left + settings.band_end_fraction * width
            if np.isfinite(x0) and np.isfinite(x1):
                self.transverse_mark(mark.station_mm, x0, x1, CATEGORY_STYLES["mark"])

    def splices(self):
        for splice in self.context.splices.splices:
            self.strip_box(
                splice.before_mm,
                splice.after_mm,
                splice.left_mm,
                splice.right_mm,
                CATEGORY_STYLES["splice"],
                padding=1,
            )

    def texture(self):
        active = self.context.texture.tiles
        step = max(1, len(active) // 110)
        grid = self.context.grid
        for tile in active[::step]:
            if tile.angle_deg is None:
                continue
            x = index_near(grid.transverse, tile.transverse_mm)
            y = index_near(grid.stations, tile.station_mm)
            start_x, start_y = self.mapping.pixel(y, x)
            theta = np.radians(tile.angle_deg)
            radius_mm = 0.21 * min(tile.to_mm - tile.from_mm, tile.right_mm - tile.left_mm)
            dx = radius_mm * np.sin(theta) / grid.column_pitch_mm
            dy = radius_mm * np.cos(theta) / grid.row_pitch_mm
            end_x, end_y = self.mapping.pixel(y + dy, x + dx)
            reflect = (2 * start_x - end_x, 2 * start_y - end_y)
            self.draw.line(
                (reflect, (end_x, end_y)),
                fill=CATEGORY_STYLES["texture"],
                width=1,
            )
