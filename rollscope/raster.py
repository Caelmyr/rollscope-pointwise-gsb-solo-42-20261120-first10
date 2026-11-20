import io
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw

from .blocks import BlockGrid
from .domain import DISPLAY_HEIGHT, DISPLAY_WIDTH, StationWindow
from .grid import Support, index_near
from .numerics import true_runs


# Marker colours used when only a minority of a block carries the evidence.
MARKER_GAP = (96, 105, 168)       # encoder gap / unsupported station
MARKER_BAD = (196, 110, 184)      # invalid sample inside an observed block
MARKER_DARK = (255, 169, 88)      # dark anomaly (spot/streak)
MARKER_BRIGHT = (91, 237, 250)    # bright anomaly (spot/streak)
# A minority marker never fills its block; minority fractions above this would
# be painted as the normal block colour instead.
MINORITY_LIMIT = 0.5


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
    region: StationWindow | None = None


@dataclass(frozen=True)
class RasterMapping:
    rows: slice
    columns: slice
    height: int
    width: int
    block_grid: BlockGrid | None = None

    @property
    def row_count(self):
        return self.rows.stop - self.rows.start

    @property
    def column_count(self):
        return self.columns.stop - self.columns.start

    @property
    def block_mode(self):
        return self.block_grid is not None and self.block_grid.downsampled

    @classmethod
    def make(cls, shape, max_width, max_height, rows=None, cols=None):
        rows = slice(0, shape[0]) if rows is None else rows
        cols = slice(0, shape[1]) if cols is None else cols
        in_rows = rows.stop - rows.start
        in_cols = cols.stop - cols.start
        if in_rows <= 0 or in_cols <= 0:
            raise ValueError("The view has no cells.")
        height = min(in_rows, max_height)
        width = min(in_cols, max_width)
        block_grid = BlockGrid.make(rows, cols, max_height, max_width)
        return cls(rows, cols, height, width, block_grid)

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

    def block_at_pixel(self, x, y):
        """Display pixel coordinates -> block index (block mode only)."""
        if not self.block_mode:
            return None
        block_row = min(self.block_grid.block_rows - 1, int(y * self.height))
        block_col = min(self.block_grid.block_columns - 1, int(x * self.width))
        return max(0, block_row), max(0, block_col)

    def block_source_bounds(self, block_row, block_col):
        """Source row/column ranges covered by one display block."""
        r0, r1 = self.block_grid.row_bounds(block_row)
        c0, c1 = self.block_grid.column_bounds(block_col)
        return slice(r0, r1), slice(c0, c1)

    def as_dict(self):
        data = {
            "first_line": self.rows.start,
            "last_line": self.rows.stop - 1,
            "first_column": self.columns.start,
            "last_column": self.columns.stop - 1,
            "display_width": self.width,
            "display_height": self.height,
            "source_width": self.column_count,
            "source_height": self.row_count,
        }
        if self.block_mode:
            data.update(
                block_mode=True,
                block_rows=self.block_grid.block_rows,
                block_columns=self.block_grid.block_columns,
            )
        else:
            data["block_mode"] = False
        return data


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


def color_block_unobserved(image, valid, gap_mask):
    """Block-level equivalent of color_unobserved with 2D block masks."""
    y, x = np.ogrid[: image.shape[0], : image.shape[1]]
    patterned = ((x + y) % 8) < 3
    unknown = ~valid
    image[unknown] = (47, 31, 43)
    image[unknown & patterned] = (77, 42, 55)
    image[gap_mask] = (29, 32, 55)
    image[gap_mask & patterned] = (63, 68, 106)


def position_mapping(context, spec):
    if spec.layer in ACQUISITION_LAYERS:
        return RasterMapping.make(
            context.capture.raw.shape, spec.max_width, spec.max_height
        )
    grid = context.grid
    frame = spec.region or spec.window
    if frame is None:
        rows = columns = None
    else:
        rows, columns = frame.indices(grid.stations, grid.transverse)
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


def _coverage_block_color(facts):
    """Dominant support category per block, mixed when no category dominates."""
    total = np.maximum(facts.total_cells, 1)
    native = facts.native_cells / total
    blend = facts.blend_cells / total
    gap = facts.gap_cells / total
    color_map = np.array(
        [[36, 39, 71], [64, 146, 127], [78, 101, 145]], dtype=np.float64
    )
    image = np.zeros(facts.shape + (3,), dtype=np.float64)
    image += native[..., None] * color_map[Support.NATIVE]
    image += blend[..., None] * color_map[Support.BLEND]
    image += gap[..., None] * color_map[Support.GAP]
    empty = facts.total_cells == 0
    image[empty] = color_map[Support.GAP]
    return np.rint(image).astype(np.uint8)


def _minority_positions(fraction):
    """Blocks whose evidence is present but does not fill the whole block."""
    return (fraction > 0) & (fraction < MINORITY_LIMIT)


def paint_minority_markers(image, facts, layer, context):
    """Keep sub-cell evidence visible without expanding it to a full block.

    The block colour already carries the median trend.  A block that is only
    partly unsupported or partly anomalous receives a marker colour instead
    of the trend colour so a one-cell missing line or a narrow spot survives
    the reduction.  These markers never replace a block that is mostly
    unsupported, which keeps broad gaps rendered with their normal colour.
    """
    gap_minority = _minority_positions(facts.gap_fraction())
    bad_minority = _minority_positions(facts.bad_fraction()) & ~gap_minority
    if np.any(gap_minority):
        image[gap_minority] = MARKER_GAP
    if np.any(bad_minority):
        image[bad_minority] = MARKER_BAD

    if layer in (Layer.SPOTS, Layer.STREAKS):
        # Only a contrast extreme strong enough to count as an anomaly earns a
        # marker; weaker residual noise must leave the median trend visible.
        threshold = (
            context.settings.spots.contrast
            if layer == Layer.SPOTS
            else context.settings.streaks.contrast
        )
        mostly_sound = (facts.gap_fraction() < MINORITY_LIMIT) & (
            facts.bad_fraction() < MINORITY_LIMIT
        )
        bright = mostly_sound & (facts.contrast_max >= threshold)
        dark = mostly_sound & ~bright & (facts.contrast_min <= -threshold)
        if np.any(bright):
            image[bright] = MARKER_BRIGHT
        if np.any(dark):
            image[dark] = MARKER_DARK
    return image


def block_preview(context, spec, mapping, facts):
    """Render one display pixel per block, preserving minority evidence."""
    layer = spec.layer
    mostly_gap = facts.gap_fraction() >= MINORITY_LIMIT
    if layer == Layer.COVERAGE:
        # Coverage already encodes every support category proportionally, so
        # minority gaps stay their own category colour instead of a marker.
        image = _coverage_block_color(facts)
    elif layer in (Layer.SPOTS, Layer.STREAKS):
        extent = (
            context.settings.spots.contrast * 2
            if layer == Layer.SPOTS
            else context.settings.streaks.contrast * 3
        )
        median = np.where(np.isfinite(facts.median), facts.median, 0.0)
        image = diverging_palette(median, extent)
        color_block_unobserved(image, facts.valid_fraction() > 0, mostly_gap)
        image = paint_minority_markers(image, facts, layer, context)
    else:
        image = gray_palette(facts.median, 0.02, 0.83)
        color_block_unobserved(image, facts.valid_fraction() > 0, mostly_gap)
        image = paint_minority_markers(image, facts, layer, context)

    picture = Image.fromarray(image)
    if spec.show_overlays:
        ReviewDrawing(picture, mapping, context).render(layer)
    return picture


def preview_image(context, spec):
    mapping = position_mapping(context, spec)
    if spec.layer not in ACQUISITION_LAYERS and mapping.block_mode:
        facts = context.block_facts(spec.layer, mapping.block_grid)
        return block_preview(context, spec, mapping, facts), mapping
    rows, columns = map_limits(mapping)
    indices = np.ix_(rows, columns)
    if spec.layer in ACQUISITION_LAYERS:
        pixels = raw_layer(context, indices, calibrated=spec.layer == Layer.FLAT)
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
