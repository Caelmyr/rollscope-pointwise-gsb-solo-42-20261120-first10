"""Grain conformance against two signed-off windows.

Each tile is a patch of physical size. Its row and column spans are rounded
from the requested millimetres using that axis' pitch, and the FFT frequency
axes use the same pitches, so wavelengths and tile sizes stay in millimetres
where the detector pitch differs between axes.

Two things are kept out of the classification:

* the tile's local brightness. A linear plane is removed before the spectrum is
  taken; the tile's spectral amplitude is reported relative to its own median
  reflectance. A slow lighting change across the detector therefore cannot
  read as a grain change.
* the absolute spectral power. Direction is circular and coarseness is the
  power-weighted geometric mean wavelength inside the tile's own spectrum.
  Both are ratios within a tile, so brightness and exposure drop out.

A signed-off envelope is the robust bulk of that window's tiles: a coherence
weighted circular median direction and median coarseness, with coincidence
bounds widened by the window's own spread and protected by a minimum
tolerance. Tiles far outside the bulk of their own window are trimmed before
the bounds are set, so one join or scratch inside a signed-off span does not
widen the whole envelope. Trimmed tiles stay visible in that window's table.

Tiles with unavailable samples, material outside the strip, a detected surface
feature across the patch, or no visible grain direction are unknown. They are
never interpolated into a class. A reference with too few measured tiles cannot
support a classification.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .calibration import safe_value, scale_summary
from .domain import UserInputError, exact_keys, expect_object
from .texture import balanced_sampling, start_cells


MAX_WINDOW_TILES = 3000
WAVE_BANDS = 12
ENCIRCLED_OFFSET = 90.0
ROLES = ("reference_a", "reference_b", "target")


class TileState(str, Enum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported_sampling"
    OUTSIDE = "outside_material"
    FAINT = "grain_not_visible"
    FEATURE = "surface_feature"


class TileClass(str, Enum):
    BOTH = "both_references"
    ONLY_A = "only_reference_a"
    ONLY_B = "only_reference_b"
    NEITHER = "neither_reference"
    UNKNOWN = "unknown"


CLASS_COLORS = {
    TileClass.BOTH.value: (64, 196, 159),
    TileClass.ONLY_A.value: (232, 178, 92),
    TileClass.ONLY_B.value: (150, 164, 244),
    TileClass.NEITHER.value: (233, 111, 104),
    TileClass.UNKNOWN.value: (74, 84, 104),
}

CLASS_TITLES = {
    TileClass.BOTH.value: "Supports both references",
    TileClass.ONLY_A.value: "Only reference A",
    TileClass.ONLY_B.value: "Only reference B",
    TileClass.NEITHER.value: "Neither reference",
    TileClass.UNKNOWN.value: "Unknown",
}


@dataclass(frozen=True)
class GrainAxes:
    selected: np.ndarray
    unit_vector: np.ndarray
    wavelength_mm: np.ndarray
    band_index: np.ndarray
    wave_edges: np.ndarray


@dataclass(frozen=True)
class GrainFeature:
    direction_deg: float
    coherence: float
    coarseness_mm: float
    rms: float
    bands: tuple


@dataclass(frozen=True)
class GrainTile:
    id: int
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    station_from_mm: float
    station_to_mm: float
    transverse_from_mm: float
    transverse_to_mm: float
    station_mm: float
    transverse_mm: float
    height_mm: float
    width_mm: float
    reflectance: float | None
    state: TileState
    direction_deg: float | None
    coherence: float | None
    coarseness_mm: float | None
    relative_amplitude: float | None
    bands: tuple

    @property
    def measured(self):
        return self.state == TileState.MEASURED

    def evidence(self):
        return {
            "tile": self.id,
            "row_start": self.row_start,
            "row_end": self.row_stop - 1,
            "column_start": self.column_start,
            "column_end": self.column_stop - 1,
            "station_from_mm": self.station_from_mm,
            "station_to_mm": self.station_to_mm,
            "transverse_from_mm": self.transverse_from_mm,
            "transverse_to_mm": self.transverse_to_mm,
            "station_mm": self.station_mm,
            "transverse_mm": self.transverse_mm,
            "height_mm": self.height_mm,
            "width_mm": self.width_mm,
            "reflectance": self.reflectance,
            "state": self.state.value,
            "direction_deg": self.direction_deg,
            "coherence": self.coherence,
            "coarseness_mm": self.coarseness_mm,
            "relative_amplitude": self.relative_amplitude,
        }

    def table_row(self):
        result = self.evidence()
        for key in (
            "row_start",
            "row_end",
            "column_start",
            "column_end",
            "station_from_mm",
            "station_to_mm",
            "transverse_from_mm",
            "transverse_to_mm",
            "height_mm",
            "width_mm",
        ):
            result.pop(key)
        return result

    def band_rows(self, wave_edges):
        return [
            {
                "from_mm": float(wave_edges[index]),
                "to_mm": float(wave_edges[index + 1]),
                "fraction": float(self.bands[index]),
            }
            for index in range(WAVE_BANDS)
        ]


@dataclass(frozen=True)
class GrainWindow:
    window_id: str
    capture_ref: str
    capture_name: str
    label: str
    region: dict
    tiles: tuple
    tile_rows: int
    tile_columns: int
    row_stride: int
    column_stride: int
    total_positions: int
    wave_edges: np.ndarray

    @property
    def measured_tiles(self):
        return [tile for tile in self.tiles if tile.measured]

    def counts(self):
        counts = {state.value: 0 for state in TileState}
        for tile in self.tiles:
            counts[tile.state.value] += 1
        return counts

    def summary(self):
        counts = self.counts()
        return {
            "window": self.label,
            "window_id": self.window_id,
            "capture": self.capture_name,
            "capture_ref": self.capture_ref,
            "station_start_mm": self.region["station_start_mm"],
            "station_end_mm": self.region["station_end_mm"],
            "transverse_start_mm": self.region["transverse_start_mm"],
            "transverse_end_mm": self.region["transverse_end_mm"],
            "tile_height_mm": self.region["tile_height_mm"],
            "tile_width_mm": self.region["tile_width_mm"],
            "tile_rows": self.tile_rows,
            "tile_columns": self.tile_columns,
            "candidate_tiles": self.total_positions,
            "evaluated_tiles": len(self.tiles),
            "measured_tiles": counts[TileState.MEASURED.value],
            "unsupported_tiles": counts[TileState.UNSUPPORTED.value],
            "outside_tiles": counts[TileState.OUTSIDE.value],
            "faint_tiles": counts[TileState.FAINT.value],
            "feature_tiles": counts[TileState.FEATURE.value],
        }


@dataclass(frozen=True)
class GrainEnvelope:
    key: str
    window_id: str
    capture_ref: str
    label: str
    status: str
    measured_tiles: int
    retained_tiles: int
    trimmed_tiles: int
    direction_center_deg: float | None
    direction_tolerance_deg: float | None
    coherence_median: float | None
    coarseness_center_mm: float | None
    coarseness_tolerance_ratio: float | None
    relative_amplitude: dict

    @property
    def usable(self):
        return self.status == "usable"

    def as_dict(self):
        return {
            "key": self.key,
            "window_id": self.window_id,
            "capture_ref": self.capture_ref,
            "window": self.label,
            "status": self.status,
            "usable": self.usable,
            "measured_tiles": self.measured_tiles,
            "retained_tiles": self.retained_tiles,
            "trimmed_tiles": self.trimmed_tiles,
            "direction_center_deg": self.direction_center_deg,
            "direction_tolerance_deg": self.direction_tolerance_deg,
            "direction_range_deg": (
                None
                if self.direction_tolerance_deg is None
                else 2 * self.direction_tolerance_deg
            ),
            "coherence_median": self.coherence_median,
            "coarseness_center_mm": self.coarseness_center_mm,
            "coarseness_tolerance_ratio": self.coarseness_tolerance_ratio,
            "coarseness_range_mm": (
                None
                if self.coarseness_center_mm is None
                else (
                    self.coarseness_center_mm / self.coarseness_tolerance_ratio,
                    self.coarseness_center_mm * self.coarseness_tolerance_ratio,
                )
            ),
            "relative_amplitude": self.relative_amplitude,
            "amplitude_excluded_from_classification": True,
        }


@dataclass(frozen=True)
class WindowReference:
    capture: str
    window: str

    @classmethod
    def parse(cls, data, label):
        data = expect_object(data, label)
        exact_keys(data, ("capture", "window"), label)
        values = [data["capture"], data["window"]]
        if any(not isinstance(value, str) or not 0 < len(value) <= 64 for value in values):
            raise UserInputError(f"{label} must name one capture and one window.")
        return cls(*values)

    def as_dict(self):
        return {"capture": self.capture, "window": self.window}


@dataclass(frozen=True)
class GrainSelection:
    reference_a: WindowReference | None = None
    reference_b: WindowReference | None = None
    target: WindowReference | None = None

    @classmethod
    def parse(cls, data):
        data = expect_object(data, "Grain references")
        exact_keys(data, ROLES, "Grain references")
        values = {}
        for role in ROLES:
            value = data[role]
            values[role] = None if value is None else WindowReference.parse(value, f"{role} window")
        choice = cls(**values)
        configured = [item for item in values.values() if item is not None]
        signatures = [(item.capture, item.window) for item in configured]
        if len(set(signatures)) != len(signatures):
            raise UserInputError("Choose three different windows for the grain comparison.")
        if (values["reference_a"] is None) != (values["reference_b"] is None):
            raise UserInputError("Provide both signed-off reference windows or neither.")
        return choice

    @property
    def configured(self):
        return (
            self.reference_a is not None
            and self.reference_b is not None
            and self.target is not None
        )

    def as_dict(self):
        return {
            role: None if getattr(self, role) is None else getattr(self, role).as_dict()
            for role in ROLES
        }

    def reference_for(self, role):
        return getattr(self, role)

    def signature(self):
        return tuple(
            None if getattr(self, role) is None else (getattr(self, role).capture, getattr(self, role).window)
            for role in ROLES
        )


def grain_axes(rows, columns, row_pitch, column_pitch, shortest, longest):
    longitudinal = np.fft.fftfreq(rows, d=row_pitch)[:, None] * np.ones((1, columns))
    transverse = np.fft.fftfreq(columns, d=column_pitch)[None, :] * np.ones((rows, 1))
    magnitude = np.hypot(longitudinal, transverse)
    selected = (magnitude >= 1 / longest) & (magnitude <= 1 / shortest)
    orientation = (np.degrees(np.arctan2(longitudinal, transverse)) + 90) % 180 - 90
    wavelengths = np.divide(
        1.0, magnitude, out=np.full(magnitude.shape, np.nan), where=magnitude > 0
    )
    wave_edges = np.geomspace(shortest, longest, WAVE_BANDS + 1)
    band_index = np.clip(
        np.searchsorted(wave_edges, wavelengths, side="right") - 1, 0, WAVE_BANDS - 1
    )
    return GrainAxes(
        selected,
        np.exp(2j * np.radians(orientation))[selected],
        wavelengths[selected],
        band_index[selected],
        wave_edges,
    )


def patch_window(rows, columns):
    window = np.hanning(rows)[:, None] * np.hanning(columns)[None, :]
    return window, float(np.mean(window**2))


def tile_feature(patch, axes, window, window_square_mean):
    values = np.asarray(patch, dtype=np.float64)
    rows, columns = values.shape
    along_axis = np.arange(rows, dtype=np.float64) - (rows - 1) / 2
    across_axis = np.arange(columns, dtype=np.float64) - (columns - 1) / 2
    center = float(np.mean(values))
    along = np.dot(np.sum(values, axis=1), along_axis) / (
        np.dot(along_axis, along_axis) * columns
    )
    across = np.dot(np.sum(values, axis=0), across_axis) / (
        np.dot(across_axis, across_axis) * rows
    )
    plane = center + along * along_axis[:, None] + across * across_axis[None, :]
    tapered = window * (values - plane)
    spectrum = np.fft.fft2(tapered)
    power = (spectrum.real**2 + spectrum.imag**2)[axes.selected] / values.size**2
    total = float(np.sum(power))
    if total <= 0:
        return None
    vector = complex(np.sum(power * axes.unit_vector))
    coherence = float(abs(vector) / total)
    direction = float(np.degrees(np.angle(vector)) / 2)
    coarseness = float(np.exp(np.sum(power * np.log(axes.wavelength_mm)) / total))
    bands = np.bincount(axes.band_index, weights=power, minlength=WAVE_BANDS) / total
    rms = float(np.sqrt(total / window_square_mean))
    return GrainFeature(direction, coherence, coarseness, rms, tuple(map(float, bands)))


def feature_mask(context):
    """Material cells covered by a detected spot, streak, join or repeat mark.

    Across one of these the grain spectrum is dominated by the feature rather
    than the surface, so a tile touching one cannot support or contradict a
    signed-off grain reference.
    """
    from .raster import index_near

    grid = context.grid
    mask = np.zeros(grid.shape, dtype=bool)

    def box(first_mm, last_mm, left_mm, right_mm, guard=2):
        row_start = max(0, index_near(grid.stations, first_mm) - guard)
        row_stop = min(grid.shape[0], index_near(grid.stations, last_mm) + guard + 1)
        column_start = max(0, index_near(grid.transverse, left_mm) - guard)
        column_stop = min(grid.shape[1], index_near(grid.transverse, right_mm) + guard + 1)
        mask[row_start:row_stop, column_start:column_stop] = True

    for spot in context.spots.spots:
        box(spot.from_mm, spot.to_mm, spot.left_mm, spot.right_mm)
    for streak in context.streaks.streaks:
        box(streak.from_mm, streak.to_mm, streak.left_mm, streak.right_mm)
    for splice in context.splices.splices:
        box(splice.before_mm, splice.after_mm, splice.left_mm, splice.right_mm)
    marks = context.settings.marks
    for mark in context.marks.marks:
        width = context.edges.width_mm[mark.row]
        left = context.edges.left_mm[mark.row]
        if np.isfinite(left) and np.isfinite(width):
            box(
                mark.station_mm,
                mark.station_mm,
                left + marks.band_start_fraction * width,
                left + marks.band_end_fraction * width,
            )
    return mask


def tile_state(valid, material, features, overlap):
    if not np.all(valid):
        return TileState.UNSUPPORTED
    if not np.all(material):
        return TileState.OUTSIDE
    if features is not None and np.mean(features) > overlap:
        return TileState.FEATURE
    return TileState.MEASURED


def window_tiles(context, settings, region, window_id, label, capture_ref=""):
    grid = context.grid
    rows, columns = region.indices(grid.stations, grid.transverse)
    row_count = rows.stop - rows.start
    column_count = columns.stop - columns.start
    tile_rows = max(6, int(round(settings.tile_size_mm / grid.row_pitch_mm)))
    tile_columns = max(6, int(round(settings.tile_size_mm / grid.column_pitch_mm)))
    row_stride = max(1, int(round(tile_rows * (1 - settings.overlap_fraction))))
    column_stride = max(1, int(round(tile_columns * (1 - settings.overlap_fraction))))
    row_starts = start_cells(row_count, tile_rows, row_stride)
    column_starts = start_cells(column_count, tile_columns, column_stride)
    starts = np.array(
        [(row, column) for row in row_starts for column in column_starts], dtype=np.int64
    ).reshape(-1, 2)
    total_positions = len(starts)
    if total_positions > MAX_WINDOW_TILES:
        starts = starts[balanced_sampling(np.arange(total_positions), MAX_WINDOW_TILES)]
    axes = grain_axes(
        tile_rows,
        tile_columns,
        grid.row_pitch_mm,
        grid.column_pitch_mm,
        settings.minimum_wave_mm,
        settings.maximum_wave_mm,
    )
    window, window_square_mean = patch_window(tile_rows, tile_columns)
    features = feature_mask(context)
    tiles = []
    for identifier, (local_row, local_column) in enumerate(starts, 1):
        row_start = int(rows.start + local_row)
        column_start = int(columns.start + local_column)
        row_stop = row_start + tile_rows
        column_stop = column_start + tile_columns
        valid = grid.valid[row_start:row_stop, column_start:column_stop]
        material = context.edges.material[row_start:row_stop, column_start:column_stop]
        patch = grid.values[row_start:row_stop, column_start:column_stop]
        state = tile_state(
            valid,
            material,
            features[row_start:row_stop, column_start:column_stop],
            settings.maximum_feature_share,
        )
        feature = None
        if state == TileState.MEASURED:
            feature = tile_feature(patch, axes, window, window_square_mean)
            if feature is None or feature.coherence < settings.minimum_coherence:
                state = TileState.FAINT
        if state != TileState.MEASURED:
            feature = None
        measured = state == TileState.MEASURED
        reflectance = float(np.median(patch)) if np.all(np.isfinite(patch)) else None
        relative = None
        if measured and reflectance is not None and reflectance > 1e-6:
            relative = float(feature.rms / reflectance)
        station_from = float(grid.stations[row_start])
        station_to = float(grid.stations[row_stop - 1])
        transverse_from = float(grid.transverse[column_start])
        transverse_to = float(grid.transverse[column_stop - 1])
        tiles.append(
            GrainTile(
                identifier,
                row_start,
                row_stop,
                column_start,
                column_stop,
                station_from,
                station_to,
                transverse_from,
                transverse_to,
                0.5 * (station_from + station_to),
                0.5 * (transverse_from + transverse_to),
                station_to - station_from,
                transverse_to - transverse_from,
                reflectance,
                state,
                feature.direction_deg if measured else None,
                feature.coherence if measured else None,
                feature.coarseness_mm if measured else None,
                relative,
                feature.bands if measured else tuple([0.0] * WAVE_BANDS),
            )
        )
    region_dict = region.as_dict() | {
        "tile_height_mm": tile_rows * grid.row_pitch_mm,
        "tile_width_mm": tile_columns * grid.column_pitch_mm,
    }
    return GrainWindow(
        window_id,
        capture_ref,
        context.capture.metadata.name,
        label,
        region_dict,
        tuple(tiles),
        tile_rows,
        tile_columns,
        row_stride,
        column_stride,
        total_positions,
        axes.wave_edges,
    )


def circular_absolute_difference(first, second):
    wrapped = (first - second + ENCIRCLED_OFFSET) % (2 * ENCIRCLED_OFFSET) - ENCIRCLED_OFFSET
    return abs(float(wrapped))


def weighted_quantile(values, weights, fraction):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    target = fraction * float(cumulative[-1])
    return float(values[order][np.searchsorted(cumulative, target, side="left")])


def weighted_median(values, weights):
    return weighted_quantile(values, weights, 0.5)


def trimmed_sigma(values, weights, settings):
    """Trim the far tail of a reference, then report its robust width.

    The trimmed window is the bulk of the signed-off samples. The width is a
    weighted median absolute deviation scaled to a Gaussian sigma, so a few
    structural tiles inside a signed-off span do not widen the envelope.
    """
    center = weighted_median(values, weights)
    deviation = np.abs(values - center)
    sigma = 1.4826 * weighted_median(deviation, weights)
    if sigma > 0:
        retain = deviation <= settings.outlier_sigmas * sigma
    else:
        retain = np.zeros(len(values), dtype=bool)
    retained = int(np.count_nonzero(retain))
    if retained < max(3, len(values) // 2):
        retain = np.ones(len(values), dtype=bool)
        retained = len(values)
        sigma = 1.4826 * weighted_median(deviation, weights)
    return float(center), float(sigma), retained


def build_envelope(window, key, settings):
    measured = window.measured_tiles
    amplitude = scale_summary(
        [tile.relative_amplitude for tile in measured if tile.relative_amplitude is not None]
    )
    if len(measured) < settings.minimum_reference_tiles:
        return GrainEnvelope(
            key, window.window_id, window.capture_ref, window.label, "insufficient",
            len(measured), 0, 0, None, None, None, None, None, amplitude,
        )
    weights = np.array([tile.coherence for tile in measured], dtype=np.float64)
    directions = np.array([tile.direction_deg for tile in measured], dtype=np.float64)
    angles = np.radians(2 * directions)
    initial = float(np.degrees(np.angle(np.sum(weights * np.exp(1j * angles)))) / 2)
    wrapped = (directions - initial + ENCIRCLED_OFFSET) % (2 * ENCIRCLED_OFFSET) - ENCIRCLED_OFFSET
    center, angle_sigma, retained = trimmed_sigma(wrapped, weights, settings)
    center = (initial + center + ENCIRCLED_OFFSET) % (2 * ENCIRCLED_OFFSET) - ENCIRCLED_OFFSET
    tolerance = float(
        min(45.0, max(settings.angle_tolerance_deg, settings.envelope_margin * angle_sigma))
    )
    logarithmic = np.log(np.array([tile.coarseness_mm for tile in measured], dtype=np.float64))
    center_log, log_sigma, _ = trimmed_sigma(logarithmic, weights, settings)
    ratio = float(
        min(4.0, max(settings.coarseness_ratio, float(np.exp(settings.envelope_margin * log_sigma))))
    )
    return GrainEnvelope(
        key,
        window.window_id,
        window.capture_ref,
        window.label,
        "usable",
        len(measured),
        retained,
        len(measured) - retained,
        float(center),
        tolerance,
        float(np.median(weights)),
        float(np.exp(center_log)),
        ratio,
        amplitude,
    )


def membership(envelope, tile):
    """Return (limit deviation, limiting feature) or None if not judgeable."""
    if not envelope.usable or not tile.measured:
        return None
    angle = circular_absolute_difference(tile.direction_deg, envelope.direction_center_deg)
    angle_deviation = angle / envelope.direction_tolerance_deg
    scale_deviation = abs(np.log(tile.coarseness_mm / envelope.coarseness_center_mm)) / np.log(
        envelope.coarseness_tolerance_ratio
    )
    if angle_deviation >= scale_deviation:
        return float(angle_deviation), "direction"
    return float(scale_deviation), "coarseness"


def classify_tile(tile, envelope_a, envelope_b):
    if not tile.measured:
        return TileClass.UNKNOWN, tile.state.value, None, None
    left = membership(envelope_a, tile)
    right = membership(envelope_b, tile)
    if left is None and right is None:
        return TileClass.UNKNOWN, "reference_unusable", None, None
    options = [entry for entry in (left, right) if entry is not None]
    nearest = min(options, key=lambda entry: entry[0])
    matched_left = left is not None and left[0] <= 1
    matched_right = right is not None and right[0] <= 1
    if matched_left and matched_right:
        category = TileClass.BOTH
    elif matched_left:
        category = TileClass.ONLY_A
    elif matched_right:
        category = TileClass.ONLY_B
    else:
        category = TileClass.NEITHER
    return category, None, nearest[0], nearest[1]


def comparison_row(tile, envelope_a, envelope_b):
    category, reason, deviation, driver = classify_tile(tile, envelope_a, envelope_b)
    left = membership(envelope_a, tile)
    right = membership(envelope_b, tile)
    return {
        "tile": tile.id,
        "station_mm": tile.station_mm,
        "transverse_mm": tile.transverse_mm,
        "direction_deg": tile.direction_deg,
        "coherence": tile.coherence,
        "coarseness_mm": tile.coarseness_mm,
        "relative_amplitude": tile.relative_amplitude,
        "reflectance": tile.reflectance,
        "classification": category.value,
        "nearest_reference": deviation,
        "limiting_feature": driver,
        "reference_a_supports": None if left is None else bool(left[0] <= 1),
        "reference_b_supports": None if right is None else bool(right[0] <= 1),
        "unknown_reason": reason,
    }


def class_counts(rows):
    counts = {category.value: 0 for category in TileClass}
    for row in rows:
        counts[row["classification"]] += 1
    return counts


def class_extents(rows):
    """Station and transverse extents occupied by each class in the target window."""
    extents = {}
    for category in TileClass:
        selected = [row for row in rows if row["classification"] == category.value]
        if not selected:
            extents[category.value] = None
            continue
        extents[category.value] = {
            "tiles": len(selected),
            "station_from_mm": min(row["station_mm"] for row in selected),
            "station_to_mm": max(row["station_mm"] for row in selected),
            "transverse_from_mm": min(row["transverse_mm"] for row in selected),
            "transverse_to_mm": max(row["transverse_mm"] for row in selected),
        }
    return extents


def support_share(window, envelope):
    measured = window.measured_tiles
    if not measured or not envelope.usable:
        return None
    inside = sum(1 for tile in measured if (membership(envelope, tile) or (2.0, None))[0] <= 1)
    return inside / len(measured)


@dataclass(frozen=True)
class GrainSource:
    role: str
    capture_ref: str
    context: object
    window: object

    def region(self):
        return self.window.region

    def slices(self):
        return self.window.region.indices(self.context.grid.stations, self.context.grid.transverse)

    def tiles(self, settings):
        return window_tiles(
            self.context,
            settings,
            self.window.region,
            self.window.id,
            self.window.label,
            self.capture_ref,
        )


@dataclass(frozen=True)
class GrainComparison:
    settings: object
    selection: GrainSelection
    target: GrainWindow
    reference_a: GrainWindow
    reference_b: GrainWindow
    envelope_a: GrainEnvelope
    envelope_b: GrainEnvelope
    rows: tuple
    reference_a_rows: tuple
    reference_b_rows: tuple
    contexts: dict

    def windows(self):
        return (
            ("target", self.target),
            ("reference_a", self.reference_a),
            ("reference_b", self.reference_b),
        )

    def window(self, key):
        for name, window in self.windows():
            if name == key:
                return window
        raise KeyError(key)

    def summary(self):
        counts = class_counts(self.rows)
        measured = sum(
            counts[category.value]
            for category in (
                TileClass.BOTH,
                TileClass.ONLY_A,
                TileClass.ONLY_B,
                TileClass.NEITHER,
            )
        )
        measured_rows = [
            row for row in self.rows if row["classification"] != TileClass.UNKNOWN.value
        ]
        outside = (
            counts[TileClass.ONLY_A.value]
            + counts[TileClass.ONLY_B.value]
            + counts[TileClass.NEITHER.value]
        )
        return {
            "target_window": self.target.label,
            "reference_a": self.reference_a.label,
            "reference_b": self.reference_b.label,
            "tile_height_mm": self.target.region["tile_height_mm"],
            "tile_width_mm": self.target.region["tile_width_mm"],
            "target_tiles": len(self.rows),
            "measured_tiles": measured,
            "both_references_tiles": counts[TileClass.BOTH.value],
            "only_reference_a_tiles": counts[TileClass.ONLY_A.value],
            "only_reference_b_tiles": counts[TileClass.ONLY_B.value],
            "neither_reference_tiles": counts[TileClass.NEITHER.value],
            "unknown_tiles": counts[TileClass.UNKNOWN.value],
            "unknown_feature_tiles": self.target.counts()[TileState.FEATURE.value],
            "unknown_unsupported_tiles": self.target.counts()[TileState.UNSUPPORTED.value],
            "unknown_outside_tiles": self.target.counts()[TileState.OUTSIDE.value],
            "unknown_faint_tiles": self.target.counts()[TileState.FAINT.value],
            "both_references_fraction": counts[TileClass.BOTH.value] / measured if measured else None,
            "outside_normal_fraction": outside / measured if measured else None,
            "reference_a_status": self.envelope_a.status,
            "reference_b_status": self.envelope_b.status,
            "reference_a_measured_tiles": self.envelope_a.measured_tiles,
            "reference_b_measured_tiles": self.envelope_b.measured_tiles,
            "reference_a_retained_tiles": self.envelope_a.retained_tiles,
            "reference_b_retained_tiles": self.envelope_b.retained_tiles,
            "reference_a_tiles_supported_by_b": support_share(self.reference_a, self.envelope_b),
            "reference_b_tiles_supported_by_a": support_share(self.reference_b, self.envelope_a),
            "target_direction_median_deg": scale_summary(
                [row["direction_deg"] for row in measured_rows]
            )["median"],
            "target_coarseness_median_mm": scale_summary(
                [row["coarseness_mm"] for row in measured_rows]
            )["median"],
        }

    def matrix(self):
        return {
            "target": class_counts(self.rows),
            "reference_a": class_counts(self.reference_a_rows),
            "reference_b": class_counts(self.reference_b_rows),
        }

    def extents(self):
        return class_extents(self.rows)


def compare_grain(settings, sources):
    """sources: sequence of GrainSource with roles reference_a, reference_b, target."""
    by_role = {source.role: source for source in sources}
    if set(by_role) != set(ROLES):
        raise UserInputError("A grain comparison needs two references and one target.")
    window_a = by_role["reference_a"].tiles(settings)
    window_b = by_role["reference_b"].tiles(settings)
    window_target = by_role["target"].tiles(settings)
    envelope_a = build_envelope(window_a, "a", settings)
    envelope_b = build_envelope(window_b, "b", settings)
    selection = GrainSelection(
        WindowReference(by_role["reference_a"].capture_ref, window_a.window_id),
        WindowReference(by_role["reference_b"].capture_ref, window_b.window_id),
        WindowReference(by_role["target"].capture_ref, window_target.window_id),
    )
    return GrainComparison(
        settings,
        selection,
        window_target,
        window_a,
        window_b,
        envelope_a,
        envelope_b,
        tuple(comparison_row(tile, envelope_a, envelope_b) for tile in window_target.tiles),
        tuple(comparison_row(tile, envelope_a, envelope_b) for tile in window_a.tiles),
        tuple(comparison_row(tile, envelope_a, envelope_b) for tile in window_b.tiles),
        {source.role: source.context for source in sources},
    )


def tile_detail(comparison, key, tile_id):
    window = comparison.window(key)
    try:
        identifier = int(tile_id)
        tile = window.tiles[identifier - 1]
    except (IndexError, TypeError, ValueError):
        raise UserInputError("That grain tile does not exist in the selected window.")
    if tile.id != identifier:
        raise UserInputError("That grain tile does not exist in the selected window.")
    category, reason, deviation, driver = classify_tile(
        tile, comparison.envelope_a, comparison.envelope_b
    )
    return {
        "window": key,
        "window_id": window.window_id,
        "capture_ref": window.capture_ref,
        "tile": tile.evidence(),
        "classification": category.value,
        "classification_title": CLASS_TITLES[category.value],
        "unknown_reason": reason,
        "nearest_reference": deviation,
        "limiting_feature": driver,
        "bands": tile.band_rows(window.wave_edges),
        "source_lines": source_profile(comparison.contexts[key], window, tile),
        "envelopes": {
            "reference_a": comparison.envelope_a.as_dict(),
            "reference_b": comparison.envelope_b.as_dict(),
        },
    }


def presentation(comparison):
    envelopes = [comparison.envelope_a.as_dict(), comparison.envelope_b.as_dict()]
    return {
        "section": "grain",
        "name": "Grain conformance",
        "summary": comparison.summary(),
        "tables": [
            {
                "key": "grain_target_tiles",
                "title": "Target tiles / measured grain",
                "rows": list(comparison.rows),
                "scope": "target",
            },
            {
                "key": "grain_reference_a_tiles",
                "title": f"Reference A tiles / {comparison.reference_a.label}",
                "rows": list(comparison.reference_a_rows),
                "scope": "reference_a",
            },
            {
                "key": "grain_reference_b_tiles",
                "title": f"Reference B tiles / {comparison.reference_b.label}",
                "rows": list(comparison.reference_b_rows),
                "scope": "reference_b",
            },
            {
                "key": "grain_envelopes",
                "title": "Signed-off grain envelopes",
                "rows": envelopes,
                "scope": "envelopes",
            },
        ],
        "charts": [],
        "references": envelopes,
        "matrix": comparison.matrix(),
        "extents": comparison.extents(),
        "classes": [
            {"key": category.value, "title": CLASS_TITLES[category.value]}
            for category in TileClass
        ],
        "selected_windows": [
            comparison.window(role).summary()
            for role in ("target", "reference_a", "reference_b")
        ],
        "selection": comparison.selection.as_dict(),
    }


def source_profile(context, window, tile, limit=160):
    """Raw scan lines that produced one tile, with their original sensor codes."""
    from .window_review import source_line_set

    grid = context.grid
    capture = context.capture
    lines = source_line_set(grid, slice(tile.row_start, tile.row_stop))
    if len(lines) > limit:
        chosen = np.linspace(0, len(lines) - 1, limit, dtype=np.int64)
        lines = [lines[int(index)] for index in chosen]
    columns = slice(tile.column_start, tile.column_stop)
    return [
        {
            "source_line": int(line),
            "tick": int(capture.ticks[line]),
            "station_mm": float(capture.stations[line]),
            "capture_time_s": float(capture.seconds[line]),
            "mean_sensor_code": float(np.mean(capture.raw[line, columns])),
            "mean_reflectance": safe_value(np.mean(context.calibrated.values[line, columns])),
            "tile": tile.id,
            "window": window.label,
        }
        for line in lines
    ]
