import math
from dataclasses import asdict, dataclass, fields
from enum import Enum

import numpy as np


MAX_ROWS = 4096
MAX_COLS = 1024
MAX_GRID_PIXELS = 3_000_000
MAX_REFERENCE_ROWS = 64
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 1400


class UserInputError(ValueError):
    pass


class MissingRecord(LookupError):
    pass


class RevisionConflict(ValueError):
    def __init__(self, revision):
        super().__init__("This capture changed in another view. Reload it before saving.")
        self.revision = revision


class Classification(str, Enum):
    OPEN = "open"
    WATCH = "watch"
    RESOLVED = "resolved"


def finite_number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UserInputError(f"{label} must be a number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise UserInputError(f"{label} is outside the supported range.") from error
    if not math.isfinite(result):
        raise UserInputError(f"{label} must be finite.")
    if minimum is not None and result < minimum:
        raise UserInputError(f"{label} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise UserInputError(f"{label} must be at most {maximum}.")
    return result


def whole_number(value, label, minimum, maximum):
    if type(value) is not int:
        raise UserInputError(f"{label} must be a whole number.")
    if not minimum <= value <= maximum:
        raise UserInputError(f"{label} must be between {minimum} and {maximum}.")
    return value


def clean_text(value, label, maximum, allow_blank=False):
    if not isinstance(value, str):
        raise UserInputError(f"{label} must be text.")
    result = value.strip()
    if not result and not allow_blank:
        raise UserInputError(f"{label} is required.")
    if len(result) > maximum:
        raise UserInputError(f"{label} must be at most {maximum} characters.")
    if any(ord(character) < 32 and character not in "\n\t" for character in result):
        raise UserInputError(f"{label} contains an unsupported control character.")
    return result


def expect_object(value, label):
    if not isinstance(value, dict):
        raise UserInputError(f"{label} must be an object.")
    return value


def read_enum(enum_type, value, label):
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        choices = ", ".join(item.value for item in enum_type)
        raise UserInputError(f"{label} must be one of {choices}.") from None


def reject_unknown(data, allowed, label):
    unknown = set(data) - set(allowed)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise UserInputError(f"{label} contains unknown fields: {names}.")


def exact_keys(data, expected, label):
    expected = set(expected)
    reject_unknown(data, expected, label)
    missing = expected - set(data)
    if missing:
        names = ", ".join(sorted(missing))
        raise UserInputError(f"{label} is missing required fields: {names}.")


@dataclass(frozen=True)
class Geometry:
    tick_mm: float
    column_mm: float
    station_origin_mm: float
    transverse_origin_mm: float

    @classmethod
    def parse(cls, data):
        data = expect_object(data, "Capture geometry")
        exact_keys(data, (field.name for field in fields(cls)), "Capture geometry")
        return cls(
            tick_mm=finite_number(data["tick_mm"], "Encoder pitch", 0.00001, 250),
            column_mm=finite_number(data["column_mm"], "Column pitch", 0.0001, 250),
            station_origin_mm=finite_number(
                data["station_origin_mm"], "Station origin", -1e7, 1e7
            ),
            transverse_origin_mm=finite_number(
                data["transverse_origin_mm"], "Transverse origin", -1e6, 1e6
            ),
        )

    def station(self, ticks):
        return self.station_origin_mm + np.asarray(ticks, dtype=np.float64) * self.tick_mm

    def transverse(self, columns):
        return (
            self.transverse_origin_mm
            + np.asarray(columns, dtype=np.float64) * self.column_mm
        )

    def column(self, transverse):
        return (transverse - self.transverse_origin_mm) / self.column_mm

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CaptureMetadata:
    name: str
    material: str
    description: str
    synthetic: bool
    geometry: Geometry

    @classmethod
    def parse(cls, data):
        data = expect_object(data, "Capture metadata")
        exact_keys(data, [field.name for field in fields(cls)], "Capture metadata")
        synthetic = data["synthetic"]
        if type(synthetic) is not bool:
            raise UserInputError("The synthetic flag must be true or false.")
        return cls(
            name=clean_text(data["name"], "Capture name", 90),
            material=clean_text(data["material"], "Material", 80),
            description=clean_text(data["description"], "Capture description", 600, True),
            synthetic=synthetic,
            geometry=Geometry.parse(data["geometry"]),
        )

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Capture:
    raw: np.ndarray
    ticks: np.ndarray
    seconds: np.ndarray
    dark: np.ndarray
    white: np.ndarray
    metadata: CaptureMetadata

    @classmethod
    def build(cls, raw, ticks, seconds, dark, white, metadata):
        metadata = CaptureMetadata.parse(metadata.as_dict())
        raw = counter_image(raw, "Scan image", 8, MAX_ROWS)
        ticks = encoder_vector(ticks, raw.shape[0])
        seconds = clock_vector(seconds, raw.shape[0])
        dark = reference_image(dark, raw.shape[1], "Dark reference")
        white = reference_image(white, raw.shape[1], "White reference")
        s_min, s_max = metadata.geometry.station(ticks[[0, -1]])
        if (s_max - s_min) > 100_000:
            raise UserInputError("A capture may span at most 100,000 mm.")
        if max(abs(s_min), abs(s_max)) > 1e8:
            raise UserInputError("Station coordinates must be within ±100,000,000 mm.")
        return cls(raw, ticks, seconds, dark, white, metadata)

    @property
    def rows(self):
        return self.raw.shape[0]

    @property
    def columns(self):
        return self.raw.shape[1]

    @property
    def stations(self):
        return self.metadata.geometry.station(self.ticks)

    @property
    def transverse(self):
        return self.metadata.geometry.transverse(np.arange(self.columns))

    @property
    def positive_pitch(self):
        differences = np.diff(self.ticks)
        return float(np.median(differences[differences > 0]) * self.metadata.geometry.tick_mm)

    def overview(self):
        station_min, station_max = self.metadata.geometry.station(self.ticks[[0, -1]])
        x_min, x_max = self.metadata.geometry.transverse([0, self.columns - 1])
        return {
            "rows": self.rows,
            "columns": self.columns,
            "station_start_mm": float(station_min),
            "station_end_mm": float(station_max),
            "transverse_start_mm": float(x_min),
            "transverse_end_mm": float(x_max),
            "duration_s": float(self.seconds[-1] - self.seconds[0]),
            "nominal_pitch_mm": self.positive_pitch,
            "dark_passes": len(self.dark),
            "white_passes": len(self.white),
        }


def counter_image(array, label, min_rows, max_rows):
    array = np.asarray(array)
    if array.ndim != 2:
        raise UserInputError(f"{label} must be a two-dimensional array.")
    height, width = array.shape
    if not min_rows <= height <= max_rows or not 16 <= width <= MAX_COLS:
        raise UserInputError(
            f"{label} must be {min_rows}–{max_rows} lines by 16–{MAX_COLS} columns."
        )
    if array.dtype.kind not in "ui":
        raise UserInputError(f"{label} must contain whole-number sensor codes.")
    if np.min(array) < 0 or np.max(array) > 65535:
        raise UserInputError(f"{label} sensor codes must be between 0 and 65535.")
    return immutable_array(array, "<u2")


def reference_image(array, width, label):
    image = counter_image(array, label, 1, MAX_REFERENCE_ROWS)
    if image.shape[1] != width:
        raise UserInputError(f"{label} must have one column per scan column.")
    return image


def encoder_vector(array, rows):
    array = np.asarray(array)
    if array.shape != (rows,) or array.dtype.kind not in "iu":
        raise UserInputError("Encoder ticks must provide a whole-number value for each line.")
    if np.min(array) < -1_000_000_000_000 or np.max(array) > 1_000_000_000_000:
        raise UserInputError("Encoder ticks must be within ±1,000,000,000,000.")
    values = immutable_array(array, "<i8")
    differences = np.diff(values)
    if np.any(differences < 0):
        raise UserInputError("Encoder ticks must not decrease within one capture.")
    if not np.any(differences > 0):
        raise UserInputError("The capture must contain at least two encoder positions.")
    return values


def clock_vector(array, rows):
    array = np.asarray(array)
    if array.shape != (rows,) or array.dtype.kind not in "iuf":
        raise UserInputError("Capture times must provide one numeric value for each line.")
    values = immutable_array(array, "<f8")
    if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1e12:
        raise UserInputError("Capture times must be finite and within ±10¹² seconds.")
    if np.any(np.diff(values) < 1e-9):
        raise UserInputError("Capture times must increase by at least one nanosecond in scan order.")
    return values


def immutable_array(value, dtype):
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class StationWindow:
    station_start_mm: float
    station_end_mm: float
    transverse_start_mm: float
    transverse_end_mm: float

    @classmethod
    def parse(cls, data):
        data = expect_object(data, "Inspection window")
        exact_keys(data, (field.name for field in fields(cls)), "Inspection window")
        window = cls(
            station_start_mm=finite_number(data["station_start_mm"], "Start station"),
            station_end_mm=finite_number(data["station_end_mm"], "End station"),
            transverse_start_mm=finite_number(data["transverse_start_mm"], "Left edge"),
            transverse_end_mm=finite_number(data["transverse_end_mm"], "Right edge"),
        )
        if window.station_start_mm >= window.station_end_mm:
            raise UserInputError("The window must have positive length along the roll.")
        if window.transverse_start_mm >= window.transverse_end_mm:
            raise UserInputError("The window must have positive width across the roll.")
        return window

    @classmethod
    def full(cls, capture):
        summary = capture.overview()
        return cls(
            summary["station_start_mm"],
            summary["station_end_mm"],
            summary["transverse_start_mm"],
            summary["transverse_end_mm"],
        )

    def within(self, capture):
        full = self.full(capture)
        checks = (
            self.station_start_mm < full.station_start_mm,
            self.station_end_mm > full.station_end_mm,
            self.transverse_start_mm < full.transverse_start_mm,
            self.transverse_end_mm > full.transverse_end_mm,
        )
        if any(checks):
            raise UserInputError("The window must lie inside the captured station and column range.")
        return self

    def indices(self, stations, transverse):
        row_start = int(np.searchsorted(stations, self.station_start_mm, side="left"))
        row_stop = int(np.searchsorted(stations, self.station_end_mm, side="right"))
        column_start = int(
            np.searchsorted(transverse, self.transverse_start_mm, side="left")
        )
        column_stop = int(
            np.searchsorted(transverse, self.transverse_end_mm, side="right")
        )
        if row_start >= row_stop or column_start >= column_stop:
            raise UserInputError("The selected window contains no sampled cell centers.")
        return slice(row_start, row_stop), slice(column_start, column_stop)

    def as_dict(self):
        return asdict(self)
