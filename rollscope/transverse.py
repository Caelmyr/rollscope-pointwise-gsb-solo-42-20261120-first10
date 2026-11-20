"""Transverse calibration from a photographed cross-web ruler.

Line-scan optics rarely give a constant millimetre-per-column ratio across the
whole detector. This module turns a picture of a ruler into a column-to-millimetre
mapping:

* :func:`detect_ruler_lines` finds the dark engraved lines in the raw codes, fits
  each slightly tilted line in sub-column coordinates and grades the evidence so
  that a faint, tilted or stained line can be recognised instead of trusted blindly.
* :func:`build_transverse_calibration` fits the accepted lines robustly. A single
  locally deformed line is rejected before it can pull neighbouring widths with it,
  and the final mapping is a monotone piecewise-linear curve through the surviving
  anchors. Columns beyond the outermost reliable line are explicitly uncalibrated.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from scipy import ndimage

from .calibration import safe_value
from .domain import UserInputError, exact_keys, expect_object, finite_number, whole_number

MAX_LINES = 128
MIN_LINES = 3
OUTLIER_SIGMA = 2.0


def identifier():
    return uuid4().hex[:16]


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RulerLine:
    """One engraved ruler line observed in the calibration picture."""

    line_id: int
    column: float
    slope: float
    intercept: float
    row_start: int
    row_end: int
    prominence: float
    contrast: float
    straightness: float
    coverage: float
    stained: bool
    stain_fraction: float
    score: float
    rejected_reason: str | None

    @property
    def reliable(self):
        return self.rejected_reason is None

    def as_dict(self):
        data = asdict(self)
        data["stained"] = bool(self.stained)
        return data


def _column_profile(raw):
    median = np.median(raw, axis=0).astype(np.float64)
    spread = ndimage.gaussian_filter1d(median, sigma=3, mode="nearest")
    deviation = spread - median
    noise = np.median(np.abs(deviation - np.median(deviation))) * 1.4826
    noise = max(noise, 1.0)
    return median, deviation, noise


def _line_components(median, deviation, noise, minimum_prominence):
    threshold = max(minimum_prominence * noise, 0.02 * float(np.ptp(median)))
    candidate = deviation >= threshold
    structure = np.ones(3, dtype=bool)
    labels, count = ndimage.label(candidate, structure=structure)
    components = []
    for label in range(1, count + 1):
        indices = np.flatnonzero(labels == label)
        if indices.size < 2:
            continue
        weights = np.clip(deviation[indices], 0, None)
        center = float(np.average(indices, weights=weights))
        prominence = float(np.average(deviation[indices], weights=weights) / noise)
        components.append((indices, center, prominence))
    components.sort(key=lambda item: item[1])
    return components, threshold / noise


def _fit_line(raw, indices, center, median, search_radius):
    rows = np.arange(raw.shape[0])
    left = max(0, int(round(center)) - search_radius)
    right = min(raw.shape[1], int(round(center)) + search_radius + 1)
    window = raw[:, left:right].astype(np.float64)
    local_median = median[left:right]
    dark = np.clip(local_median[None, :] - window, 0, None)
    column_weights = dark.sum(axis=0)
    if column_weights.sum() <= 0:
        column_weights = np.ones(right - left)
    offsets = np.arange(left, right, dtype=np.float64)
    row_weight = dark.sum(axis=1)
    usable = row_weight > 0
    if usable.sum() < 2:
        usable = np.ones(raw.shape[0], dtype=bool)
    centers = (dark * offsets[None, :]).sum(axis=1) / np.maximum(dark.sum(axis=1), 1e-9)
    valid_rows = rows[usable]
    valid_centers = centers[usable]
    slope, intercept = np.polyfit(valid_rows, valid_centers, 1)
    fitted = slope * rows + intercept
    residual = centers[usable] - fitted[usable]
    straightness = float(np.sqrt(np.average(residual**2, weights=row_weight[usable])))
    coverage = float(np.count_nonzero(usable) / raw.shape[0])
    return slope, intercept, straightness, coverage, dark, usable


def _stain_evidence(raw, indices, slope, intercept, median, contrast, search_radius):
    """A stain drags the per-row line centre sideways over a contiguous band of rows."""

    rows = np.arange(raw.shape[0])
    center = float(np.average(slope * rows + intercept))
    left = max(0, int(round(center)) - search_radius)
    right = min(raw.shape[1], int(round(center)) + search_radius + 1)
    window = raw[:, left:right].astype(np.float64)
    offsets = np.arange(left, right, dtype=np.float64)
    dark = np.clip(median[left:right][None, :] - window, 0, None)
    centers = (dark * offsets[None, :]).sum(axis=1) / np.maximum(dark.sum(axis=1), 1e-9)
    design = np.column_stack((rows, np.ones_like(rows)))
    fit_slope, fit_intercept = np.linalg.lstsq(design, centers, rcond=None)[0]
    residual = centers - (fit_slope * rows + fit_intercept)
    robust_center = np.median(residual)
    robust_scale = np.median(np.abs(residual - robust_center)) * 1.4826
    outliers = np.abs(residual - robust_center) > 3.0 * max(robust_scale, 0.05)
    outliers = ndimage.binary_closing(outliers, structure=np.ones(3, dtype=bool))
    fraction = float(np.count_nonzero(outliers) / len(outliers))
    if not outliers.any():
        return False, fraction
    labels, count = ndimage.label(outliers, structure=np.ones(3, dtype=bool))
    largest = max(
        (np.count_nonzero(labels == label) for label in range(1, count + 1)), default=0
    )
    stained = (
        largest >= max(4, int(round(0.05 * raw.shape[0]))) or fraction > 0.10
    )
    return stained, fraction


def detect_ruler_lines(raw, minimum_prominence=3.0):
    """Detect dark ruler lines and grade the supporting evidence per line."""

    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise UserInputError("The ruler picture must be a two-dimensional image.")
    rows, columns = raw.shape
    median, deviation, noise = _column_profile(raw)
    components, threshold = _line_components(median, deviation, noise, minimum_prominence)
    search_radius = max(4, columns // 80)
    findings = []
    for order, (indices, center, prominence) in enumerate(components, 1):
        slope, intercept, straightness, coverage, dark, usable = _fit_line(
            raw, indices, center, median, search_radius
        )
        contrast = float(np.average(deviation[indices]) / max(np.ptp(median), 1.0))
        stained, stain_fraction = _stain_evidence(
            raw, indices, slope, intercept, median, contrast, search_radius
        )
        reason = None
        if prominence < minimum_prominence:
            reason = "faint_line"
        elif coverage < 0.55:
            reason = "partial_line"
        elif straightness > 2.5:
            reason = "bent_line"
        score = (
            min(prominence / 12.0, 1.0) * 0.45
            + min(coverage, 1.0) * 0.25
            + max(0.0, 1.0 - straightness / 2.5) * 0.18
            + (0.0 if stained else 0.12)
        )
        findings.append(
            RulerLine(
                line_id=order,
                column=float(np.average(slope * np.arange(rows) + intercept)),
                slope=float(slope),
                intercept=float(intercept),
                row_start=int(np.flatnonzero(usable).min()),
                row_end=int(np.flatnonzero(usable).max()),
                prominence=float(prominence),
                contrast=contrast,
                straightness=straightness,
                coverage=coverage,
                stained=stained,
                stain_fraction=stain_fraction,
                score=float(round(score, 5)),
                rejected_reason=reason,
            )
        )
    return tuple(findings), float(threshold)


def _weighted_quadratic(columns, positions, weights):
    """Weighted quadratic fit returning residuals in millimetres.

    With only three anchors the quadratic term is not identifiable, so a robust
    weighted line is used instead.
    """

    degree = 2 if len(columns) >= 4 else 1
    terms = [columns**power for power in range(degree, -1, -1)]
    design = np.column_stack(terms)
    weighted = design * np.sqrt(weights)[:, None]
    coefficients, *_ = np.linalg.lstsq(weighted, positions * np.sqrt(weights), rcond=None)
    fitted = design @ coefficients
    return positions - fitted, fitted


def build_transverse_calibration(raw, lines, known_positions_mm, accepted_ids, source):
    """Fit a monotone column-to-millimetre calibration from accepted ruler lines."""

    raw = np.asarray(raw)
    rows, columns = raw.shape
    lines = tuple(lines)
    if len(lines) > MAX_LINES:
        raise UserInputError(f"A ruler picture can contain at most {MAX_LINES} lines.")
    positions = _parse_positions(known_positions_mm, len(lines))
    accepted = _select_lines(lines, accepted_ids)
    if len(accepted) < MIN_LINES:
        raise UserInputError(
            f"Keep at least {MIN_LINES} reliable ruler lines to fit a transverse calibration."
        )
    by_id = {line.line_id: line for line in lines}
    ordered = sorted(accepted, key=lambda item: by_id[item].column)
    anchor_columns = np.array([by_id[item].column for item in ordered], dtype=np.float64)
    anchor_positions = np.array([positions[item - 1] for item in ordered], dtype=np.float64)
    if np.any(np.diff(anchor_columns) <= 0) or np.any(np.diff(anchor_positions) <= 0):
        raise UserInputError(
            "Ruler lines and their known positions must both increase across the picture."
        )
    weights = np.array([max(by_id[item].score, 0.05) for item in ordered], dtype=np.float64)
    residuals, _ = _weighted_quadratic(anchor_columns, anchor_positions, weights)
    scale = max(np.std(residuals), 1e-6)
    outlier = np.abs(residuals) > OUTLIER_SIGMA * scale
    survivors = ~outlier
    if survivors.sum() < MIN_LINES:
        survivors = np.ones(len(ordered), dtype=bool)
        outlier = np.zeros(len(ordered), dtype=bool)
    anchor_records = []
    for index, line_id in enumerate(ordered):
        line = by_id[line_id]
        anchor_records.append(
            {
                "line_id": line_id,
                "column": round(float(anchor_columns[index]), 5),
                "known_mm": round(float(anchor_positions[index]), 6),
                "fit_residual_mm": round(float(residuals[index]), 6),
                "robust_outlier": bool(outlier[index]),
                "score": float(line.score),
                "stained": bool(line.stained),
                "prominence": round(float(line.prominence), 4),
                "coverage": round(float(line.coverage), 4),
                "straightness": round(float(line.straightness), 4),
            }
        )
    kept_columns = anchor_columns[survivors]
    kept_positions = anchor_positions[survivors]
    transverse = _extrapolated_mapping(
        np.arange(columns, dtype=np.float64), kept_columns, kept_positions
    )
    calibrated = np.zeros(columns, dtype=bool)
    first = int(np.ceil(kept_columns[0]))
    last = int(np.floor(kept_columns[-1]))
    calibrated[max(0, first) : min(columns, last + 1)] = True
    rms = float(np.sqrt(np.average(residuals[survivors] ** 2, weights=weights[survivors])))
    maximum_error = float(np.max(np.abs(residuals[survivors])))
    return TransverseCalibration(
        transverse=transverse.astype(np.float64),
        calibrated=calibrated,
        anchor_columns=kept_columns,
        anchor_positions=kept_positions,
        anchors=tuple(anchor_records),
        source_capture_id=source.get("capture_id"),
        source_name=source.get("name"),
        source_sha256=source.get("sha256"),
        columns=columns,
        rms_error_mm=round(rms, 6),
        maximum_error_mm=round(maximum_error, 6),
        rejected_lines=tuple(
            line.as_dict() for line in lines if line.line_id not in set(accepted)
        ),
        created_at=utc_timestamp(),
        calibration_id=identifier(),
    )


def _parse_positions(values, line_count):
    if not isinstance(values, list) or len(values) != line_count:
        raise UserInputError("Provide one known transverse position per detected line.")
    positions = [
        finite_number(value, f"Ruler line {index + 1} known position", -1_000_000, 1_000_000)
        for index, value in enumerate(values)
    ]
    return positions


def _extrapolated_mapping(query_columns, anchor_columns, anchor_positions):
    """Piecewise-linear mapping with linear extension beyond the outer anchors."""

    inside = np.interp(query_columns, anchor_columns, anchor_positions)
    left_slope = (anchor_positions[1] - anchor_positions[0]) / (
        anchor_columns[1] - anchor_columns[0]
    )
    right_slope = (anchor_positions[-1] - anchor_positions[-2]) / (
        anchor_columns[-1] - anchor_columns[-2]
    )
    left = anchor_positions[0] + (query_columns - anchor_columns[0]) * left_slope
    right = anchor_positions[-1] + (query_columns - anchor_columns[-1]) * right_slope
    return np.where(
        query_columns < anchor_columns[0],
        left,
        np.where(query_columns > anchor_columns[-1], right, inside),
    )


def _select_lines(lines, accepted_ids):
    if accepted_ids is None:
        return tuple(line.line_id for line in lines if line.reliable)
    if not isinstance(accepted_ids, list):
        raise UserInputError("The accepted line list must contain ruler line identifiers.")
    valid = {line.line_id for line in lines}
    selected = []
    for value in accepted_ids:
        line_id = whole_number(value, "Ruler line identifier", 1, MAX_LINES)
        if line_id not in valid:
            raise UserInputError(f"Ruler line {line_id} was not detected in this picture.")
        if line_id not in selected:
            selected.append(line_id)
    return tuple(selected)


@dataclass(frozen=True)
class TransverseCalibration:
    """Piecewise-linear transverse mapping with an explicit validity mask."""

    transverse: np.ndarray
    calibrated: np.ndarray
    anchor_columns: np.ndarray
    anchor_positions: np.ndarray
    anchors: tuple
    rejected_lines: tuple
    columns: int
    rms_error_mm: float
    maximum_error_mm: float
    source_capture_id: str | None
    source_name: str | None
    source_sha256: str | None
    created_at: str
    calibration_id: str = field(default_factory=identifier)

    def __post_init__(self):
        if self.transverse.shape != (self.columns,) or self.calibrated.shape != (self.columns,):
            raise UserInputError("The transverse calibration must provide one value per column.")
        reliable = self.transverse[self.calibrated]
        if reliable.size < 2 or np.any(np.diff(reliable) <= 0):
            raise UserInputError(
                "A transverse calibration must stay strictly increasing inside its reliable span."
            )

    @property
    def reliable_span_mm(self):
        if not np.any(self.calibrated):
            return None
        return float(self.transverse[self.calibrated][-1] - self.transverse[self.calibrated][0])

    @property
    def first_column(self):
        return int(np.flatnonzero(self.calibrated)[0])

    @property
    def last_column(self):
        return int(np.flatnonzero(self.calibrated)[-1])

    def column_at(self, transverse_mm):
        """Inverse mapping: recover the original detector column for a millimetre point."""

        return float(np.interp(transverse_mm, self.transverse, np.arange(self.columns)))

    def is_calibrated_mm(self, transverse_mm):
        column = self.column_at(transverse_mm)
        index = int(np.clip(round(column), 0, self.columns - 1))
        return bool(self.calibrated[index])

    def summary(self):
        return {
            "calibration_id": self.calibration_id,
            "columns": self.columns,
            "calibrated_columns": int(np.count_nonzero(self.calibrated)),
            "calibrated_fraction": float(np.count_nonzero(self.calibrated) / self.columns),
            "first_column": self.first_column,
            "last_column": self.last_column,
            "first_mm": float(self.transverse[self.first_column]),
            "last_mm": float(self.transverse[self.last_column]),
            "reliable_span_mm": self.reliable_span_mm,
            "anchor_count": len(self.anchors),
            "rms_error_mm": self.rms_error_mm,
            "maximum_error_mm": self.maximum_error_mm,
            "source_name": self.source_name,
            "source_capture_id": self.source_capture_id,
            "created_at": self.created_at,
        }

    def anchor_table(self):
        return [
            dict(anchor, calibrated_mm=safe_value(self.transverse[int(round(anchor["column"]))]))
            for anchor in self.anchors
        ]

    def as_dict(self):
        return {
            "calibration_id": self.calibration_id,
            "transverse_mm": [round(float(value), 6) for value in self.transverse],
            "calibrated": [bool(value) for value in self.calibrated],
            "anchor_columns": [round(float(value), 5) for value in self.anchor_columns],
            "anchor_positions_mm": [round(float(value), 6) for value in self.anchor_positions],
            "anchors": [dict(anchor) for anchor in self.anchors],
            "rejected_lines": [dict(line) for line in self.rejected_lines],
            "columns": self.columns,
            "rms_error_mm": self.rms_error_mm,
            "maximum_error_mm": self.maximum_error_mm,
            "source_capture_id": self.source_capture_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "created_at": self.created_at,
        }

    @classmethod
    def parse(cls, data):
        data = expect_object(data, "Transverse calibration")
        exact_keys(
            data,
            (
                "calibration_id",
                "transverse_mm",
                "calibrated",
                "anchor_columns",
                "anchor_positions_mm",
                "anchors",
                "rejected_lines",
                "columns",
                "rms_error_mm",
                "maximum_error_mm",
                "source_capture_id",
                "source_name",
                "source_sha256",
                "created_at",
            ),
            "Transverse calibration",
        )
        columns = whole_number(data["columns"], "Calibration column count", 16, 1024)
        transverse = _numeric_vector(data["transverse_mm"], columns, "Calibration positions")
        calibrated = _boolean_vector(data["calibrated"], columns)
        anchor_columns = np.array(
            [finite_number(v, "Anchor column", -0.5, columns - 0.5) for v in data["anchor_columns"]]
        )
        anchor_positions = np.array(
            [
                finite_number(v, "Anchor position", -1_000_000, 1_000_000)
                for v in data["anchor_positions_mm"]
            ]
        )
        if not isinstance(data["anchors"], list):
            raise UserInputError("The anchor record must be a list.")
        kept = [anchor for anchor in data["anchors"] if not anchor.get("robust_outlier")]
        if len(kept) < MIN_LINES:
            raise UserInputError("A transverse calibration needs at least three kept anchor lines.")
        if not isinstance(data["rejected_lines"], list):
            raise UserInputError("The rejected line record must be a list.")
        return cls(
            transverse=transverse,
            calibrated=calibrated,
            anchor_columns=anchor_columns,
            anchor_positions=anchor_positions,
            anchors=tuple(data["anchors"]),
            rejected_lines=tuple(data["rejected_lines"]),
            columns=columns,
            rms_error_mm=finite_number(data["rms_error_mm"], "Calibration RMS error", 0, 1000),
            maximum_error_mm=finite_number(
                data["maximum_error_mm"], "Calibration maximum error", 0, 1000
            ),
            source_capture_id=_optional_text(data["source_capture_id"]),
            source_name=_optional_text(data["source_name"]),
            source_sha256=_optional_text(data["source_sha256"]),
            created_at=data["created_at"],
            calibration_id=data["calibration_id"],
        )


def _optional_text(value):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 180:
        raise UserInputError("A calibration source label is invalid.")
    return value


def _numeric_vector(values, length, label):
    if not isinstance(values, list) or len(values) != length:
        raise UserInputError(f"{label} must provide {length} numeric entries.")
    return np.array(
        [finite_number(value, label, -1_000_000, 1_000_000) for value in values],
        dtype=np.float64,
    )


def _boolean_vector(values, length):
    if not isinstance(values, list) or len(values) != length or not all(
        isinstance(value, bool) for value in values
    ):
        raise UserInputError(f"The calibration mask must provide {length} true/false entries.")
    return np.array(values, dtype=bool)
