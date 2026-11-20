"""Along-roll motion-smear compensation for a saved review window.

A line-scan row integrates the material over its exposure interval. At high
transport speed that interval covers several station cells, so a fine mark is
recorded as a long shallow patch; at the same station reached at low speed the
mark stays distinct. The blur therefore changes along a coil, and one
compensation for a whole segment would over-correct the slow part and
under-correct the fast part.

The station grid sums the evidence for a row from one or two encoder groups. A
row's recorded value is therefore a weighted average of the exposure intervals of
the lines it uses, placed at those lines' stations. This module builds that
kernel for every row from the per-line transport speed and the uniform exposure
interval, then inverts it with a fixed-count non-negative Richardson-Lucy
deconvolution. The iteration only redistributes recorded intensity, so it cannot
invert noise into excursions the sensor never saw, and an identity kernel leaves
a row untouched.

Rows whose exposure displacement cannot be established — clock pauses, encoder
jumps, repeated ticks at an interruption and capture gaps — are never filled in;
they keep an explicit state so a restored artefact cannot pass for a recorded
feature.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy import sparse

from .calibration import scale_summary
from .grid import Support
from .numerics import true_runs

MINIMUM_ROW_SUPPORT = 0.5
MAXIMUM_PADDING_ROWS = 64
SPEED_WINDOW = 2
MAX_REPORT_ROWS = 240


class SmearState(IntEnum):
    GAP = 0
    UNKNOWN_EXPOSURE = 1
    INSUFFICIENT_EVIDENCE = 2
    BEYOND_LIMIT = 3
    WITHHELD = 4
    RESTORED = 5


STATE_MEANING = {
    SmearState.GAP: "capture gap",
    SmearState.UNKNOWN_EXPOSURE: "motion during exposure unknown",
    SmearState.INSUFFICIENT_EVIDENCE: "sparse evidence",
    SmearState.BEYOND_LIMIT: "blur beyond limit",
    SmearState.WITHHELD: "solution withheld as unstable",
    SmearState.RESTORED: "restored",
}


@dataclass(frozen=True)
class TransportProfile:
    station: np.ndarray
    speed_mm_s: np.ndarray
    exposure_ms: np.ndarray
    known: np.ndarray
    nominal_exposure_ms: float
    unknown_ranges: tuple

    def exposure_distance_mm(self):
        return self.speed_mm_s * self.exposure_ms / 1000.0


@dataclass(frozen=True)
class StationKernels:
    starts: np.ndarray
    offsets: np.ndarray
    widths: np.ndarray
    weights: np.ndarray
    blur_rows: np.ndarray
    blur_mm: np.ndarray
    speed_mm_s: np.ndarray
    exposure_ms: np.ndarray
    state: np.ndarray

    def components(self, row):
        return range(int(self.starts[row]), int(self.starts[row + 1]))


@dataclass(frozen=True)
class SmearRow:
    first_row: int
    last_row: int
    rows: int
    state: SmearState
    blur_rows: float | None
    speed_mm_s: float | None

    def as_dict(self):
        return {
            "first_row": self.first_row,
            "last_row": self.last_row,
            "rows": self.rows,
            "state": self.state.name.lower(),
            "meaning": STATE_MEANING[self.state],
            "blur_rows": self.blur_rows,
            "speed_mm_s": self.speed_mm_s,
        }


@dataclass(frozen=True)
class RestoredRun:
    first_row: int
    last_row: int
    rows: int
    blur_min: float | None
    blur_max: float | None
    residual: float
    iterations: int
    overshoot: float
    withheld: bool

    def as_dict(self):
        return {
            "first_row": self.first_row,
            "last_row": self.last_row,
            "rows": self.rows,
            "blur_rows_min": self.blur_min,
            "blur_rows_max": self.blur_max,
            "residual": self.residual,
            "iterations": self.iterations,
            "overshoot": self.overshoot,
            "withheld": self.withheld,
        }


@dataclass(frozen=True)
class SmearWindow:
    first_row: int
    stations: np.ndarray
    transverse: np.ndarray
    original: np.ndarray
    restored: np.ndarray
    state: np.ndarray
    blur_rows: np.ndarray
    blur_mm: np.ndarray
    speed_mm_s: np.ndarray
    exposure_ms: np.ndarray
    support_fraction: np.ndarray
    original_valid: np.ndarray
    runs: tuple
    row_ranges: tuple
    profile: TransportProfile

    def state_counts(self):
        return {
            state.name.lower(): int(np.count_nonzero(self.state == state.value))
            for state in SmearState
        }

    def changed_fraction(self):
        usable = (self.state == SmearState.RESTORED)[:, None] & np.isfinite(
            self.restored
        ) & np.isfinite(self.original)
        if not usable.any():
            return None
        difference = np.abs(self.restored[usable] - self.original[usable])
        recorded = self.original[usable]
        span = max(float(np.percentile(recorded, 98) - np.percentile(recorded, 2)), 1e-6)
        return float(np.median(difference) / span)


def estimator_lines(acquisition):
    """Intervals that describe ordinary transport rather than an event."""
    return (
        (acquisition.delay_ms <= acquisition.summary["pause_limit_ms"])
        & (acquisition.step_mm > 0)
        & (acquisition.step_mm <= acquisition.summary["gap_limit_mm"])
    )


def nominal_exposure_ms(capture, acquisition):
    """The uniform per-line exposure interval, excluding pauses and jumps."""
    delay = np.diff(capture.seconds) * 1000.0
    usable = delay[estimator_lines(acquisition)]
    if usable.size:
        return float(np.median(usable))
    return float(np.median(delay))


def estimate_transport(capture, acquisition, settings):
    """Per-line along-roll speed and exposure interval.

    Each line's speed is the encoder travel over its surrounding ordinary
    intervals divided by their elapsed time. A line whose averaging window touches
    a pause, jump or repeated tick has no trustworthy speed and stays unknown
    rather than borrowing a neighbour's value.
    """
    steps = acquisition.step_mm
    delay_ms = acquisition.delay_ms
    ordinary = estimator_lines(acquisition)
    intervals = len(steps)
    nominal = (
        float(settings.exposure_ms)
        if settings.exposure_ms is not None
        else nominal_exposure_ms(capture, acquisition)
    )
    speed = np.full(capture.rows, np.nan, dtype=np.float64)
    exposure = np.full(capture.rows, np.nan, dtype=np.float64)
    known = np.zeros(capture.rows, dtype=bool)
    for line in range(capture.rows):
        low = max(0, line - SPEED_WINDOW)
        high = min(intervals, line + SPEED_WINDOW)
        if high - low < 2 or not np.all(ordinary[low:high]):
            continue
        travel = float(np.sum(steps[low:high]))
        elapsed = float(np.sum(delay_ms[low:high])) / 1000.0
        if travel > 0 and elapsed > 0:
            speed[line] = travel / elapsed
            exposure[line] = nominal
            known[line] = True

    breaking = (
        (delay_ms > acquisition.summary["pause_limit_ms"])
        | (steps > acquisition.summary["gap_limit_mm"])
        | (steps <= 0)
    )
    halo = np.zeros(capture.rows, dtype=bool)
    for index in np.flatnonzero(breaking):
        halo[max(0, index - 1) : min(capture.rows, index + 2)] = True
    known &= ~halo
    speed[~known] = np.nan
    exposure[~known] = np.nan

    unknown_ranges = tuple(
        {
            "first_line": int(start),
            "last_line": int(stop - 1),
            "line_count": int(stop - start),
        }
        for start, stop in true_runs(~known)
    )
    return TransportProfile(
        capture.stations.copy(), speed, exposure, known, nominal, unknown_ranges
    )


def station_kernels(capture, grid, profile):
    """The averaging kernel applied by every station grid row.

    A row's value is ``sum_g w_g * box(width_g)`` shifted to the station of group
    ``g``. A GAP row has no measurement at all; a row touching a line without a
    known speed has no determinable kernel and is left unknown.
    """
    plan = grid.plan
    groups = grid.groups
    geometry = capture.metadata.geometry
    distance = profile.exposure_distance_mm()
    pitch = grid.row_pitch_mm
    count = len(grid.stations)
    starts = np.zeros(count + 1, dtype=np.int64)
    offsets = []
    widths = []
    weights = []
    blur_rows = np.full(count, np.nan, dtype=np.float64)
    blur_mm = np.full(count, np.nan, dtype=np.float64)
    speed = np.full(count, np.nan, dtype=np.float64)
    exposure = np.full(count, np.nan, dtype=np.float64)
    state = np.full(count, SmearState.UNKNOWN_EXPOSURE, dtype=np.uint8)
    for row in range(count):
        support = Support(plan.support[row])
        starts[row + 1] = len(offsets)
        if support == Support.GAP:
            state[row] = SmearState.GAP
            continue
        sources = [(int(plan.left[row]), 1.0 - float(plan.right_weight[row]))]
        if support == Support.BLEND:
            sources.append((int(plan.right[row]), float(plan.right_weight[row])))
        components = []
        complete = True
        for group, weight in sources:
            members = list(groups.sources(group))
            if not all(profile.known[line] for line in members):
                complete = False
                break
            station = float(geometry.station(groups.ticks[group]))
            components.append(
                (
                    (station - float(grid.stations[row])) / pitch,
                    float(np.mean(distance[members])) / pitch,
                    weight,
                    float(np.mean(profile.speed_mm_s[members])),
                    float(np.mean(profile.exposure_ms[members])),
                    float(np.mean(distance[members])),
                )
            )
        if not complete:
            continue
        total = sum(item[2] for item in components)
        variance = 0.0
        weighted_speed = 0.0
        weighted_exposure = 0.0
        weighted_distance = 0.0
        for offset, width, weight, share_speed, share_exposure, share_distance in components:
            share = weight / total
            offsets.append(offset)
            widths.append(max(width, 1e-6))
            weights.append(share)
            variance += share * (offset**2 + width**2 / 12.0)
            weighted_speed += share * share_speed
            weighted_exposure += share * share_exposure
            weighted_distance += share * share_distance
        blur_rows[row] = np.sqrt(12.0 * variance)
        blur_mm[row] = weighted_distance
        speed[row] = weighted_speed
        exposure[row] = weighted_exposure
        state[row] = SmearState.RESTORED
    return StationKernels(
        starts,
        np.array(offsets, dtype=np.float64),
        np.array(widths, dtype=np.float64),
        np.array(weights, dtype=np.float64),
        blur_rows,
        blur_mm,
        speed,
        exposure,
        state,
    )


def window_rows(window, grid):
    return window.indices(grid.stations, grid.transverse)


def blur_operator(kernels, rows):
    """Sparse along-roll averaging operator for a contiguous row range."""
    first = rows.start
    count = rows.stop - rows.start
    operator_rows = []
    operator_columns = []
    operator_values = []
    for position in range(count):
        source = first + position
        begin = int(kernels.starts[source])
        end = int(kernels.starts[source + 1])
        mass = 0.0
        entries = []
        for component in range(begin, end):
            center = position + kernels.offsets[component]
            width = max(kernels.widths[component], 1e-6)
            low = center - width / 2.0
            high = center + width / 2.0
            share = kernels.weights[component]
            start = max(0, int(np.ceil(low - 0.5)))
            stop = min(count - 1, int(np.floor(high + 0.5)))
            for target in range(start, stop + 1):
                overlap = min(high, target + 0.5) - max(low, target - 0.5)
                if overlap > 1e-12:
                    entries.append((target, share * overlap / width))
                    mass += share * overlap / width
        if not entries or mass <= 0:
            entries = [(position, 1.0)]
            mass = 1.0
        for target, value in entries:
            operator_rows.append(position)
            operator_columns.append(target)
            operator_values.append(value / mass)
    return sparse.csr_matrix(
        (
            np.array(operator_values),
            (np.array(operator_rows), np.array(operator_columns)),
        ),
        shape=(count, count),
    )


def blended_rows(values, valid):
    """Interpolate unusable cells so the solve sees an unbroken row signal."""
    if valid.all():
        return values
    filled = values.copy()
    index = np.arange(values.shape[0])
    for column in range(values.shape[1]):
        mask = valid[:, column]
        if mask.all():
            continue
        if not mask.any():
            filled[:, column] = 0.0
            continue
        filled[~mask, column] = np.interp(index[~mask], index[mask], values[mask, column])
    return filled


def solve(operator, values, iterations):
    """Non-negative Richardson-Lucy deconvolution of one run of station rows.

    The iteration only redistributes the recorded intensity: it never produces a
    negative reflectance, and with an identity operator it leaves the input
    unchanged. A fixed, small iteration count bounds how far fine-scale noise can
    be sharpened, so amplified noise cannot grow into apparent new marks. The
    result is what the recorded signal supports, not an unbounded inverse.
    """
    count = operator.shape[0]
    if count < 3:
        return np.asarray(operator @ values, dtype=np.float64), 0
    observed = np.maximum(np.where(np.isfinite(values), values, 0.0), 1e-4)
    tolerance = max(float(np.max(observed)) * 1e-5, 1e-9)
    limit = np.maximum(np.asarray(operator.T @ np.ones((count, 1))), 1e-12)
    transpose = operator.T.tocsr()
    outcome = observed.copy()
    steps = 0
    for _ in range(iterations):
        ratio = observed / np.maximum(operator @ outcome, tolerance)
        updated = outcome * np.asarray(transpose @ ratio) / limit
        np.maximum(updated, 0.0, out=updated)
        change = float(np.max(np.abs(updated - outcome)))
        outcome = updated
        steps += 1
        if change <= tolerance:
            break
    return outcome, steps


def row_state(grid, rows, kernels, valid, settings):
    support = np.count_nonzero(valid, axis=1) / valid.shape[1]
    resolved = kernels.state[rows].copy()
    width = kernels.blur_rows[rows]
    resolved[(resolved == SmearState.RESTORED) & (support < MINIMUM_ROW_SUPPORT)] = (
        SmearState.INSUFFICIENT_EVIDENCE
    )
    resolved[(resolved == SmearState.RESTORED) & (width > settings.maximum_blur_rows)] = (
        SmearState.BEYOND_LIMIT
    )
    return resolved, support


def overshoot(values, patch):
    """How far a solve leaves the reflectance range recorded in its run.

    Richardson-Lucy is non-negative, but it can still push a sharpened edge
    slightly past the extremes the sensor recorded. That overshoot is the useful
    signal for "invented values": it is independent of how flat the material is.
    The range is widened by a small margin to allow for sampling noise, and the
    result is expressed relative to the recorded span.
    """
    finite = patch[np.isfinite(patch)]
    if not finite.size:
        return 0.0
    low, high = np.percentile(finite, [0.5, 99.5])
    span = max(high - low, 1e-6)
    margin = 0.02 * span
    excess = max(0.0, float(np.min(values)) - (low - margin))
    excess = max(excess, float(np.max(values)) - (high + margin))
    return excess / span


def restore_window(context, window):
    grid = context.grid
    settings = context.settings.smear
    profile = context.transport
    kernels = context.station_kernels

    rows, columns = window_rows(window, grid)
    values = grid.values[rows, columns].astype(np.float64)
    valid = grid.valid[rows, columns]
    state, support = row_state(grid, rows, kernels, valid, settings)

    finite = kernels.blur_rows[rows][state == SmearState.RESTORED]
    padding = 2 + (int(np.ceil(float(np.max(finite)))) if finite.size else 0)
    padding = min(padding, MAXIMUM_PADDING_ROWS)
    first = max(0, rows.start - padding)
    stop = min(grid.shape[0], rows.stop + padding)
    span = slice(first, stop)
    padded_valid = grid.valid[span, columns]
    padded_values = grid.values[span, columns].astype(np.float64)
    padded_state, _ = row_state(grid, span, kernels, padded_valid, settings)
    padded_width = kernels.blur_rows[span]

    operator = blur_operator(kernels, span)
    restored = np.full(padded_values.shape, np.nan, dtype=np.float64)
    runs = []
    local_start = rows.start - first
    local_stop = rows.stop - first
    for start, finish in true_runs(padded_state == SmearState.RESTORED):
        patch_valid = padded_valid[start:finish]
        patch = blended_rows(padded_values[start:finish], patch_valid)
        piece = operator[start:finish, start:finish]
        outcome, steps = solve(piece, patch, settings.iterations)
        # The iteration sharpens noise with the detail. Where its result leaves
        # the reflectance range recorded in the run by more than the supported
        # fraction, the excess would read as a new dark or bright feature, so the
        # run keeps its recorded values and is reported as withheld instead. The
        # allowance grows with the blur width, because a wider kernel genuinely
        # needs a larger excursion to restore the same feature.
        excess = overshoot(outcome, patch)
        widths = padded_width[start:finish]
        allowance = settings.maximum_overshoot * max(
            1.0, float(np.nanmean(widths)) if np.isfinite(widths).any() else 1.0
        )
        withheld = excess > allowance
        padded_state[start:finish] = (
            SmearState.WITHHELD if withheld else SmearState.RESTORED
        )
        if not withheld:
            restored[start:finish] = outcome
        visible_first = max(start, local_start)
        visible_stop = min(finish, local_stop)
        if visible_stop > visible_first:
            local = slice(visible_first - start, visible_stop - start)
            residual = float(np.sqrt(np.mean((piece @ outcome - patch) ** 2)))
            widths = padded_width[start:finish][local]
            widths = widths[np.isfinite(widths)]
            runs.append(
                RestoredRun(
                    int(first + visible_first),
                    int(first + visible_stop - 1),
                    int(visible_stop - visible_first),
                    float(np.min(widths)) if widths.size else None,
                    float(np.max(widths)) if widths.size else None,
                    residual,
                    steps,
                    float(excess),
                    bool(withheld),
                )
            )

    local = slice(local_start, local_stop)
    state = padded_state[local]
    restored = restored[local]

    blur_rows = kernels.blur_rows[rows]
    blur_mm = kernels.blur_mm[rows]
    speed = kernels.speed_mm_s[rows]
    row_ranges = []
    for state_value in SmearState:
        if state_value == SmearState.RESTORED:
            continue
        for start, finish in true_runs(state == state_value):
            widths = blur_rows[start:finish][np.isfinite(blur_rows[start:finish])]
            rates = speed[start:finish][np.isfinite(speed[start:finish])]
            row_ranges.append(
                SmearRow(
                    int(rows.start + start),
                    int(rows.start + finish - 1),
                    int(finish - start),
                    state_value,
                    float(np.median(widths)) if widths.size else None,
                    float(np.median(rates)) if rates.size else None,
                )
            )
    row_ranges.sort(key=lambda item: item.first_row)

    return SmearWindow(
        int(rows.start),
        grid.stations[rows],
        grid.transverse[columns],
        values.astype(np.float32),
        np.where((state == SmearState.RESTORED)[:, None], restored, np.nan).astype(
            np.float32
        ),
        state,
        blur_rows,
        blur_mm,
        speed,
        kernels.exposure_ms[rows],
        support,
        valid,
        tuple(runs),
        tuple(row_ranges),
        profile,
    )


def transport_summary(context):
    """Capture-wide exposure and blur model, without inverting any window."""
    profile = context.transport
    kernels = context.station_kernels
    grid = context.grid
    speed = profile.speed_mm_s[profile.known]
    distance = profile.exposure_distance_mm()[profile.known]
    known_rows = kernels.state == SmearState.RESTORED
    blur_rows = kernels.blur_rows[known_rows]
    exposure_rows = kernels.blur_mm[known_rows]
    return {
        "capture_lines": int(len(profile.known)),
        "lines_with_known_motion": int(np.count_nonzero(profile.known)),
        "lines_with_unknown_motion": int(np.count_nonzero(~profile.known)),
        "unknown_motion_spans": len(profile.unknown_ranges),
        "station_rows": int(len(kernels.state)),
        "rows_with_known_exposure": int(np.count_nonzero(known_rows)),
        "rows_with_unknown_exposure": int(np.count_nonzero(~known_rows)),
        "nominal_exposure_ms": profile.nominal_exposure_ms,
        "speed_mm_s": scale_summary(speed),
        "exposure_distance_mm": scale_summary(distance),
        "blur_rows_at_pitch": scale_summary(blur_rows),
        "blur_rows_at_pitch_mm": scale_summary(exposure_rows),
        "row_pitch_mm": grid.row_pitch_mm,
        "maximum_blur_rows": context.settings.smear.maximum_blur_rows,
    }


def unknown_line_table(context):
    profile = context.transport
    return [
        item
        | {
            "station_from_mm": float(context.capture.stations[item["first_line"]]),
            "station_to_mm": float(context.capture.stations[item["last_line"]]),
        }
        for item in profile.unknown_ranges
    ]


def profile_series(context, max_points=MAX_REPORT_ROWS):
    kernels = context.station_kernels
    grid = context.grid
    count = len(kernels.state)
    chosen = (
        np.unique(np.linspace(0, count - 1, max_points, dtype=np.int64))
        if count > max_points
        else np.arange(count)
    )

    def number(value):
        return float(value) if np.isfinite(value) else None

    return [
        {
            "row": int(index),
            "station_mm": float(grid.stations[index]),
            "speed_mm_s": number(kernels.speed_mm_s[index]),
            "exposure_mm": number(kernels.blur_mm[index]),
            "blur_rows": number(kernels.blur_rows[index]),
            "state": SmearState(int(kernels.state[index])).name.lower(),
        }
        for index in chosen
    ]


def summary(report, context):
    counts = report.state_counts()
    restored = counts["restored"]
    total = len(report.state)

    def finite(values):
        return values[np.isfinite(values)]

    change = report.changed_fraction()
    return {
        "station_from_mm": float(report.stations[0]),
        "station_to_mm": float(report.stations[-1]),
        "rows": total,
        "columns": len(report.transverse),
        "restored_rows": restored,
        "restored_fraction": restored / total if total else 0.0,
        "gap_rows": counts["gap"],
        "unknown_exposure_rows": counts["unknown_exposure"],
        "sparse_rows": counts["insufficient_evidence"],
        "beyond_limit_rows": counts["beyond_limit"],
        "nominal_exposure_ms": report.profile.nominal_exposure_ms,
        "speed_mm_s": scale_summary(finite(report.speed_mm_s)),
        "exposure_distance_mm": scale_summary(finite(report.blur_mm)),
        "blur_rows": scale_summary(finite(report.blur_rows)),
        "restored_runs": sum(not run.withheld for run in report.runs),
        "withheld_runs": sum(run.withheld for run in report.runs),
        "median_change": change,
        "overshoot": scale_summary([run.overshoot for run in report.runs]),
        "median_residual": scale_summary([run.residual for run in report.runs]),
        "maximum_overshoot": context.settings.smear.maximum_overshoot,
        "maximum_blur_rows": context.settings.smear.maximum_blur_rows,
        "unknown_source_lines": sum(
            item["line_count"] for item in report.profile.unknown_ranges
        ),
    }


def state_table(report):
    return [item.as_dict() for item in report.row_ranges]


def run_table(report):
    return [run.as_dict() for run in report.runs]
