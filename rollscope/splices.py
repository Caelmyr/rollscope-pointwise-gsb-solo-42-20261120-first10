from dataclasses import asdict, dataclass

import numpy as np

from .calibration import safe_value, scale_summary
from .numerics import segmented_reference, true_runs


# A row keeps an evidence run only when it spans at least this many columns.
MINIMUM_RUN_COLUMNS = 4
# Consecutive observed rows without a matching run before a segment is closed.
MAXIMUM_MISSED_ROWS = 3
# Observed, contrast-free stretches shorter than this still count as bridgeable.
SHORT_HOLE_MM = 2.5


@dataclass(frozen=True)
class EvidenceRun:
    row: int
    first_col: int
    last_col: int
    sign: int
    cells: int
    strength: float

    @property
    def center(self):
        return 0.5 * (self.first_col + self.last_col)


@dataclass
class RawSegment:
    sign: int
    runs: list

    @property
    def cells(self):
        return sum(run.cells for run in self.runs)

    @property
    def first_row(self):
        return min(run.row for run in self.runs)

    @property
    def last_row(self):
        return max(run.row for run in self.runs)


@dataclass(frozen=True)
class FittedSegment:
    sign: int
    runs: tuple
    intercept: float
    slope_row_per_col: float
    fit_error_cols: float

    @property
    def first_row(self):
        return min(run.row for run in self.runs)

    @property
    def last_row(self):
        return max(run.row for run in self.runs)

    @property
    def first_col(self):
        return min(run.first_col for run in self.runs)

    @property
    def last_col(self):
        return max(run.last_col for run in self.runs)

    @property
    def cells(self):
        return sum(run.cells for run in self.runs)


@dataclass(frozen=True)
class Splice:
    id: int
    track_id: int
    station_mm: float
    polarity: str
    coverage: float
    contrast: float
    left_mm: float
    right_mm: float
    before_mm: float
    after_mm: float
    slope: float
    evidence_fraction: float
    link: str
    gap_mm: float

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SpliceLink:
    id: int
    track_id: int
    polarity: str
    station_before_mm: float
    station_after_mm: float
    transverse_mm: float
    gap_mm: float
    gap_kind: str
    status: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SpliceReport:
    stations: np.ndarray
    score: np.ndarray
    covered_fraction: np.ndarray
    splices: tuple
    links: tuple
    geometry: tuple = ()

    def summary(self):
        track_ids = {splice.track_id for splice in self.splices}
        return {
            "join_tracks": len(track_ids),
            "join_segments": len(self.splices),
            "continued_gaps": sum(link.status == "continued" for link in self.links),
            "ambiguous_gaps": sum(link.status == "ambiguous" for link in self.links),
            "contrast": scale_summary([splice.contrast for splice in self.splices]),
            "width_fraction": scale_summary([splice.coverage for splice in self.splices]),
            "evaluated_rows": int(np.count_nonzero(np.isfinite(self.score))),
        }

    def series(self, indices):
        return [
            {
                "station_mm": float(self.stations[index]),
                "score": safe_value(self.score[index]),
                "width_fraction": safe_value(self.covered_fraction[index]),
            }
            for index in indices
        ]

    def table(self):
        return [splice.as_dict() for splice in self.splices]

    def links_table(self):
        return [link.as_dict() for link in self.links]


def directional_votes(residual, valid, minimum_contrast, minimum_width):
    """Per-row same-sign contrast vote; retained for the station score chart."""
    strengths = np.zeros(residual.shape[0], dtype=np.float32)
    fractions = np.full(residual.shape[0], np.nan, dtype=np.float32)
    signs = np.zeros(residual.shape[0], dtype=np.int8)
    bright = valid & (residual >= minimum_contrast)
    dark = valid & (residual <= -minimum_contrast)
    for row in range(residual.shape[0]):
        width = np.count_nonzero(valid[row])
        if width < minimum_width:
            continue
        bright_count = np.count_nonzero(bright[row])
        dark_count = np.count_nonzero(dark[row])
        sign = 1 if bright_count >= dark_count else -1
        selected = bright[row] if sign > 0 else dark[row]
        count = max(bright_count, dark_count)
        fractions[row] = count / width
        signs[row] = sign
        if count:
            strengths[row] = float(np.median(np.abs(residual[row, selected])))
    return strengths, fractions, signs


def row_evidence_runs(residual, valid, sign, minimum_contrast, minimum_cells):
    signed = np.where(valid, residual * sign, np.nan)
    hits = valid & (signed >= minimum_contrast)
    for row in range(residual.shape[0]):
        for first, last in true_runs(hits[row]):
            cells = last - first
            if cells < minimum_cells:
                continue
            strength = float(np.median(signed[row, first:last]))
            yield EvidenceRun(row, int(first), int(last - 1), sign, int(cells), strength)


def merge_same_row_runs(runs, tolerance_cols):
    """Join same-polarity runs on one row when only a short blind gap separates them."""
    merged = []
    for run in runs:
        if (
            merged
            and merged[-1].row == run.row
            and run.first_col - merged[-1].last_col <= tolerance_cols + 1
        ):
            previous = merged[-1]
            cells = previous.cells + run.cells
            strength = (
                previous.strength * previous.cells + run.strength * run.cells
            ) / cells
            merged[-1] = EvidenceRun(
                run.row,
                previous.first_col,
                run.last_col,
                run.sign,
                previous.cells + run.cells,
                strength,
            )
        else:
            merged.append(run)
    return merged


def seed_segments(runs):
    segments = []
    for run in runs:
        occupied = [segment for segment in segments if segment.runs[-1].row == run.row]
        if occupied:
            # Runs on the same row are separate evidence; each opens its own line.
            segments.append(RawSegment(run.sign, [run]))
            continue
        candidates = [
            segment
            for segment in segments
            if segment.sign == run.sign and run.row - segment.runs[-1].row <= MAXIMUM_MISSED_ROWS
        ]
        best = None
        best_distance = None
        for segment in candidates:
            previous = segment.runs[-1]
            row_step = run.row - previous.row
            drift = 0.0
            if len(segment.runs) >= 2:
                drift = segment.runs[-1].center - segment.runs[-2].center
            expected = previous.center + drift * max(0, row_step - 1)
            distance = abs(run.center - expected)
            if best_distance is None or distance < best_distance:
                best = segment
                best_distance = distance
        if best is None:
            segments.append(RawSegment(run.sign, [run]))
        else:
            best.runs.append(run)
    return segments


def fit_segment(segment):
    # Fit through run centres, weighted by evidence count. Centre points are
    # more stable than run edges when a fragment is short or partially blind.
    centres = np.array([run.center for run in segment.runs], dtype=np.float64)
    rows = np.array([run.row for run in segment.runs], dtype=np.float64)
    weights = np.array(
        [max(1, run.cells) for run in segment.runs], dtype=np.float64
    )
    if len(rows) >= 2 and np.ptp(centres) > 0:
        slope, intercept = np.polyfit(centres, rows, 1, w=weights)
    else:
        slope, intercept = 0.0, float(np.average(rows, weights=weights))
    residuals = rows - (intercept + slope * centres)
    error = float(np.sqrt(np.average(residuals**2, weights=weights)))
    return FittedSegment(
        segment.sign, segment.runs, float(intercept), float(slope), error
    )


def segment_evidence_fraction(segment, valid, band_cols=None):
    """Share of measured cells along the evidence line that carry the contrast.

    Each run's own transverse extent is used as the local band, padded by
    ``band_cols`` so short blind interruptions count as unsampled rather than
    absent evidence. A diagonal join is therefore judged on the strip it
    actually crosses, not the full material width of every row.
    """
    padding = 0 if band_cols is None else max(1, band_cols // 8)
    first_row, last_row = segment.first_row, segment.last_row
    observed = np.zeros(last_row - first_row + 1, dtype=np.int64)
    hit = np.zeros_like(observed)
    for run in segment.runs:
        offset = run.row - first_row
        left = max(0, run.first_col - padding)
        right = min(valid.shape[1], run.last_col + padding + 1)
        observed[offset] += np.count_nonzero(valid[run.row, left:right])
        hit[offset] += min(run.cells, observed[offset])
    total_observed = int(observed.sum())
    if total_observed == 0:
        return 0.0
    return float(min(hit.sum(), total_observed) / total_observed)


def keep_segments(segments, valid, settings, row_pitch_mm, column_pitch_mm):
    minimum_cells = MINIMUM_RUN_COLUMNS * max(2, int(np.ceil(6 / row_pitch_mm)))
    minimum_span_mm = max(4.0, settings.track_tolerance_mm * 2)
    band_cols = max(6, int(np.ceil(settings.track_tolerance_mm * 3 / column_pitch_mm)))
    accepted = []
    for raw in segments:
        fitted = fit_segment(raw)
        span_mm = (fitted.last_col - fitted.first_col) * column_pitch_mm
        slope_mm = fitted.slope_row_per_col * column_pitch_mm / row_pitch_mm
        fraction = segment_evidence_fraction(fitted, valid, band_cols)
        if fitted.cells < minimum_cells:
            continue
        if span_mm < minimum_span_mm:
            continue
        if abs(slope_mm) > settings.max_slope:
            continue
        if fraction < settings.minimum_evidence_fraction:
            continue
        accepted.append(fitted)
    return merge_collinear(accepted, settings, row_pitch_mm, column_pitch_mm)


def merge_collinear(segments, settings, row_pitch_mm, column_pitch_mm):
    """Merge same-polarity pieces of one roughly straight line.

    Two nearby parallel lines stay separate because their transverse separation
    exceeds the tracking tolerance; fragmented runs of one line fall inside it.
    """
    tolerance_cols = max(1.0, settings.track_tolerance_mm / column_pitch_mm)
    max_row_gap = max(1, int(round(settings.minimum_spacing_mm / row_pitch_mm)))
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        for index, segment in enumerate(merged):
            for other_index in range(index + 1, len(merged)):
                other = merged[other_index]
                if other.sign != segment.sign:
                    continue
                row_gap = max(
                    0,
                    max(segment.first_row, other.first_row)
                    - min(segment.last_row, other.last_row),
                )
                if row_gap > max_row_gap:
                    continue
                # Transverse ranges must overlap or nearly touch.
                column_gap = max(segment.first_col, other.first_col) - min(
                    segment.last_col, other.last_col
                )
                if column_gap > tolerance_cols:
                    continue
                # Pieces sharing a row are fragments of one transverse line
                # even when their fitted centres differ across the width.
                shares_row = not (
                    segment.last_row < other.first_row or other.last_row < segment.first_row
                )
                if not shares_row:
                    check_row = min(segment.last_row, other.last_row)
                    if (
                        abs(line_at_row(segment, check_row) - line_at_row(other, check_row))
                        > tolerance_cols
                    ):
                        continue
                slope_limit = abs(segment.slope_row_per_col - other.slope_row_per_col) * max(
                    segment.last_col, other.last_col, 1
                )
                if slope_limit > tolerance_cols:
                    continue
                combined = RawSegment(
                    segment.sign, list(segment.runs) + list(other.runs)
                )
                merged[index] = fit_segment(combined)
                del merged[other_index]
                changed = True
                break
            if changed:
                break
    return merged


def line_at_row(segment, row):
    return (row - segment.intercept) / segment.slope_row_per_col if segment.slope_row_per_col else (
        0.5 * (segment.first_col + segment.last_col)
    )


def line_endpoints(segment):
    if segment.slope_row_per_col:
        start_col = line_at_row(segment, segment.first_row)
        end_col = line_at_row(segment, segment.last_row)
    else:
        start_col = segment.first_col
        end_col = segment.last_col
    return start_col, end_col


def extrapolated_separation(previous, following, tolerance_cols):
    """Collinearity distance across the gap, in columns.

    Each segment's fitted line is extrapolated to the other's boundary. A
    diagonal join has no transverse overlap across an encoder gap, so the test
    relies on line continuity and matching slope rather than shared columns.
    Short fragments carry an unstable intercept, so their position budget
    grows gently with the gap they have to bridge.
    """
    if following.first_row <= previous.last_row:
        return None
    gap_rows = following.first_row - previous.last_row
    previous_at_gap = line_at_row(previous, following.first_row)
    following_at_gap = line_at_row(following, previous.last_row)
    forward = abs(following.first_col - previous_at_gap)
    backward = abs(previous.last_col - following_at_gap)
    extent = max(previous.last_col, following.last_col, 1)
    slope_limit = abs(previous.slope_row_per_col - following.slope_row_per_col) * extent
    slope_budget = max(tolerance_cols, 0.04 * extent)
    if slope_limit > slope_budget:
        return None
    position_budget = tolerance_cols + 0.5 * gap_rows * abs(
        previous.slope_row_per_col - following.slope_row_per_col
    ) + 0.18 * gap_rows
    if forward <= position_budget and backward <= position_budget:
        return max(forward, backward) + 0.05 * gap_rows
    return None


def observed_gap_rows(observed, previous, following):
    between = observed[previous.last_row + 1 : following.first_row]
    return int(np.count_nonzero(between)), int(len(between))


def link_segments(segments, observed, settings, grid):
    """Join collinear same-polarity pieces across missing or unsupported stretches."""
    row_pitch = grid.row_pitch_mm
    tolerance_cols = max(1.0, settings.track_tolerance_mm / grid.column_pitch_mm)
    max_gap_rows = settings.max_gap_mm / row_pitch
    short_hole_rows = max(1, int(round(SHORT_HOLE_MM / row_pitch)))
    links = []
    track_of = {id(segment): index for index, segment in enumerate(segments)}
    next_track = len(segments)

    ordered = sorted(segments, key=lambda item: (item.last_row, item.first_col))
    for index, previous in enumerate(ordered):
        candidates = []
        for following in ordered[index + 1 :]:
            if following.sign != previous.sign:
                continue
            if following.first_row <= previous.last_row:
                continue
            gap_rows = following.first_row - previous.last_row
            if gap_rows > max_gap_rows:
                continue
            separation = extrapolated_separation(previous, following, tolerance_cols)
            if separation is None:
                continue
            observed_count, between_count = observed_gap_rows(observed, previous, following)
            unsupported = between_count - observed_count
            if unsupported == 0 and observed_count > short_hole_rows:
                # Measured material with no join evidence: do not paper over it.
                continue
            candidates.append((separation, following, unsupported > 0))
        if not candidates:
            continue
        chosen = min(candidates, key=lambda item: item[0])
        ambiguous = len(candidates) > 1
        following = chosen[1]
        unsupported_gap = chosen[2]
        gap_rows = following.first_row - previous.last_row
        gap_kind = "unsupported" if unsupported_gap else "observed"
        status = "ambiguous" if ambiguous else "continued"
        links.append((previous, following, gap_rows, gap_kind, status))
        if not ambiguous:
            track_of[id(following)] = track_of[id(previous)]
        else:
            track_of[id(following)] = next_track
            next_track += 1

    # Relabel tracks as dense 1..n identifiers ordered by their first station.
    groups = {}
    for segment in segments:
        groups.setdefault(track_of[id(segment)], []).append(segment)
    ordered_groups = sorted(
        groups.values(), key=lambda members: min(s.first_row for s in members)
    )
    track_number = {
        id(segment): number
        for number, members in enumerate(ordered_groups, 1)
        for segment in members
    }
    return links, track_number


def build_records(segments, links, track_number, grid, valid, band_cols):
    row_pitch = grid.row_pitch_mm
    column_pitch = grid.column_pitch_mm
    material_width = np.diff(np.flatnonzero(valid.any(axis=0))[[0, -1]])[0] * column_pitch
    material_width = max(material_width, 1.0)
    link_by_segment = {id(previous): (following, gap_rows, kind, status)
                       for previous, following, gap_rows, kind, status in links}

    splices = []
    link_records = []
    ordered = sorted(segments, key=lambda item: (item.first_row, item.first_col))
    for index, segment in enumerate(ordered, 1):
        start_col, end_col = line_endpoints(segment)
        slope_mm = segment.slope_row_per_col * column_pitch / row_pitch
        strength = float(np.median([run.strength for run in segment.runs]))
        coverage = (segment.last_col - segment.first_col) * column_pitch / material_width
        connection = link_by_segment.get(id(segment))
        link_state = connection[3] if connection else "end"
        gap_mm = connection[1] * row_pitch if connection else 0.0
        splices.append(
            Splice(
                index,
                track_number[id(segment)],
                float(grid.stations[segment.first_row]),
                "bright" if segment.sign > 0 else "dark",
                float(coverage),
                strength,
                float(grid.transverse[segment.first_col]),
                float(grid.transverse[segment.last_col]),
                float(grid.stations[segment.first_row]),
                float(grid.stations[segment.last_row]),
                float(slope_mm),
                float(segment_evidence_fraction(segment, valid, band_cols)),
                link_state,
                float(gap_mm),
            )
        )

    for identifier, (previous, following, gap_rows, kind, status) in enumerate(links, 1):
        join_col = 0.5 * (line_at_row(previous, following.first_row) + following.first_col)
        link_records.append(
            SpliceLink(
                identifier,
                track_number[id(previous)],
                "bright" if previous.sign > 0 else "dark",
                float(grid.stations[previous.last_row]),
                float(grid.stations[following.first_row]),
                float(grid.transverse[0] + join_col * column_pitch),
                float(gap_rows * row_pitch),
                kind,
                status,
            )
        )
    return tuple(splices), tuple(link_records)


def enforce_track_width(splices, links, segments, track_of_segment, minimum_fraction):
    """A complete tracked join must span the required share of the strip width.

    The span is measured over the whole track (the union of its measured
    pieces), so a diagonal line split by an encoder gap is not rejected just
    because each measured fragment covers only part of the material width.
    """
    track_extent = {}
    for segment in segments:
        track = track_of_segment[id(segment)]
        left, right = track_extent.get(track, (segment.first_col, segment.last_col))
        track_extent[track] = (min(left, segment.first_col), max(right, segment.last_col))
    widest = max(
        (right - left for left, right in track_extent.values()), default=1
    )
    kept_tracks = {
        track
        for track, (left, right) in track_extent.items()
        if (right - left) / widest >= minimum_fraction
    }
    kept = tuple(splice for splice in splices if splice.track_id in kept_tracks)
    kept_links = tuple(link for link in links if link.track_id in kept_tracks)
    splices, links, track_map = renumber(kept, kept_links)
    kept_segments = tuple(
        (segment, track_map[track_of_segment[id(segment)]])
        for segment in segments
        if track_of_segment[id(segment)] in kept_tracks
    )
    return splices, links, kept_segments


def renumber(splices, links):
    track_map = {
        track_id: number
        for number, track_id in enumerate(
            sorted({splice.track_id for splice in splices}), 1
        )
    }
    splices = tuple(
        Splice(
            number,
            track_map[splice.track_id],
            splice.station_mm,
            splice.polarity,
            splice.coverage,
            splice.contrast,
            splice.left_mm,
            splice.right_mm,
            splice.before_mm,
            splice.after_mm,
            splice.slope,
            splice.evidence_fraction,
            splice.link,
            splice.gap_mm,
        )
        for number, splice in enumerate(splices, 1)
    )
    links = tuple(
        SpliceLink(
            number,
            track_map[link.track_id],
            link.polarity,
            link.station_before_mm,
            link.station_after_mm,
            link.transverse_mm,
            link.gap_mm,
            link.gap_kind,
            link.status,
        )
        for number, link in enumerate(links, 1)
    )
    return splices, links, track_map


def join_residuals(grid, material, settings):
    """Complementary local backgrounds for horizontal and diagonal joins.

    A tall longitudinal window exposes a near-horizontal join. A wide
    transverse window exposes a diagonal line, whose transverse footprint on
    each row would otherwise be averaged into an isotropic background. A
    compact isotropic window catches short oblique pieces near strip edges.
    Tracking takes the union of evidence from all three bands.
    """
    bands = []
    wide_sigma = (settings.local_scale_mm / grid.row_pitch_mm, 0.2)
    wide_bg, wide_weights = segmented_reference(grid.values, material, wide_sigma, 0.5)
    bands.append((np.where(material & (wide_weights >= 0.5), grid.values - wide_bg, np.nan),
                  material & (wide_weights >= 0.5)))

    compact_sigma = (
        max(0.6, 1.5 / grid.row_pitch_mm),
        max(0.6, 1.5 / grid.column_pitch_mm),
    )
    compact_bg, compact_weights = segmented_reference(
        grid.values, material, compact_sigma, 0.5
    )
    bands.append((
        np.where(material & (compact_weights >= 0.5), grid.values - compact_bg, np.nan),
        material & (compact_weights >= 0.5),
    ))

    diagonal_sigma = (
        max(1.0, 2.2 / grid.row_pitch_mm),
        max(20.0, 11.0 / grid.column_pitch_mm),
    )
    diagonal_bg, diagonal_weights = segmented_reference(
        grid.values, material, diagonal_sigma, 0.5
    )
    bands.append((
        np.where(material & (diagonal_weights >= 0.5), grid.values - diagonal_bg, np.nan),
        material & (diagonal_weights >= 0.5),
    ))
    return bands


def inspect_splices(grid, edges, settings):
    material = edges.inner_mask(3)
    bands = join_residuals(grid, material, settings)
    wide_residual, wide_valid = bands[0]
    minimum_width = max(6, grid.shape[1] // 6)
    strength, coverage, _ = directional_votes(
        wide_residual, wide_valid, settings.contrast, minimum_width
    )
    score = np.where(np.isfinite(coverage), strength * coverage, np.nan)

    run_width = max(MINIMUM_RUN_COLUMNS, grid.shape[1] // 64)
    join_tolerance_cols = max(1.0, settings.track_tolerance_mm / grid.column_pitch_mm)
    evidence_valid = np.zeros_like(material)
    raw_segments = []
    for sign in (-1, 1):
        runs = []
        for band_residual, band_valid in bands:
            evidence_valid |= band_valid
            runs.extend(
                row_evidence_runs(
                    band_residual, band_valid, sign, settings.contrast, run_width
                )
            )
        runs.sort(key=lambda run: (run.row, run.first_col))
        runs = merge_same_row_runs(runs, join_tolerance_cols)
        raw_segments.extend(seed_segments(runs))
    fitted = keep_segments(
        raw_segments, evidence_valid, settings, grid.row_pitch_mm, grid.column_pitch_mm
    )
    band_cols = max(6, int(np.ceil(settings.track_tolerance_mm * 3 / grid.column_pitch_mm)))
    links, track_number = link_segments(fitted, grid.plan.observed, settings, grid)
    splices, link_records = build_records(
        fitted, links, track_number, grid, evidence_valid, band_cols
    )
    splices, link_records, geometry = enforce_track_width(
        splices, link_records, fitted, track_number, settings.minimum_fraction
    )
    return SpliceReport(
        grid.stations, score, coverage, splices, link_records, tuple(geometry)
    )
