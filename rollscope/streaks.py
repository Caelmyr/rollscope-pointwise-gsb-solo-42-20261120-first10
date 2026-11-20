from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import scale_summary
from .edges import EdgeState
from .grid import Support
from .numerics import bridge_short_holes, gaussian_reference, true_runs


MAX_STREAKS = 180
LINK_TOLERANCE_MM = 1.5
SPREAD_TOLERANCE_MM = 1.4
MOTION_MIN_MM = 0.6
MIN_EDGE_COVERAGE = 0.5


@dataclass(frozen=True)
class ColumnTrack:
    column: int
    first: int
    last: int
    polarity: str
    count: int
    amplitude: float

    @property
    def length(self):
        return self.last - self.first

    def overlap(self, other):
        return min(self.last, other.last) - max(self.first, other.first)


@dataclass
class TrackGroup:
    members: list

    @property
    def last_columns(self):
        return self.members[-1].column

    def merge_score(self, other):
        previous = self.members[-1]
        overlap = previous.overlap(other)
        if previous.polarity != other.polarity or other.column > previous.column + 1:
            return 0
        return overlap / max(previous.length, other.length)


@dataclass(frozen=True)
class Streak:
    id: int
    polarity: str
    from_mm: float
    to_mm: float
    left_mm: float
    right_mm: float
    length_mm: float
    center_mm: float
    evidence_fraction: float
    contrast: float

    def as_dict(self):
        return {
            "id": self.id,
            "polarity": self.polarity,
            "from_mm": self.from_mm,
            "to_mm": self.to_mm,
            "left_mm": self.left_mm,
            "right_mm": self.right_mm,
            "length_mm": self.length_mm,
            "center_mm": self.center_mm,
            "evidence_fraction": self.evidence_fraction,
            "contrast": self.contrast,
        }


@dataclass(frozen=True)
class StreakPath:
    id: int
    polarity: str
    alignment: str
    from_mm: float
    to_mm: float
    left_mm: float
    right_mm: float
    length_mm: float
    drift_mm: float
    strip_motion_mm: float
    edge_fraction: float
    evidence_fraction: float
    contrast: float
    members: tuple
    station_mm: float
    transverse_mm: float
    points: tuple

    def as_dict(self):
        return {
            "id": self.id,
            "polarity": self.polarity,
            "alignment": self.alignment,
            "from_mm": self.from_mm,
            "to_mm": self.to_mm,
            "left_mm": self.left_mm,
            "right_mm": self.right_mm,
            "length_mm": self.length_mm,
            "drift_mm": self.drift_mm,
            "strip_motion_mm": self.strip_motion_mm,
            "edge_fraction": self.edge_fraction,
            "evidence_fraction": self.evidence_fraction,
            "contrast": self.contrast,
            "members": list(self.members),
            "station_mm": self.station_mm,
            "transverse_mm": self.transverse_mm,
        }


@dataclass(frozen=True)
class StreakReport:
    residual: np.ndarray
    evidence: np.ndarray
    streaks: tuple
    detected_groups: int
    paths: tuple

    def summary(self):
        alignments = {"material": 0, "fixed": 0, "unclear": 0}
        for path in self.paths:
            alignments[path.alignment] += 1
        return {
            "detected_groups": self.detected_groups,
            "reported_streaks": len(self.streaks),
            "paths": alignments,
            "length_mm": scale_summary([streak.length_mm for streak in self.streaks]),
            "streak_contrast": scale_summary([streak.contrast for streak in self.streaks]),
            "truncated": self.detected_groups > len(self.streaks),
        }

    def table(self):
        return [streak.as_dict() for streak in self.streaks]

    def path_table(self):
        return [path.as_dict() for path in self.paths]


def transverse_residual(grid, edges, settings):
    guard = max(1, int(np.ceil(settings.background_mm / grid.column_pitch_mm / 2)))
    material = edges.inner_mask(guard)
    sigma = (0.2, settings.background_mm / grid.column_pitch_mm)
    backdrop, weights = gaussian_reference(grid.values, material, sigma, 0.48)
    evidence = material & (weights >= 0.48)
    residual = np.where(evidence, grid.values - backdrop, np.nan)
    return residual, evidence


def keep_sustained_lines(residual, evidence):
    kernel = np.ones(5, dtype=np.float32)
    count = ndimage.convolve1d(evidence.astype(np.float32), kernel, axis=0, mode="constant")
    total = ndimage.convolve1d(np.where(evidence, residual, 0), kernel, axis=0, mode="constant")
    final = np.full(residual.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=final, where=evidence & (count >= 4))
    return final


def column_tracks(contrast_line, evidence, pitch, settings, column):
    minimum_rows = max(2, int(np.ceil(settings.minimum_length_mm / pitch)))
    bridge_limit = max(1, int(min(6, np.ceil(2 / pitch))))
    for sign, polarity in ((-1, "dark"), (1, "bright")):
        signed = contrast_line * sign
        candidate = evidence & (signed >= settings.contrast)
        bridge = bridge_short_holes(candidate, evidence, bridge_limit)
        for first, last in true_runs(bridge):
            length = last - first
            if length < minimum_rows:
                continue
            actual = candidate[first:last]
            count = int(np.count_nonzero(actual))
            if count / length < settings.minimum_coverage:
                continue
            amplitude = float(np.median(signed[first:last][actual]))
            yield ColumnTrack(column, int(first), int(last), polarity, count, amplitude)


def assemble_tracks(residual, evidence, pitch, settings):
    groups = []
    active = []
    for column in range(residual.shape[1]):
        active = [group for group in active if group.last_columns >= column - 1]
        tracks = list(column_tracks(residual[:, column], evidence[:, column], pitch, settings, column))
        for track in tracks:
            eligible = [group for group in active if group.last_columns < column]
            best = max(eligible, key=lambda item: item.merge_score(track), default=None)
            if best is not None and best.merge_score(track) >= 0.58:
                best.members.append(track)
            else:
                created = TrackGroup([track])
                groups.append(created)
                active.append(created)
    return groups


def group_evidence(group, grid, identifier):
    first = min(track.first for track in group.members)
    last = max(track.last for track in group.members)
    left = group.members[0].column
    right = group.members[-1].column
    possible = sum(track.length for track in group.members)
    count = sum(track.count for track in group.members)
    amplitude = float(np.median([track.amplitude for track in group.members]))
    weights = np.array([track.count for track in group.members])
    coordinates = grid.transverse[[track.column for track in group.members]]
    return Streak(
        identifier,
        group.members[0].polarity,
        float(grid.stations[first]),
        float(grid.stations[last - 1]),
        float(grid.transverse[left]),
        float(grid.transverse[right]),
        float((last - first) * grid.row_pitch_mm),
        float(np.average(coordinates, weights=weights)),
        float(count / possible),
        amplitude,
    )


@dataclass(frozen=True)
class Anchor:
    row: int
    position: float
    fraction: float | None
    width: float | None


@dataclass(frozen=True)
class GroupShape:
    first: int
    last: int
    position: float
    fraction: float | None
    width: float | None
    start: Anchor
    end: Anchor


def track_anchor(track, row, grid, edges):
    position = float(grid.transverse[track.column])
    if edges.states[row] != EdgeState.COMPLETE:
        return Anchor(int(row), position, None, None)
    width = float(edges.width_mm[row])
    fraction = (position - float(edges.left_mm[row])) / width
    return Anchor(int(row), position, fraction, width)


def group_shape(group, grid, edges):
    first_track = min(group.members, key=lambda track: (track.first, track.column))
    last_track = max(group.members, key=lambda track: (track.last, track.column))
    weights = np.array([track.count for track in group.members], dtype=np.float64)
    columns = np.array([track.column for track in group.members])
    position = float(np.average(grid.transverse[columns], weights=weights))
    fractions = []
    widths = []
    for track in group.members:
        complete = edges.states[track.first : track.last] == EdgeState.COMPLETE
        if not complete.any():
            continue
        left = edges.left_mm[track.first : track.last][complete]
        width = edges.width_mm[track.first : track.last][complete]
        fractions.append((grid.transverse[track.column] - left) / width)
        widths.append(width)
    if fractions:
        fraction = float(np.median(np.concatenate(fractions)))
        width = float(np.median(np.concatenate(widths)))
    else:
        fraction = width = None
    return GroupShape(
        int(first_track.first),
        int(last_track.last),
        position,
        fraction,
        width,
        track_anchor(first_track, first_track.first, grid, edges),
        track_anchor(last_track, last_track.last - 1, grid, edges),
    )


def anchor_distance(first, second):
    sensor = abs(first.position - second.position)
    if first.fraction is None or second.fraction is None:
        return sensor
    width = 0.5 * (first.width + second.width)
    return min(sensor, abs(first.fraction - second.fraction) * width)


def shape_distance(first, second):
    sensor = abs(first.position - second.position)
    if first.fraction is None or second.fraction is None:
        return sensor
    width = 0.5 * (first.width + second.width)
    return min(sensor, abs(first.fraction - second.fraction) * width)


def parallel_match(first, second):
    overlap = min(first.last, second.last) - max(first.first, second.first)
    shorter = min(first.last - first.first, second.last - second.first)
    if shorter <= 0 or overlap < 0.6 * shorter:
        return False
    return shape_distance(first, second) <= LINK_TOLERANCE_MM


def sequential_match(first, second, support, max_gap_rows):
    for earlier, later in ((first, second), (second, first)):
        gap_rows = later.first - earlier.last
        if gap_rows < -2 or gap_rows > max_gap_rows:
            continue
        if np.any(support[earlier.last : later.first] == Support.GAP):
            continue
        if anchor_distance(earlier.end, later.start) <= LINK_TOLERANCE_MM:
            return True
    return False


def linked_components(groups, grid, edges, max_gap_rows):
    shapes = [group_shape(group, grid, edges) for group in groups]
    parent = list(range(len(groups)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    support = grid.plan.support
    for first_index, (group, shape) in enumerate(zip(groups, shapes)):
        polarity = group.members[0].polarity
        for other_index in range(first_index + 1, len(groups)):
            other = groups[other_index]
            if other.members[0].polarity != polarity:
                continue
            other_shape = shapes[other_index]
            if parallel_match(shape, other_shape) or sequential_match(
                shape, other_shape, support, max_gap_rows
            ):
                root, other_root = find(first_index), find(other_index)
                if root != other_root:
                    parent[other_root] = root
    components = {}
    for index, group in enumerate(groups):
        components.setdefault(find(index), []).append(group)
    return list(components.values())


def classify_alignment(sensor_spread, relative_spread, strip_travel, edge_coverage):
    if edge_coverage < MIN_EDGE_COVERAGE or strip_travel < MOTION_MIN_MM:
        return "unclear"
    if relative_spread is not None:
        if relative_spread <= SPREAD_TOLERANCE_MM and relative_spread <= 0.5 * strip_travel:
            return "material"
        if sensor_spread <= SPREAD_TOLERANCE_MM and relative_spread >= 0.5 * strip_travel:
            return "fixed"
    elif sensor_spread <= SPREAD_TOLERANCE_MM:
        return "fixed"
    return "unclear"


def path_positions(component, sustained, grid):
    polarity = component[0].members[0].polarity
    sign = 1 if polarity == "bright" else -1
    columns = np.array(
        sorted({track.column for group in component for track in group.members})
    )
    estimates = {}
    for group in component:
        for track in group.members:
            window = sustained[track.first : track.last][:, columns] * sign
            weights = np.where(np.isfinite(window), np.clip(window, 0, None), 0.0)
            totals = weights.sum(axis=1)
            rows = np.arange(track.first, track.last)
            usable = totals > 0
            centroid = np.full(len(rows), np.nan)
            centroid[usable] = (weights[usable] * columns).sum(axis=1) / totals[usable]
            for row, value in zip(rows[usable].tolist(), centroid[usable].tolist()):
                estimates.setdefault(row, []).append(value)
    return sorted(
        (row, float(np.mean(values))) for row, values in estimates.items()
    )


def build_path(component, identifier, member_ids, grid, edges, sustained):
    tracks = [track for group in component for track in group.members]
    tracks.sort(key=lambda track: (track.first, track.column))
    first = min(track.first for track in tracks)
    last = max(track.last for track in tracks)
    total_rows = sum(track.length for track in tracks)
    complete = edges.states == EdgeState.COMPLETE
    positions = path_positions(component, sustained, grid)
    if not positions:
        positions = [
            (track.first + offset, float(track.column))
            for track in tracks
            for offset in (0, track.length - 1)
        ]
    rows = np.array([row for row, _ in positions], dtype=np.int64)
    columns = np.array([column for _, column in positions])
    observed = grid.transverse[0] + columns * grid.column_pitch_mm
    covered_rows = sum(
        int(np.count_nonzero(complete[track.first : track.last])) for track in tracks
    )
    fraction_rows = []
    width_rows = []
    for row, position in zip(rows, observed):
        if complete[row]:
            fraction_rows.append((position - edges.left_mm[row]) / edges.width_mm[row])
            width_rows.append(edges.width_mm[row])
    relative = None
    if fraction_rows:
        pooled = np.asarray(fraction_rows)
        path_width = float(np.median(width_rows))
        low, high = np.percentile(pooled, [5, 95])
        relative = float((high - low) * path_width)
    center = edges.center_mm[first:last]
    known = center[np.isfinite(center)]
    travel = (
        float(np.subtract(*np.percentile(known, [95, 5]))) if known.size >= 2 else 0.0
    )
    sensor_spread = (
        float(np.subtract(*np.percentile(observed, [95, 5]))) if len(observed) >= 2 else 0.0
    )
    edge_fraction = covered_rows / total_rows if total_rows else 0.0
    alignment = classify_alignment(sensor_spread, relative, travel, edge_fraction)
    middle = positions[len(positions) // 2] if positions else (first, float(tracks[0].column))
    middle_mm = grid.transverse[0] + middle[1] * grid.column_pitch_mm
    points = tuple((int(row), float(column)) for row, column in positions)
    return StreakPath(
        identifier,
        component[0].members[0].polarity,
        alignment,
        float(grid.stations[first]),
        float(grid.stations[last - 1]),
        float(observed.min()),
        float(observed.max()),
        float((last - first) * grid.row_pitch_mm),
        sensor_spread,
        travel,
        float(edge_fraction),
        float(sum(track.count for track in tracks) / total_rows),
        float(np.median([track.amplitude for track in tracks])),
        member_ids,
        float(grid.stations[middle[0]]),
        float(middle_mm),
        points,
    )


def assemble_paths(groups, grid, edges, settings, sustained):
    if not groups:
        return ()
    max_gap_rows = max(1, int(np.ceil(settings.link_gap_mm / grid.row_pitch_mm)))
    components = linked_components(groups, grid, edges, max_gap_rows)
    components.sort(
        key=lambda component: (
            min(track.first for group in component for track in group.members),
            min(track.column for group in component for track in group.members),
        )
    )
    streak_ids = {id(group): index for index, group in enumerate(groups, 1)}
    return tuple(
        build_path(
            component,
            identifier,
            tuple(sorted(streak_ids[id(group)] for group in component)),
            grid,
            edges,
            sustained,
        )
        for identifier, component in enumerate(components, 1)
    )


def measure_streaks(grid, edges, settings):
    residual, evidence = transverse_residual(grid, edges, settings)
    sustained = keep_sustained_lines(residual, evidence)
    groups = assemble_tracks(sustained, evidence, grid.row_pitch_mm, settings)
    selected = sorted(
        groups,
        key=lambda group: -sum(track.length * track.amplitude for track in group.members),
    )[:MAX_STREAKS]
    selected.sort(key=lambda group: (group.members[0].first, group.members[0].column))
    streaks = tuple(
        group_evidence(group, grid, identifier) for identifier, group in enumerate(selected, 1)
    )
    paths = assemble_paths(selected, grid, edges, settings, sustained)
    return StreakReport(sustained, evidence, streaks, len(groups), paths)
