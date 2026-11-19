from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import scale_summary
from .numerics import bridge_short_holes, gaussian_reference, true_runs


MAX_STREAKS = 180


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
class StreakReport:
    residual: np.ndarray
    evidence: np.ndarray
    streaks: tuple
    detected_groups: int

    def summary(self):
        return {
            "detected_groups": self.detected_groups,
            "reported_streaks": len(self.streaks),
            "length_mm": scale_summary([streak.length_mm for streak in self.streaks]),
            "streak_contrast": scale_summary([streak.contrast for streak in self.streaks]),
            "truncated": self.detected_groups > len(self.streaks),
        }

    def table(self):
        return [streak.as_dict() for streak in self.streaks]


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
    return StreakReport(sustained, evidence, streaks, len(groups))
