from dataclasses import asdict, dataclass

import numpy as np

from .calibration import safe_value, scale_summary
from .numerics import segmented_reference, select_peaks


@dataclass(frozen=True)
class Splice:
    id: int
    station_mm: float
    polarity: str
    coverage: float
    contrast: float
    left_mm: float
    right_mm: float
    before_mm: float
    after_mm: float

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SpliceReport:
    stations: np.ndarray
    score: np.ndarray
    covered_fraction: np.ndarray
    splices: tuple

    def summary(self):
        return {
            "splice_candidates": len(self.splices),
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


def directional_votes(residual, valid, minimum_contrast, minimum_width):
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


def evidence_rows(eligible, signs, row):
    first = row
    last = row
    while first > 0 and eligible[first - 1] and signs[first - 1] == signs[row]:
        first -= 1
    while last + 1 < len(eligible) and eligible[last + 1] and signs[last + 1] == signs[row]:
        last += 1
    return first, last


def inspect_splices(grid, edges, settings):
    material = edges.inner_mask(3)
    sigma = (settings.local_scale_mm / grid.row_pitch_mm, 0.2)
    background, weights = segmented_reference(grid.values, material, sigma, 0.5)
    valid = material & (weights >= 0.5)
    residual = np.where(valid, grid.values - background, np.nan)
    minimum_width = max(6, grid.shape[1] // 6)
    strength, coverage, signs = directional_votes(
        residual, valid, settings.contrast, minimum_width
    )
    score = np.where(np.isfinite(coverage), strength * coverage, np.nan)
    eligible = np.isfinite(score) & (coverage >= settings.minimum_fraction)
    separation = max(1, int(np.ceil(settings.minimum_spacing_mm / grid.row_pitch_mm)))
    peaks = select_peaks(score, eligible, separation)
    splices = []
    for identifier, peak in enumerate(peaks, 1):
        row = peak.index
        first, last = evidence_rows(eligible, signs, row)
        signed = residual[row] * signs[row]
        hits = np.flatnonzero(valid[row] & (signed >= settings.contrast))
        splices.append(
            Splice(
                identifier,
                float(grid.stations[row]),
                "bright" if signs[row] > 0 else "dark",
                float(coverage[row]),
                float(strength[row]),
                float(grid.transverse[hits[0]]),
                float(grid.transverse[hits[-1]]),
                float(grid.stations[first]),
                float(grid.stations[last]),
            )
        )
    return SpliceReport(grid.stations, score, coverage, tuple(splices))
