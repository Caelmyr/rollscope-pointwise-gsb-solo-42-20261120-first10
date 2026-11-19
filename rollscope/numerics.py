from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def true_runs(mask):
    padded = np.empty(len(mask) + 2, dtype=np.int8)
    padded[0] = padded[-1] = 0
    padded[1:-1] = mask
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return zip(starts, ends)


def supported_slices(matrix, minimum_columns=1):
    candidate = np.count_nonzero(matrix, axis=1) >= minimum_columns
    return [slice(start, stop) for start, stop in true_runs(candidate)]


def bridge_short_holes(candidate, allowed, maximum_size):
    result = candidate.copy()
    if maximum_size == 0:
        return result

    for start, stop in true_runs(~candidate):
        inside = start > 0 and stop < len(candidate)
        if inside and stop - start <= maximum_size and np.all(allowed[start:stop]):
            result[start:stop] = True
    return result


def gaussian_reference(values, valid, sigma, minimum_support, maximum_radius=150):
    sigma = tuple(max(0.2, float(value)) for value in sigma)
    radius = tuple(min(maximum_radius, int(np.ceil(3 * value))) for value in sigma)
    numerator = ndimage.gaussian_filter(
        np.where(valid, values, 0).astype(np.float32),
        sigma=sigma,
        radius=radius,
        mode="constant",
        cval=0,
    )
    weights = ndimage.gaussian_filter(
        valid.astype(np.float32), sigma=sigma, radius=radius, mode="constant", cval=0
    )
    reference = np.full_like(numerator, np.nan)
    np.divide(numerator, weights, where=weights >= minimum_support, out=reference)
    return reference, weights


def segmented_reference(values, valid, sigma, minimum_support):
    reference = np.full_like(values, np.nan, dtype=np.float32)
    support = np.zeros_like(values, dtype=np.float32)
    for section in supported_slices(valid):
        reference[section], support[section] = gaussian_reference(
            values[section], valid[section], sigma, minimum_support
        )
    return reference, support


@dataclass(frozen=True)
class Peak:
    index: int
    prominence: float


def select_peaks(scores, valid, minimum_separation):
    positions = np.flatnonzero(valid & np.isfinite(scores))
    if not len(positions):
        return []
    candidates = []
    for position in positions:
        score = scores[position]
        lies_after_rise = position == 0 or not valid[position - 1] or scores[position - 1] < score
        lies_before_fall = (
            position == len(scores) - 1 or not valid[position + 1] or scores[position + 1] <= score
        )
        if lies_after_rise and lies_before_fall:
            candidates.append(Peak(int(position), float(score)))

    suppressed = np.zeros(scores.shape, dtype=bool)
    selected = []
    for candidate in sorted(candidates, key=lambda peak: (-peak.prominence, peak.index)):
        index = candidate.index
        if suppressed[index]:
            continue
        selected.append(candidate)
        start = max(0, index - minimum_separation + 1)
        stop = min(len(scores), index + minimum_separation)
        suppressed[start:stop] = True
    return sorted(selected, key=lambda peak: peak.index)


def bin_quantiles(values, valid, coordinate, block_points):
    output = []
    for start in range(0, len(values), block_points):
        stop = min(len(values), start + block_points)
        mask = valid[start:stop]
        observations = values[start:stop][mask]
        low, median, high = (None, None, None)
        if observations.size:
            low, median, high = map(float, np.percentile(observations, [10, 50, 90]))
        output.append(
            {
                "from_mm": float(coordinate[start]),
                "to_mm": float(coordinate[stop - 1]),
                "p10": low,
                "median": median,
                "p90": high,
                "sampled_fraction": float(np.count_nonzero(mask) / mask.size),
            }
        )
    return output
